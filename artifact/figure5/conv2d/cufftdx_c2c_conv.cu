// Tiled 2D convolution with thread-level cuFFTDx 64-point C2C FFTs (FP32).
#include "common.cuh"

#include <cufftdx.hpp>

#ifndef THFF_CUFFTDX_SM
#error "THFF_CUFFTDX_SM must name the single compiled architecture, e.g. 800 or 900"
#endif

namespace figure5 {
namespace {

using namespace cufftdx;

constexpr int kTile = 64;
constexpr int kPitch = kTile + 1;

using complex_type = cufftdx::complex<float>;
using fft_base = decltype(Thread() + Precision<float>() + SM<THFF_CUFFTDX_SM>() + Size<kTile>() + Type<fft_type::c2c>());
using fft_forward = decltype(fft_base() + Direction<fft_direction::forward>());
using fft_inverse = decltype(fft_base() + Direction<fft_direction::inverse>());

constexpr size_t kSmemBytes = sizeof(complex_type) * kTile * kPitch;

__launch_bounds__(kTile) __global__ void conv_kernel(const float* __restrict__ in, const complex_type* __restrict__ ht,
                                                    float* __restrict__ out, int N, int M, int S) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    complex_type* tile = reinterpret_cast<complex_type*>(smem_raw);
    complex_type d[fft_forward::storage_size];
    const int t = threadIdx.x;
    const int row0 = blockIdx.y * S, col0 = blockIdx.x * S;

    const int gx = col0 + t;
#pragma unroll
    for (int y = 0; y < kTile; ++y) {
        const int gy = row0 + y;
        d[y] = complex_type(gy < N && gx < N ? in[size_t(gy) * N + gx] : 0.0f, 0.0f);
    }
    fft_forward().execute(d);
#pragma unroll
    for (int ky = 0; ky < kTile; ++ky) tile[ky * kPitch + t] = d[ky];
    __syncthreads();

#pragma unroll
    for (int x = 0; x < kTile; ++x) d[x] = tile[t * kPitch + x];
    fft_forward().execute(d);
#pragma unroll
    for (int kx = 0; kx < kTile; ++kx) d[kx] = d[kx] * ht[kx * kTile + t];
    fft_inverse().execute(d);
#pragma unroll
    for (int x = 0; x < kTile; ++x) tile[t * kPitch + x] = d[x];
    __syncthreads();

#pragma unroll
    for (int ky = 0; ky < kTile; ++ky) d[ky] = tile[ky * kPitch + t];
    fft_inverse().execute(d);
#pragma unroll
    for (int y = 0; y < kTile; ++y) asm volatile("" ::"f"(d[y].x), "f"(d[y].y));
    if (t < S && col0 + t < M) {
#pragma unroll
        for (int y = 0; y < kTile; ++y)
            if (y < S && row0 + y < M) out[size_t(row0 + y) * M + col0 + t] = d[y].x;
    }
}

__global__ void pad_filter(const float* __restrict__ filter, float2* __restrict__ pad, int f) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < kTile * kTile; k += gridDim.x * blockDim.x) {
        const int r = k / kTile, c = k % kTile, i = (kTile - r) % kTile, j = (kTile - c) % kTile;
        pad[k] = make_float2(i < f && j < f ? filter[i * f + j] * (1.0f / (kTile * kTile)) : 0.0f, 0.0f);
    }
}

__global__ void transpose_spectrum(const float2* __restrict__ H, float2* __restrict__ ht) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < kTile * kTile; k += gridDim.x * blockDim.x)
        ht[(k % kTile) * kTile + k / kTile] = H[k];
}

class CufftdxMethod final : public Method {
public:
    explicit CufftdxMethod(const Problem& p)
        : problem_(p), padded_(kTile * kTile), spectrum_(kTile * kTile), ht_(kTile * kTile), output_(size_t(p.M) * p.M) {
        if (p.f < 1 || p.f > kTile) throw std::runtime_error("cuFFTDx tile requires 1 <= f <= 64");
        stride_ = kTile - p.f + 1;
        tiles_ = (p.M + stride_ - 1) / stride_;
        FIG5_CUDA(cudaFuncSetAttribute(conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmemBytes)));
        FIG5_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_, conv_kernel, kTile, kSmemBytes));
        FIG5_CUFFT(cufftPlan2d(&plan_, kTile, kTile, CUFFT_C2C));
        FIG5_CUFFT(cufftSetStream(plan_, p.stream));
    }
    ~CufftdxMethod() override { cufftDestroy(plan_); }
    void run() override {
        const Problem& p = problem_;
        pad_filter<<<16, 256, 0, p.stream>>>(p.d_filter, padded_.data, p.f);
        FIG5_CUFFT(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(padded_.data),
                                reinterpret_cast<cufftComplex*>(spectrum_.data), CUFFT_FORWARD));
        transpose_spectrum<<<16, 256, 0, p.stream>>>(spectrum_.data, ht_.data);
        conv_kernel<<<dim3(tiles_, tiles_), kTile, kSmemBytes, p.stream>>>(
            p.d_input, reinterpret_cast<const complex_type*>(ht_.data), output_.data, p.N, p.M, stride_);
        FIG5_CUDA(cudaGetLastError());
    }
    const float* output() const override { return output_.data; }
    size_t output_pitch() const override { return size_t(problem_.M); }
    bool filter_fft_included() const override { return true; }
    std::string details() const override {
        return "{\"cufftdx_version\":" + std::to_string(CUFFTDX_VERSION) + ",\"sm\":" + std::to_string(THFF_CUFFTDX_SM) +
               ",\"execution\":\"Thread\",\"type\":\"c2c\",\"tile\":64,\"tiles_per_dim\":" + std::to_string(tiles_) +
               ",\"block_threads\":64,\"blocks_per_sm\":" + std::to_string(blocks_per_sm_) +
               ",\"smem_bytes\":" + std::to_string(kSmemBytes) + "}";
    }

private:
    Problem problem_;
    DeviceBuffer<float2> padded_, spectrum_, ht_;
    DeviceBuffer<float> output_;
    cufftHandle plan_ = 0;
    int stride_ = 0, tiles_ = 0, blocks_per_sm_ = 0;
};

}  // namespace

std::unique_ptr<Method> make_cufftdx_c2c(const Problem& p) { return std::make_unique<CufftdxMethod>(p); }

}  // namespace figure5
