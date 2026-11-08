// Tiled 2D convolution on the ThunderFFT device API (64-point TF32 Tensor Core FFTs).
#include "common.cuh"

#include <thunderfft/thunderfft.cuh>

#include <cuda_pipeline.h>

#include <algorithm>
#include <string>

namespace figure5 {
namespace {

constexpr int kTile = 64;
constexpr int kStagePitch = 72;
constexpr int kStageFloats = kTile * kStagePitch;
constexpr int kWorkspace = 63 * 68 + 63 + 3 + 12 + 1;
constexpr size_t kSmemBytes = sizeof(float) * kStageFloats + sizeof(float2) * kWorkspace;

struct Geometry {
    int N, M, S, tiles_x, tiles;
};

__device__ __forceinline__ int stage_index(int row, int col) { return row * kStagePitch + col + 4 * (col >> 5); }
__device__ __forceinline__ int ws_index(int row, int col) { return row * 68 + col + (col >> 4) + 4 * (row >> 4); }
__device__ __forceinline__ int out_index(int row, int col) {
    return row * kTile + (col & 32) + ((col & 31) ^ (((row & 7) << 1) | (col >> 5)));
}
__device__ __forceinline__ int rev2(int i) { return ((i & 3) << 2) | (i >> 2); }

__device__ __forceinline__ void rev2_permute(float2* r) {
#define FIG5_SWAP(a, b) { const float2 t_ = r[a]; r[a] = r[b]; r[b] = t_; }
    FIG5_SWAP(1, 4) FIG5_SWAP(2, 8) FIG5_SWAP(3, 12) FIG5_SWAP(6, 9) FIG5_SWAP(7, 13) FIG5_SWAP(11, 14)
    FIG5_SWAP(17, 20) FIG5_SWAP(18, 24) FIG5_SWAP(19, 28) FIG5_SWAP(22, 25) FIG5_SWAP(23, 29) FIG5_SWAP(27, 30)
#undef FIG5_SWAP
}

__device__ __forceinline__ void load_tile(float* stage, const float* __restrict__ in, const Geometry& g, int tile, int tid) {
    const int c2 = tid & 31, row0 = tid >> 5;
    const int ty = tile / g.tiles_x, tx = tile - ty * g.tiles_x;
    const int gy = ty * g.S + row0, gx = tx * g.S + 2 * c2;
    const int rows = gx < g.N ? (g.N - gy + 3) >> 2 : 0;
    const float* src = in + size_t(gy) * g.N + gx;
    float* dst = stage + stage_index(row0, 2 * c2);
#pragma unroll
    for (int k = 0; k < kTile / 4; ++k) {
        float* d = dst + 4 * k * kStagePitch;
        if (k < rows) __pipeline_memcpy_async(d, src + size_t(4 * k) * g.N, sizeof(float2));
        else *reinterpret_cast<float2*>(d) = make_float2(0.0f, 0.0f);
    }
    __pipeline_commit();
}

__global__ __launch_bounds__(128) void conv_kernel(const float* __restrict__ in, const float2* __restrict__ hfrag,
                                                   float* __restrict__ out, Geometry g) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float* stage = reinterpret_cast<float*>(smem_raw);
    float2* ws = reinterpret_cast<float2*>(smem_raw + sizeof(float) * kStageFloats);
    const int lane = threadIdx.x, w = threadIdx.y, tid = lane + 32 * w;
    const int q = lane & 3, r4 = lane >> 2;
    const int r0 = 16 * w + r4, r1 = r0 + 8;
    float2 reg[32];

    int tile = blockIdx.x;
    if (tile < g.tiles) load_tile(stage, in, g, tile, tid);
    for (; tile < g.tiles; tile += gridDim.x) {
        __pipeline_wait_prior(0);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int col = 16 * q + rev2(i);
            reg[i] = make_float2(stage[stage_index(r0, col)], 0.0f);
            reg[i + 16] = make_float2(stage[stage_index(r1, col)], 0.0f);
        }
        __syncthreads();
        if (tile + int(gridDim.x) < g.tiles) load_tile(stage, in, g, tile + gridDim.x, tid);

        thunderfft::ThunderFFT_kernel_reg<float, 64, 16, true>(reg, nullptr, nullptr);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ws[ws_index(r0, 16 * q + i)] = reg[i];
            ws[ws_index(r1, 16 * q + i)] = reg[i + 16];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row = 16 * q + rev2(i);
            reg[i] = ws[ws_index(row, r0)];
            reg[i + 16] = ws[ws_index(row, r1)];
        }
        thunderfft::ThunderFFT_kernel_reg<float, 64, 16, true>(reg, nullptr, nullptr);

        const float2* h = hfrag + (w * 32) * 32 + lane;
#pragma unroll
        for (int i = 0; i < 32; ++i) {
            const float2 a = reg[i], b = __ldg(h + i * 32);
            reg[i] = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
        }
        rev2_permute(reg);
        thunderfft::ThunderFFT_kernel_reg<float, 64, 16, false>(reg, nullptr, nullptr);
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            ws[ws_index(16 * q + i, r0)] = reg[i];
            ws[ws_index(16 * q + i, r1)] = reg[i + 16];
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int col = 16 * q + rev2(i);
            reg[i] = ws[ws_index(r0, col)];
            reg[i + 16] = ws[ws_index(r1, col)];
        }
        thunderfft::ThunderFFT_kernel_reg<float, 64, 16, false>(reg, nullptr, nullptr);

        if (16 * w < g.S) {
            float* buf = reinterpret_cast<float*>(ws + ws_index(16 * w, 0));
            __syncwarp();
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                buf[out_index(r4, 16 * q + i)] = reg[i].x;
                buf[out_index(r4 + 8, 16 * q + i)] = reg[i + 16].x;
            }
            __syncwarp();
            const int ty = tile / g.tiles_x, tx = tile - ty * g.tiles_x;
            const int rows = min(16, min(g.S - 16 * w, g.M - (ty * g.S + 16 * w)));
            const int cols = min(g.S, g.M - tx * g.S);
            float* dst = out + size_t(ty * g.S + 16 * w) * g.M + size_t(tx) * g.S;
#pragma unroll
            for (int rr = 0; rr < 16; ++rr) {
                if (rr < rows) {
#pragma unroll
                    for (int c = 0; c < 2; ++c) {
                        const int x = lane + 32 * c;
                        if (x < cols) dst[size_t(rr) * g.M + x] = buf[out_index(rr, x)];
                    }
                }
            }
        }
    }
}

__global__ void pad_filter(const float* __restrict__ filter, float2* __restrict__ pad, int f) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < kTile * kTile; k += gridDim.x * blockDim.x) {
        const int r = k / kTile, c = k % kTile, i = (kTile - r) % kTile, j = (kTile - c) % kTile;
        pad[k] = make_float2(i < f && j < f ? filter[i * f + j] * (1.0f / (kTile * kTile)) : 0.0f, 0.0f);
    }
}

__global__ void permute_spectrum(const float2* __restrict__ H, float2* __restrict__ hfrag) {
    for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < kTile * kTile; k += gridDim.x * blockDim.x) {
        const int lane = k % 32, i = (k / 32) % 32, w = k / 1024;
        const int ky = 16 * (lane & 3) + (i % 16), kx = 16 * w + (lane >> 2) + 8 * (i / 16);
        hfrag[k] = H[ky * kTile + kx];
    }
}

class ThunderFFTMethod final : public Method {
public:
    explicit ThunderFFTMethod(const Problem& p)
        : problem_(p), padded_(kTile * kTile), spectrum_(kTile * kTile), hfrag_(kTile * kTile), output_(size_t(p.M) * p.M) {
        if (p.f < 1 || p.f > kTile) throw std::runtime_error("ThunderFFT tile requires 1 <= f <= 64");
        if (p.N % 2) throw std::runtime_error("ThunderFFT tile requires an even N");
        geometry_.N = p.N;
        geometry_.M = p.M;
        geometry_.S = kTile - p.f + 1;
        geometry_.tiles_x = (p.M + geometry_.S - 1) / geometry_.S;
        geometry_.tiles = geometry_.tiles_x * geometry_.tiles_x;
        FIG5_CUDA(cudaFuncSetAttribute(conv_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kSmemBytes)));
        int device = 0, sms = 0;
        FIG5_CUDA(cudaGetDevice(&device));
        FIG5_CUDA(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
        FIG5_CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm_, conv_kernel, 128, kSmemBytes));
        if (blocks_per_sm_ < 1) throw std::runtime_error("ThunderFFT tile kernel cannot be resident");
        grid_ = std::min(geometry_.tiles, sms * blocks_per_sm_);
        FIG5_CUFFT(cufftPlan2d(&plan_, kTile, kTile, CUFFT_C2C));
        FIG5_CUFFT(cufftSetStream(plan_, p.stream));
    }
    ~ThunderFFTMethod() override { cufftDestroy(plan_); }
    void run() override {
        const Problem& p = problem_;
        pad_filter<<<16, 256, 0, p.stream>>>(p.d_filter, padded_.data, p.f);
        FIG5_CUFFT(cufftExecC2C(plan_, reinterpret_cast<cufftComplex*>(padded_.data),
                                reinterpret_cast<cufftComplex*>(spectrum_.data), CUFFT_FORWARD));
        permute_spectrum<<<16, 256, 0, p.stream>>>(spectrum_.data, hfrag_.data);
        conv_kernel<<<grid_, dim3(32, 4), kSmemBytes, p.stream>>>(p.d_input, hfrag_.data, output_.data, geometry_);
        FIG5_CUDA(cudaGetLastError());
    }
    const float* output() const override { return output_.data; }
    size_t output_pitch() const override { return size_t(problem_.M); }
    bool filter_fft_included() const override { return true; }
    std::string details() const override {
        return "{\"tile\":64,\"tile_stride\":" + std::to_string(geometry_.S) + ",\"tiles\":" + std::to_string(geometry_.tiles) +
               ",\"grid\":" + std::to_string(grid_) + ",\"blocks_per_sm\":" + std::to_string(blocks_per_sm_) +
               ",\"block\":[32,4],\"smem_bytes\":" + std::to_string(kSmemBytes) + "}";
    }

private:
    Problem problem_;
    Geometry geometry_{};
    DeviceBuffer<float2> padded_, spectrum_, hfrag_;
    DeviceBuffer<float> output_;
    cufftHandle plan_ = 0;
    int grid_ = 0, blocks_per_sm_ = 0;
};

}  // namespace

std::unique_ptr<Method> make_thunderfft_c2c(const Problem& p) { return std::make_unique<ThunderFFTMethod>(p); }

}  // namespace figure5
