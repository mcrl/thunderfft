// Figure 5 artifact implementation: fused 64x64 tiled complex cross-correlation.
// Uses the current public N=64 TF32 kernels, not a recovered paper binary.
#include <thunderfft/thunderfft.cuh>
#include "../utils.h"
#include <functional>
#include <vector>

namespace {
constexpr int kTile = 64, kStride = 65;
using Rows = thunderfft::layout_t<64, 16, 1, kStride, 1, 0, false>;
using Cols = thunderfft::layout_t<64, 16, kStride, 1, 1, 0, false>;

template <typename Layout, bool Forward>
__device__ __forceinline__ void transform(float2* base) {
    float2 reg[32];
    thunderfft::ThunderFFT_smem2reg<float, Layout>(reg, base);
    __syncwarp();
    thunderfft::ThunderFFT_kernel_reg<float, 64, 16, Forward>(reg, nullptr, nullptr);
    thunderfft::ThunderFFT_reg2smem<float, Layout>(base, reg);
}

__global__ __launch_bounds__(128) void tiled64(const float2* x, const float2* filter, float2* y, int n, int f) {
    __shared__ float2 tile[kTile * kStride];
    const int valid = kTile - f + 1, out = n - f + 1;
    const int y0 = int(blockIdx.y) * valid, x0 = int(blockIdx.x) * valid;
    const int warp = threadIdx.x / 32;
    for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x) {
        const int r = i / kTile, c = i % kTile;
        tile[r * kStride + c] = y0 + r < n && x0 + c < n ? x[size_t(y0+r)*n+x0+c] : make_float2(0, 0);
    }
    __syncthreads();
    transform<Rows, true>(tile + warp * 16 * kStride);
    __syncthreads();
    transform<Cols, true>(tile + warp * 16);
    __syncthreads();
    for (int i = threadIdx.x; i < kTile * kTile; i += blockDim.x) {
        const int r = i / kTile, c = i % kTile;
        const float2 a = tile[r * kStride + c], b = filter[i];
        tile[r * kStride + c] = make_float2(a.x*b.x-a.y*b.y, a.x*b.y+a.y*b.x);
    }
    __syncthreads();
    transform<Cols, false>(tile + warp * 16);
    __syncthreads();
    transform<Rows, false>(tile + warp * 16 * kStride);
    __syncthreads();
    for (int i = threadIdx.x; i < valid * valid; i += blockDim.x) {
        const int r = i / valid, c = i % valid;
        if (y0+r < out && x0+c < out) y[size_t(y0+r)*out+x0+c] = tile[r*kStride+c];
    }
}
}

std::vector<float> thunder_tiled64_convolution(const float2* input, const float2* filter, int n, int f,
                                              const std::function<bool(const float2*)>& gate,
                                              int warmup, int runs, int trials) {
    const int out = n-f+1, valid = kTile-f+1;
    float2 *dx=nullptr, *dh=nullptr, *df=nullptr, *dy=nullptr;
    CHECK_CUDA(cudaMalloc(&dx, sizeof(float2)*size_t(n)*n));
    CHECK_CUDA(cudaMalloc(&dh, sizeof(float2)*kTile*kTile));
    CHECK_CUDA(cudaMalloc(&df, sizeof(float2)*kTile*kTile));
    CHECK_CUDA(cudaMalloc(&dy, sizeof(float2)*size_t(out)*out));
    CHECK_CUDA(cudaMemcpy(dx, input, sizeof(float2)*size_t(n)*n, cudaMemcpyHostToDevice));
    std::vector<float2> padded(kTile*kTile, make_float2(0, 0));
    for (int r=0; r<f; ++r) for (int c=0; c<f; ++c) {
        const float2 v=filter[r*f+c];
        padded[((kTile-r)%kTile)*kTile+(kTile-c)%kTile]=make_float2(v.x/(kTile*kTile),v.y/(kTile*kTile));
    }
    CHECK_CUDA(cudaMemcpy(dh, padded.data(), sizeof(float2)*padded.size(), cudaMemcpyHostToDevice));
    // NaNs expose missing output writes to the reference gate.
    CHECK_CUDA(cudaMemset(dy, 0xff, sizeof(float2)*size_t(out)*out));
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, kTile, kTile, CUFFT_C2C));
    auto run = [&]() {
        CHECK_CUFFT(cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dh), reinterpret_cast<cufftComplex*>(df), CUFFT_FORWARD));
        tiled64<<<dim3((out+valid-1)/valid,(out+valid-1)/valid),128>>>(dx,df,dy,n,f);
        CHECK_CUDA(cudaGetLastError());
    };
    run();
    std::vector<float2> actual(size_t(out)*out);
    CHECK_CUDA(cudaMemcpy(actual.data(),dy,sizeof(float2)*actual.size(),cudaMemcpyDeviceToHost));
    std::vector<float> times;
    if (gate(actual.data())) {
        const unsigned count=adaptive_runs(run,runs);
        for (int t=0;t<trials;++t) times.push_back(measure_execution_ms(run,t==0?std::min<unsigned>(warmup,(count+2)/3):0,count));
    }
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(dx)); CHECK_CUDA(cudaFree(dh)); CHECK_CUDA(cudaFree(df)); CHECK_CUDA(cudaFree(dy));
    return times;
}
