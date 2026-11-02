#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace {

#define CHECK_CUDA(x)                                                                 \
    do {                                                                              \
        cudaError_t err__ = (x);                                                      \
        if (err__ != cudaSuccess) {                                                   \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                  \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";              \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

#define CHECK_CUDNN(x)                                                                \
    do {                                                                              \
        cudnnStatus_t st__ = (x);                                                     \
        if (st__ != CUDNN_STATUS_SUCCESS) {                                           \
            std::cerr << "cuDNN error: " << cudnnGetErrorString(st__)                 \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";              \
            std::exit(EXIT_FAILURE);                                                  \
        }                                                                             \
    } while (0)

constexpr int kFlashBlockX = 256;
constexpr int kFlashTileL = 4;

template <typename T>
__device__ __forceinline__ float to_float(T x);
template <>
__device__ __forceinline__ float to_float<half>(half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ half from_float<half>(float x) { return __float2half_rn(x); }

template <typename T>
__global__ void flash_depthwise_conv1d_bdl_kernel(const T* __restrict__ u,
                                                  const T* __restrict__ w,
                                                  const T* __restrict__ b,
                                                  T* __restrict__ out,
                                                  int B,
                                                  int D,
                                                  int L,
                                                  int K,
                                                  int Lout) {
    const int bb = blockIdx.z;
    const int dd = blockIdx.y;
    const int l0 = (int(blockIdx.x) * blockDim.x * kFlashTileL) + int(threadIdx.x);
    if (bb >= B || dd >= D) return;

    const int in_base = (bb * D + dd) * L;
    const int w_base = dd * K;
    const int out_base = (bb * D + dd) * Lout;
    const float bias = to_float<T>(b[dd]);

    #pragma unroll
    for (int t = 0; t < kFlashTileL; ++t) {
        const int l = l0 + t * blockDim.x;
        if (l >= Lout) continue;
        float acc = bias;
        #pragma unroll 1
        for (int k = 0; k < K; ++k) {
            acc = fmaf(to_float<T>(u[in_base + l + k]), to_float<T>(w[w_base + k]), acc);
        }
        out[out_base + l] = from_float<T>(acc);
    }
}

float measure_cuda_ms(void (*fn)(void*), void* ctx, int warmup, int runs) {
    cudaEvent_t st, ed;
    CHECK_CUDA(cudaEventCreate(&st));
    CHECK_CUDA(cudaEventCreate(&ed));
    for (int i = 0; i < warmup; ++i) fn(ctx);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(st));
    for (int i = 0; i < runs; ++i) fn(ctx);
    CHECK_CUDA(cudaEventRecord(ed));
    CHECK_CUDA(cudaEventSynchronize(ed));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, st, ed));
    CHECK_CUDA(cudaEventDestroy(st));
    CHECK_CUDA(cudaEventDestroy(ed));
    return ms / runs;
}

struct FlashCtx {
    const half* u;
    const half* w;
    const half* b;
    half* out;
    int B, D, L, K, Lout;
    dim3 grid, block;
};

void launch_flash(void* p) {
    auto* c = reinterpret_cast<FlashCtx*>(p);
    flash_depthwise_conv1d_bdl_kernel<half><<<c->grid, c->block>>>(
        c->u, c->w, c->b, c->out, c->B, c->D, c->L, c->K, c->Lout);
    CHECK_CUDA(cudaGetLastError());
}

struct CudnnCtx {
    cudnnHandle_t h{};
    cudnnTensorDescriptor_t xDesc{}, yDesc{}, biasDesc{};
    cudnnFilterDescriptor_t wDesc{};
    cudnnConvolutionDescriptor_t convDesc{};
    cudnnConvolutionFwdAlgo_t algo{};
    const half* x;
    const half* w;
    const half* b;
    half* y;
    void* ws{};
    size_t ws_bytes{};
};

void launch_cudnn(void* p) {
    auto* c = reinterpret_cast<CudnnCtx*>(p);
    constexpr float alpha = 1.0f;
    constexpr float beta0 = 0.0f;
    constexpr float beta1 = 1.0f;
    CHECK_CUDNN(cudnnConvolutionForward(c->h, &alpha, c->xDesc, c->x, c->wDesc, c->w,
                                        c->convDesc, c->algo, c->ws, c->ws_bytes,
                                        &beta0, c->yDesc, c->y));
    CHECK_CUDNN(cudnnAddTensor(c->h, &alpha, c->biasDesc, c->b, &beta1, c->yDesc, c->y));
}

float run_thunder_ms(const std::string& thff_bin, int L, int K, int device, int warmup, int runs) {
    std::ostringstream cmd;
    cmd << thff_bin << " " << L << " " << K << " " << device << " 0 "
        << warmup << " " << runs;
    FILE* fp = popen(cmd.str().c_str(), "r");
    if (!fp) {
        std::cerr << "failed to run: " << cmd.str() << "\n";
        std::exit(EXIT_FAILURE);
    }
    std::string text;
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp) != nullptr) text += buf;
    pclose(fp);
    std::regex re("Kernel pipeline time: ([0-9]+\\.?[0-9]*) ms");
    std::smatch m;
    if (!std::regex_search(text, m, re)) {
        std::cerr << "failed to parse thunder output\n" << text << "\n";
        std::exit(EXIT_FAILURE);
    }
    return std::stof(m[1].str());
}

}  // namespace

int main(int argc, char** argv) {
    int L = 16384 * 16384;
    int Kb = 3;
    int Ke = 63;
    int Ks = 2;
    int device = 0;
    int B = 1;
    int D = 1;
    int warmup = 3;
    int runs = 20;
    std::string csv = "/home/chaewon/fft/thunderfft/sample/conv1d/compare_flash_cudnn_thunder.csv";
    std::string thff_bin = "/home/chaewon/fft/thunderfft/build/sample/conv1d/thff_conv1d";

    if (argc >= 2) L = std::atoi(argv[1]);
    if (argc >= 3) Kb = std::atoi(argv[2]);
    if (argc >= 4) Ke = std::atoi(argv[3]);
    if (argc >= 5) Ks = std::atoi(argv[4]);
    if (argc >= 6) device = std::atoi(argv[5]);
    if (argc >= 7) B = std::atoi(argv[6]);
    if (argc >= 8) D = std::atoi(argv[7]);
    if (argc >= 9) warmup = std::atoi(argv[8]);
    if (argc >= 10) runs = std::atoi(argv[9]);
    if (argc >= 11) csv = argv[10];
    if (argc >= 12) thff_bin = argv[11];

    if (Kb < 1 || Ke < Kb || Ks < 1) {
        std::cerr << "invalid K range\n";
        return EXIT_FAILURE;
    }
    CHECK_CUDA(cudaSetDevice(device));

    std::ofstream ofs(csv);
    if (!ofs) {
        std::cerr << "failed to open csv: " << csv << "\n";
        return EXIT_FAILURE;
    }
    ofs << "method,L,B,D,K,output_len,outputs_total,time_ms,ns_per_output\n";

    cudnnHandle_t h{};
    CHECK_CUDNN(cudnnCreate(&h));

    for (int K = Kb; K <= Ke; K += Ks) {
        const int Lout = L - K + 1;
        if (Lout <= 0) continue;
        const size_t x_elems = size_t(B) * D * L;
        const size_t y_elems = size_t(B) * D * Lout;
        const size_t w_elems = size_t(D) * K;
        const size_t b_elems = size_t(D);

        half *d_x = nullptr, *d_w = nullptr, *d_b = nullptr, *d_y = nullptr;
        CHECK_CUDA(cudaMalloc(&d_x, x_elems * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_w, w_elems * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_b, b_elems * sizeof(half)));
        CHECK_CUDA(cudaMalloc(&d_y, y_elems * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_x, 0, x_elems * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_w, 0, w_elems * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_b, 0, b_elems * sizeof(half)));
        CHECK_CUDA(cudaMemset(d_y, 0, y_elems * sizeof(half)));

        FlashCtx fctx{};
        fctx.u = d_x; fctx.w = d_w; fctx.b = d_b; fctx.out = d_y;
        fctx.B = B; fctx.D = D; fctx.L = L; fctx.K = K; fctx.Lout = Lout;
        fctx.block = dim3(kFlashBlockX, 1, 1);
        fctx.grid = dim3((Lout + (kFlashBlockX * kFlashTileL) - 1) / (kFlashBlockX * kFlashTileL), D, B);
        const float flash_ms = measure_cuda_ms(launch_flash, &fctx, warmup, runs);
        const double flash_ns = (flash_ms * 1.0e6) / double(y_elems);

        CudnnCtx cctx{};
        cctx.h = h;
        cctx.x = d_x; cctx.w = d_w; cctx.b = d_b; cctx.y = d_y;
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&cctx.xDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&cctx.yDesc));
        CHECK_CUDNN(cudnnCreateTensorDescriptor(&cctx.biasDesc));
        CHECK_CUDNN(cudnnCreateFilterDescriptor(&cctx.wDesc));
        CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&cctx.convDesc));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(cctx.xDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, B, D, 1, L));
        CHECK_CUDNN(cudnnSetFilter4dDescriptor(cctx.wDesc, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, D, 1, 1, K));
        CHECK_CUDNN(cudnnSetConvolution2dDescriptor(cctx.convDesc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
        CHECK_CUDNN(cudnnSetConvolutionGroupCount(cctx.convDesc, D));
        CHECK_CUDNN(cudnnSetConvolutionMathType(cctx.convDesc, CUDNN_TENSOR_OP_MATH));
        int n, c, h2, w2;
        CHECK_CUDNN(cudnnGetConvolution2dForwardOutputDim(cctx.convDesc, cctx.xDesc, cctx.wDesc, &n, &c, &h2, &w2));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(cctx.yDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n, c, h2, w2));
        CHECK_CUDNN(cudnnSetTensor4dDescriptor(cctx.biasDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, 1, D, 1, 1));

        int ret = 0;
        cudnnConvolutionFwdAlgoPerf_t perf[8];
        CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cctx.h, cctx.xDesc, cctx.wDesc, cctx.convDesc, cctx.yDesc, 8, &ret, perf));
        cctx.algo = perf[0].algo;
        for (int i = 0; i < ret; ++i) {
            if (perf[i].status == CUDNN_STATUS_SUCCESS) {
                cctx.algo = perf[i].algo;
                break;
            }
        }
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cctx.h, cctx.xDesc, cctx.wDesc, cctx.convDesc, cctx.yDesc, cctx.algo, &cctx.ws_bytes));
        if (cctx.ws_bytes > 0) CHECK_CUDA(cudaMalloc(&cctx.ws, cctx.ws_bytes));
        const float cudnn_ms = measure_cuda_ms(launch_cudnn, &cctx, warmup, runs);
        const double cudnn_ns = (cudnn_ms * 1.0e6) / double(y_elems);

        const float thunder_ms = run_thunder_ms(thff_bin, L, K, device, warmup, runs);
        const int thunder_out = L - K + 1;
        const double thunder_ns = (thunder_ms * 1.0e6) / double(thunder_out);

        ofs << "flash_conv1d_cuda," << L << "," << B << "," << D << "," << K << ","
            << Lout << "," << y_elems << "," << flash_ms << "," << flash_ns << "\n";
        ofs << "cudnn_depthwise," << L << "," << B << "," << D << "," << K << ","
            << Lout << "," << y_elems << "," << cudnn_ms << "," << cudnn_ns << "\n";
        ofs << "thunderfft_conv1d," << L << "," << 1 << "," << 1 << "," << K << ","
            << thunder_out << "," << thunder_out << "," << thunder_ms << "," << thunder_ns << "\n";

        std::cout << "K=" << K
                  << " flash=" << flash_ms << "ms"
                  << " cudnn=" << cudnn_ms << "ms"
                  << " thunder=" << thunder_ms << "ms\n";

        if (cctx.ws) CHECK_CUDA(cudaFree(cctx.ws));
        CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(cctx.convDesc));
        CHECK_CUDNN(cudnnDestroyFilterDescriptor(cctx.wDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(cctx.biasDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(cctx.yDesc));
        CHECK_CUDNN(cudnnDestroyTensorDescriptor(cctx.xDesc));
        CHECK_CUDA(cudaFree(d_x));
        CHECK_CUDA(cudaFree(d_w));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_y));
    }

    CHECK_CUDNN(cudnnDestroy(h));
    std::cout << "saved csv: " << csv << "\n";
    return 0;
}
