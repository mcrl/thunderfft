// Figure 5 driver: one 2D convolution library per run, correctness gate before timing.
//
// Operation: valid 2D cross-correlation (complex FP32 for cuFFTDx; real parts for cuDNN/cuFFT)
//   y[i][j] = sum_{a,b < f} x[i+a][j+b] * h[a][b],  0 <= i, j < N-f+1,
// with x (N x N) and h (f x f) row-major, generated from seed 42 with U[0,1] parts.
// The reference is a direct double-precision sum at sampled outputs that include
// every corner, the last row/column and both sides of each 64-point tile boundary.
// FFT paths include the filter FFT in every timed iteration; the zero-padded,
// flipped spatial filter layout is prepared once before timing.
// cuDNN and cuFFT follow the earlier evaluation: real FP32 (the real parts of x and
// h); cuDNN with CUDNN_CROSS_CORRELATION and the fastest of the top three heuristic
// algorithms, cuFFT with full-image R2C/C2R. cuFFTDx (and tiled ThunderFFT) are complex.
#include <algorithm>
#include <cmath>
#include <random>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "../utils.h"
#ifdef THFF_WITH_CUDNN
#include <cudnn.h>
#define CHECK_CUDNN(stmt)                                                                  \
    do {                                                                                   \
        cudnnStatus_t st__ = (stmt);                                                       \
        if (st__ != CUDNN_STATUS_SUCCESS) {                                                \
            std::fprintf(stderr, "cuDNN error %s at %s:%d\n", cudnnGetErrorString(st__),    \
                         __FILE__, __LINE__);                                              \
            std::exit(EXIT_FAILURE);                                                       \
        }                                                                                  \
    } while (0)
#endif

using Gate = std::function<bool(const float2*)>;
std::vector<float> thunder_tiled64_convolution(const float2*, const float2*, int, int, const Gate&, int, int, int);

// Implemented in cufftdx.cu: tiled (64) cuFFTDx pipeline; one time per trial, none if the gate fails.
template <int f>
std::vector<float> cufftdx_convolution(const float2* h_input, const float2* h_filter, int N, const Gate& gate,
                                       int warmup, int runs, int trials);

// Pointwise product with the inverse-FFT normalization folded in.
__global__ void complex_multiply_kernel(cufftComplex* a, const cufftComplex* b, float scale, size_t n) {
    const size_t idx = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
    if (idx < n) {
        const float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        const float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
        a[idx].x = real * scale;
        a[idx].y = imag * scale;
    }
}

// Full-image cuFFT baseline, as in the earlier evaluation: real input (real parts of
// x and h), R2C of the image and of the N x N padded filter, product, C2R. The
// filter FFT is in every timed iteration; plans, buffers and the uploaded image are
// prepared once per process and reused for every filter size.
struct CufftState {
    int N = 0;
    float *input = nullptr, *filter = nullptr, *result = nullptr;
    cufftComplex *input_fft = nullptr, *filter_fft = nullptr;
    cufftHandle r2c{}, c2r{};
};

static std::vector<float> cufft_convolution(const float2* h_input, const float2* h_filter, int N, int f,
                                            const Gate& gate, int warmup, int runs, int trials) {
    const int out_size = N - f + 1;
    const size_t count = size_t(N) * N, spectrum = size_t(N) * (N / 2 + 1);
    // ponytail: one image size per process; buffers live until exit (the driver's lifetime).
    static CufftState st;
    if (st.N != N) {
        st.N = N;
        CHECK_CUDA(cudaMalloc(&st.input, sizeof(float) * count));
        CHECK_CUDA(cudaMalloc(&st.filter, sizeof(float) * count));
        CHECK_CUDA(cudaMalloc(&st.result, sizeof(float) * count));
        CHECK_CUDA(cudaMalloc(&st.input_fft, sizeof(cufftComplex) * spectrum));
        CHECK_CUDA(cudaMalloc(&st.filter_fft, sizeof(cufftComplex) * spectrum));
        std::vector<float> real(count);
        for (size_t i = 0; i < count; ++i) real[i] = h_input[i].x;
        CHECK_CUDA(cudaMemcpy(st.input, real.data(), sizeof(float) * count, cudaMemcpyHostToDevice));
        CHECK_CUFFT(cufftPlan2d(&st.r2c, N, N, CUFFT_R2C));
        CHECK_CUFFT(cufftPlan2d(&st.c2r, N, N, CUFFT_C2R));
    }
    // Correlation as convolution with the filter flipped around the origin: only the
    // f rows (N - i) % N are nonzero, so clear the buffer and upload those rows.
    CHECK_CUDA(cudaMemset(st.filter, 0, sizeof(float) * count));
    std::vector<float> row(N);
    for (int i = 0; i < f; ++i) {
        std::fill(row.begin(), row.end(), 0.0f);
        for (int j = 0; j < f; ++j) row[(N - j) % N] = h_filter[i * f + j].x;
        CHECK_CUDA(cudaMemcpy(st.filter + size_t((N - i) % N) * N, row.data(), sizeof(float) * N,
                              cudaMemcpyHostToDevice));
    }
    const int threads = 256;
    const unsigned blocks = unsigned((spectrum + threads - 1) / threads);
    auto run_once = [&]() {
        CHECK_CUFFT(cufftExecR2C(st.r2c, st.input, st.input_fft));
        CHECK_CUFFT(cufftExecR2C(st.r2c, st.filter, st.filter_fft));
        complex_multiply_kernel<<<blocks, threads>>>(st.input_fft, st.filter_fft, 1.0f / float(count), spectrum);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUFFT(cufftExecC2R(st.c2r, st.input_fft, st.result));
    };
    run_once();
    // Read back only the valid region for the gate.
    std::vector<float> valid(size_t(out_size) * out_size);
    CHECK_CUDA(cudaMemcpy2D(valid.data(), sizeof(float) * out_size, st.result, sizeof(float) * N,
                            sizeof(float) * out_size, out_size, cudaMemcpyDeviceToHost));
    std::vector<float2> h_output(valid.size());
    for (size_t i = 0; i < valid.size(); ++i) h_output[i] = make_float2(valid[i], 0.0f);
    std::vector<float> times;
    if (gate(h_output.data())) {
        const unsigned timed = adaptive_runs(run_once, runs);
        // Warm up once before the first trial; later trials follow back to back.
        for (int t = 0; t < trials; ++t)
            times.push_back(measure_execution_ms(run_once, t == 0 ? std::min<unsigned>(warmup, (timed + 2) / 3) : 0, timed));
    }
    return times;
}

#ifdef THFF_WITH_CUDNN
static std::vector<float> cudnn_convolution(const float2* h_input, const float2* h_filter, int N, int f,
                                            const Gate& gate, int warmup, int runs, int trials) {
    const int out_size = N - f + 1;
    if (cudnnGetVersion() != CUDNN_VERSION) {
        std::cerr << "cuDNN runtime " << cudnnGetVersion() << " differs from headers " << CUDNN_VERSION
                  << "; set CUDNN_LIB and LD_LIBRARY_PATH to one installation\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<float> x(size_t(N) * N), w(size_t(f) * f);
    for (size_t i = 0; i < x.size(); ++i) x[i] = h_input[i].x;
    for (size_t i = 0; i < w.size(); ++i) w[i] = h_filter[i].x;
    cudnnHandle_t handle;
    CHECK_CUDNN(cudnnCreate(&handle));
    cudnnTensorDescriptor_t in_desc, out_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&in_desc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&out_desc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter_desc));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, N, N));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 1, 1, f, f));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 0, 0, 1, 1, 1, 1, CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, 1, out_size, out_size));
    float *d_input = nullptr, *d_filter = nullptr, *d_output = nullptr;
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float) * x.size()));
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(float) * w.size()));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float) * size_t(out_size) * out_size));
    CHECK_CUDA(cudaMemcpy(d_input, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_filter, w.data(), sizeof(float) * w.size(), cudaMemcpyHostToDevice));
    // Best algorithm: time cuDNN's top heuristic candidates and keep the fastest.
    // cudnnFind is avoided: at N=16384, f>=31 (cuDNN 9.12) it grew host memory past
    // 300 GB while trying FFT_TILING, which takes minutes per call here.
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    int found = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(handle, in_desc, filter_desc, conv_desc, out_desc, 8, &found,
                                                       perf));
    const float alpha = 1.0f, beta = 0.0f;
    cudnnConvolutionFwdAlgo_t algo{};
    size_t workspace_bytes = 0;
    void* workspace = nullptr;
    float best_ms = -1.0f;
    for (int k = 0, tried = 0; k < found && tried < 3; ++k) {
        if (perf[k].status != CUDNN_STATUS_SUCCESS) continue;
        ++tried;
        size_t bytes = 0;
        CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle, in_desc, filter_desc, conv_desc, out_desc,
                                                            perf[k].algo, &bytes));
        void* buffer = nullptr;
        if (bytes) CHECK_CUDA(cudaMalloc(&buffer, bytes));
        const float ms = measure_execution_ms([&]() {
            CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, in_desc, d_input, filter_desc, d_filter, conv_desc,
                                                perf[k].algo, buffer, bytes, &beta, out_desc, d_output));
        }, 0, 1);
        std::cout << "cuDNN candidate algorithm " << int(perf[k].algo) << ": " << ms << " ms\n";
        if (best_ms < 0.0f || ms < best_ms) {
            if (workspace) CHECK_CUDA(cudaFree(workspace));
            best_ms = ms, algo = perf[k].algo, workspace = buffer, workspace_bytes = bytes;
        } else if (buffer) {
            CHECK_CUDA(cudaFree(buffer));
        }
    }
    if (best_ms < 0.0f) {
        std::cerr << "cuDNN heuristics returned no supported algorithm\n";
        std::exit(EXIT_FAILURE);
    }
    std::cout << "cuDNN " << cudnnGetVersion() << " algorithm: " << int(algo)
              << " workspace_bytes=" << workspace_bytes << "\n";
    auto run_once = [&]() {
        CHECK_CUDNN(cudnnConvolutionForward(handle, &alpha, in_desc, d_input, filter_desc, d_filter, conv_desc, algo,
                                            workspace, workspace_bytes, &beta, out_desc, d_output));
    };
    run_once();
    std::vector<float> real(size_t(out_size) * out_size);
    CHECK_CUDA(cudaMemcpy(real.data(), d_output, sizeof(float) * real.size(), cudaMemcpyDeviceToHost));
    std::vector<float2> h_output(real.size());
    for (size_t i = 0; i < real.size(); ++i) h_output[i] = make_float2(real[i], 0.0f);
    std::vector<float> times;
    if (gate(h_output.data())) {
        const unsigned timed = adaptive_runs(run_once, runs);
        // Warm up once before the first trial; later trials follow back to back.
        for (int t = 0; t < trials; ++t)
            times.push_back(measure_execution_ms(run_once, t == 0 ? std::min<unsigned>(warmup, (timed + 2) / 3) : 0, timed));
    }
    if (workspace) CHECK_CUDA(cudaFree(workspace));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDNN(cudnnDestroyConvolutionDescriptor(conv_desc));
    CHECK_CUDNN(cudnnDestroyFilterDescriptor(filter_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(out_desc));
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(in_desc));
    CHECK_CUDNN(cudnnDestroy(handle));
    return times;
}
#endif

// Sampled direct reference; returns relative L2 over the sampled outputs.
static double sampled_rel_l2(const std::vector<float2>& x, const std::vector<float2>& h, const float2* y,
                             int N, int f, size_t* samples) {
    const int out = N - f + 1;
    const int valid_tile = 64 - f + 1;
    std::vector<int> coords = {0, 1, out / 2, out - 2, out - 1};
    for (int t = valid_tile; t < out; t += valid_tile) {
        coords.push_back(t - 1);
        coords.push_back(t);
    }
    std::sort(coords.begin(), coords.end());
    coords.erase(std::unique(coords.begin(), coords.end()), coords.end());
    // Every boundary row against a strided subset of columns, and vice versa:
    // checking all boundary pairs would cost ~1e10 operations at 16384^2.
    if (N <= 256) {
        coords.clear();
        for (int i=0; i<out; ++i) coords.push_back(i);
    }
    std::vector<int> subset;
    const size_t stride = N <= 256 ? 1 : std::max<size_t>(1, coords.size() / 64);
    for (size_t k = 0; k < coords.size(); ++k)
        if (k % stride == 0 || k + 2 >= coords.size()) subset.push_back(coords[k]);
    std::vector<std::pair<int, int>> points;
    for (int i : coords)
        for (int j : subset) points.emplace_back(i, j);
    for (int i : subset)
        for (int j : coords) points.emplace_back(i, j);
    double num = 0.0, den = 0.0;
    const size_t n = points.size();
#pragma omp parallel for reduction(+ : num, den) schedule(dynamic, 64)
    for (size_t k = 0; k < n; ++k) {
        const int i = points[k].first, j = points[k].second;
        double re = 0.0, im = 0.0;
        for (int a = 0; a < f; ++a)
            for (int b = 0; b < f; ++b) {
                const float2 xv = x[size_t(i + a) * N + j + b], hv = h[a * f + b];
                re += double(xv.x) * hv.x - double(xv.y) * hv.y;
                im += double(xv.x) * hv.y + double(xv.y) * hv.x;
            }
        const float2 yv = y[size_t(i) * out + j];
        // A non-finite output makes num non-finite, which fails the gate.
        num += (yv.x - re) * (yv.x - re) + (yv.y - im) * (yv.y - im);
        den += re * re + im * im;
    }
    *samples = n;
    return std::sqrt(num / den);
}

#ifdef THFF_WITH_CUFFTDX
template <int f>
static std::vector<float> dispatch_cufftdx(const float2* x, const float2* h, int N, const Gate& gate, int warmup,
                                           int runs, int trials) {
    return cufftdx_convolution<f>(x, h, N, gate, warmup, runs, trials);
}

static std::vector<float> run_cufftdx(int f, const float2* x, const float2* h, int N, const Gate& gate, int warmup,
                                      int runs, int trials) {
    switch (f) {
#define THFF_CASE(F) case F: return dispatch_cufftdx<F>(x, h, N, gate, warmup, runs, trials);
        THFF_CASE(3) THFF_CASE(5) THFF_CASE(7) THFF_CASE(9) THFF_CASE(11) THFF_CASE(13) THFF_CASE(15)
        THFF_CASE(17) THFF_CASE(19) THFF_CASE(21) THFF_CASE(23) THFF_CASE(25) THFF_CASE(27) THFF_CASE(29)
        THFF_CASE(31) THFF_CASE(33) THFF_CASE(35) THFF_CASE(37) THFF_CASE(39) THFF_CASE(41) THFF_CASE(43)
        THFF_CASE(45) THFF_CASE(47) THFF_CASE(49) THFF_CASE(51) THFF_CASE(53) THFF_CASE(55) THFF_CASE(57)
        THFF_CASE(59)
#undef THFF_CASE
        default:
            std::cerr << "cuFFTDx tiled path supports odd f from 3 to 59\n";
            std::exit(EXIT_FAILURE);
    }
}

#endif

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " N f[,f...] cufft|cufftdx|cudnn|thunderfft [warmup=5] [runs=20] [trials=1] [device=0]\n";
        return 2;
    }
    const int N = std::atoi(argv[1]);
    std::vector<int> filters;
    for (const char* p = argv[2]; *p;) {
        filters.push_back(std::atoi(p));
        while (*p && *p != ',') ++p;
        if (*p) ++p;
    }
    const std::string library = argv[3];
    const int warmup = argc >= 5 ? std::atoi(argv[4]) : 5;
    const int runs = argc >= 6 ? std::atoi(argv[5]) : 20;
    const int trials = argc >= 7 ? std::atoi(argv[6]) : 1;
    const int device = argc >= 8 ? std::atoi(argv[7]) : 0;
    const bool bad_filter = filters.empty() || std::any_of(filters.begin(), filters.end(), [](int f) { return f < 1 || f > 63; });
    if (bad_filter || trials <= 0 || N < 64 || warmup < 0 || runs <= 0 || (library != "cufft" && library != "cufftdx" && library != "cudnn" && library != "thunderfft")) {
        std::cerr << "invalid arguments\n";
        return 2;
    }
    CHECK_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    std::cerr << "GPU=" << prop.name << " compute=" << prop.major << "." << prop.minor << " batch=1 seed=42\n";

    // The image is generated once (seed 42) and reused for every filter size; each
    // filter is drawn from seed 42+f. cuDNN and cuFFT (R2C) are real: their inputs
    // and reference use the real parts only.
    const bool real = library == "cudnn" || library == "cufft";
    std::vector<float2> x(size_t(N) * N);
    std::minstd_rand rng(42);
    for (auto& v : x) v = make_float2(float(rng()) / float(rng.max()), float(rng()) / float(rng.max()));
    if (real)
        for (auto& v : x) v.y = 0.0f;
    for (const int f : filters) {
        std::cout << "Filter size: " << f << "\n";
        std::vector<float2> h(size_t(f) * f);
        std::minstd_rand rng(42 + f);
        for (auto& v : h) v = make_float2(float(rng()) / float(rng.max()), real ? 0.0f : float(rng()) / float(rng.max()));
        const Gate gate = [&](const float2* y) {
            size_t samples = 0;
            const double rel = sampled_rel_l2(x, h, y, N, f, &samples);
            const bool passed = std::isfinite(rel) && rel <= 5e-3;
            std::cout << std::setprecision(6) << "Validation 2D (" << library << ") vs direct reference: L2_rel="
                      << rel << " samples=" << samples << "\n"
                      << "Convolution validation: " << (passed ? "PASS" : "FAIL") << " (relative L2 limit=0.005)\n";
            return passed;
        };
        std::vector<float> times;
        if (library == "cufft") {
            times = cufft_convolution(x.data(), h.data(), N, f, gate, warmup, runs, trials);
        } else if (library == "thunderfft") {
            times = thunder_tiled64_convolution(x.data(), h.data(), N, f, gate, warmup, runs, trials);
        } else if (library == "cufftdx") {
#ifdef THFF_WITH_CUFFTDX
            times = run_cufftdx(f, x.data(), h.data(), N, gate, warmup, runs, trials);
#else
            std::cerr << "built without cuFFTDx; configure MATHDX_ROOT\n";
            return 78;
#endif
        } else {
#ifdef THFF_WITH_CUDNN
            times = cudnn_convolution(x.data(), h.data(), N, f, gate, warmup, runs, trials);
#else
            std::cerr << "built without cuDNN\n";
            return 78;
#endif
        }
        if (times.empty()) return EXIT_FAILURE;
        for (size_t t = 0; t < times.size(); ++t)
            std::cout << std::fixed << std::setprecision(6) << library << " 2D conv (N=" << N << ", f=" << f
                      << ", filter_fft_included=" << (library == "cudnn" ? 0 : 1) << ", trial=" << t
                      << ") time: " << times[t] << " ms\n" << std::defaultfloat;
    }
    return 0;
}
