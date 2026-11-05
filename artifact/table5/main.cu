#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "thunderfft_bench.h"

#ifdef THFF_WITH_CUFFTDX
#include <cufftdx.hpp>
#endif

namespace {

#define CHECK_CUFFT(call)                                                         \
    do {                                                                          \
        cufftResult err__ = (call);                                               \
        if (err__ != CUFFT_SUCCESS) {                                             \
            std::cerr << "cuFFT error code " << static_cast<int>(err__)           \
                      << " at " << __FILE__ << ":" << __LINE__ << "\n";           \
            std::exit(EXIT_FAILURE);                                              \
        }                                                                         \
    } while (0)

enum class InputMode {
    Gaussian,
    AllOnes,
    Impulse,
};

enum class PrecisionMode {
    Both,
    TF32,
    FP16,
    FP32,
    CufftdxFP16,
};

enum class ReferenceMode {
    FP64,
    FP32,
};

struct Options {
    std::vector<int> n_list{64, 128, 256, 512, 1024, 2048, 4096};
    int batch = 65536;
    InputMode mode = InputMode::Gaussian;
    PrecisionMode precision = PrecisionMode::Both;
    float alpha = 1.0f;
    bool alpha_sweep = false;
    int alpha_exp_min = -20;
    int alpha_exp_max = 20;
    bool fast_finite_only = false;
    ReferenceMode reference = ReferenceMode::FP64;
    uint64_t seed = 1234;
    std::string csv_path = "fft_accuracy.csv";
    std::string environment_path;
};

struct Metrics {
    double rel_l2 = 0.0;
    double max_abs = 0.0;
    double rel_linf = 0.0;
    bool all_finite = true;
};

struct Row {
    int N = 0;
    int batch = 0;
    std::string mode;
    double alpha = 1.0;
    std::string precision;
    Metrics m;
};

static std::string mode_to_string(InputMode m) {
    switch (m) {
        case InputMode::Gaussian: return "gaussian";
        case InputMode::AllOnes: return "all_ones";
        case InputMode::Impulse: return "impulse";
    }
    return "unknown";
}

static bool parse_mode(const std::string& s, InputMode& mode) {
    if (s == "gaussian") {
        mode = InputMode::Gaussian;
        return true;
    }
    if (s == "all_ones") {
        mode = InputMode::AllOnes;
        return true;
    }
    if (s == "impulse") {
        mode = InputMode::Impulse;
        return true;
    }
    return false;
}

static bool parse_precision(const std::string& s, PrecisionMode& p) {
    if (s == "both") {
        p = PrecisionMode::Both;
        return true;
    }
    if (s == "tf32") {
        p = PrecisionMode::TF32;
        return true;
    }
    if (s == "fp16") {
        p = PrecisionMode::FP16;
        return true;
    }
    if (s == "fp32") {
        p = PrecisionMode::FP32;
        return true;
    }
    if (s == "cufftdx_fp16") {
        p = PrecisionMode::CufftdxFP16;
        return true;
    }
    return false;
}

static bool parse_reference(const std::string& s, ReferenceMode& r) {
    if (s == "fp64") {
        r = ReferenceMode::FP64;
        return true;
    }
    if (s == "fp32") {
        r = ReferenceMode::FP32;
        return true;
    }
    return false;
}

static std::vector<int> parse_n_list(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        out.push_back(std::stoi(tok));
    }
    return out;
}

static void print_usage(const char* prog) {
    std::cout
        << "Usage: " << prog << " [options]\n"
        << "  --N-list <comma-separated list>   (default: 64,128,256,512,1024,2048,4096)\n"
        << "  --batch <int>                     (default: 65536)\n"
        << "  --mode <gaussian|all_ones|impulse> (default: gaussian)\n"
        << "  --precision <both|tf32|fp16|fp32|cufftdx_fp16> (default: both)\n"
        << "                                    (cufftdx_fp16 requires a THFF_WITH_CUFFTDX build)\n"
        << "  --alpha <float>                   (default: 1.0)\n"
        << "  --alpha-sweep                     (run alpha = 2^k sweep)\n"
        << "  --alpha-exp-min <int>             (default: -20)\n"
        << "  --alpha-exp-max <int>             (default: 20)\n"
        << "  --fast-finite-only                (skip reference FFT; check finite only)\n"
        << "  --reference <fp64|fp32>           (default: fp64)\n"
        << "  --seed <uint64>                   (default: 1234)\n"
        << "  --csv <path>                      (default: fft_accuracy.csv)\n"
        << "  --environment-json <path>         (record selected GPU and library versions)\n";
}

static Options parse_args(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                std::exit(EXIT_FAILURE);
            }
            return std::string(argv[++i]);
        };

        if (a == "--N-list") {
            opt.n_list = parse_n_list(need("--N-list"));
        } else if (a == "--batch") {
            opt.batch = std::stoi(need("--batch"));
        } else if (a == "--mode") {
            InputMode m;
            if (!parse_mode(need("--mode"), m)) {
                std::cerr << "Invalid --mode\n";
                std::exit(EXIT_FAILURE);
            }
            opt.mode = m;
        } else if (a == "--precision") {
            PrecisionMode p;
            if (!parse_precision(need("--precision"), p)) {
                std::cerr << "Invalid --precision\n";
                std::exit(EXIT_FAILURE);
            }
            opt.precision = p;
        } else if (a == "--alpha") {
            opt.alpha = std::stof(need("--alpha"));
        } else if (a == "--alpha-sweep") {
            opt.alpha_sweep = true;
        } else if (a == "--alpha-exp-min") {
            opt.alpha_exp_min = std::stoi(need("--alpha-exp-min"));
        } else if (a == "--alpha-exp-max") {
            opt.alpha_exp_max = std::stoi(need("--alpha-exp-max"));
        } else if (a == "--fast-finite-only") {
            opt.fast_finite_only = true;
        } else if (a == "--reference") {
            ReferenceMode r;
            if (!parse_reference(need("--reference"), r)) {
                std::cerr << "Invalid --reference\n";
                std::exit(EXIT_FAILURE);
            }
            opt.reference = r;
        } else if (a == "--seed") {
            opt.seed = static_cast<uint64_t>(std::stoull(need("--seed")));
        } else if (a == "--csv") {
            opt.csv_path = need("--csv");
        } else if (a == "--environment-json") {
            opt.environment_path = need("--environment-json");
        } else if (a == "--help" || a == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown arg: " << a << "\n";
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
#ifndef THFF_WITH_CUFFTDX
    if (opt.precision == PrecisionMode::CufftdxFP16) {
        std::cerr << "--precision cufftdx_fp16 requires a build with THFF_WITH_CUFFTDX (configure with MATHDX_ROOT)\n";
        std::exit(EXIT_FAILURE);
    }
#endif
    if (opt.alpha_sweep && opt.alpha_exp_min > opt.alpha_exp_max) {
        std::swap(opt.alpha_exp_min, opt.alpha_exp_max);
    }
    return opt;
}

static std::string json_string(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result << '\\' << c;
        else if (c < 0x20) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else result << c;
    }
    result << '"';
    return result.str();
}

static void write_environment(const std::string& path) {
    if (path.empty()) return;
    int device = 0, driver = 0, runtime = 0, cufft_version = 0;
    cudaDeviceProp properties{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&properties, device));
    CHECK_CUDA(cudaDriverGetVersion(&driver));
    CHECK_CUDA(cudaRuntimeGetVersion(&runtime));
    CHECK_CUFFT(cufftGetVersion(&cufft_version));
    char pci_bus_id[32]{};
    CHECK_CUDA(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device));
    std::ostringstream gpu_uuid;
    gpu_uuid << "GPU-" << std::hex << std::setfill('0');
    for (int index = 0; index < 16; ++index) {
        if (index == 4 || index == 6 || index == 8 || index == 10) gpu_uuid << '-';
        gpu_uuid << std::setw(2) << static_cast<unsigned int>(
            static_cast<unsigned char>(properties.uuid.bytes[index]));
    }
    std::ofstream output(path);
    if (!output) {
        std::cerr << "failed to open environment path: " << path << '\n';
        std::exit(EXIT_FAILURE);
    }
    output << "{\n  \"gpu_name\": " << json_string(properties.name)
           << ",\n  \"gpu_uuid\": " << json_string(gpu_uuid.str())
           << ",\n  \"local_cuda_device\": " << device
           << ",\n  \"pci_bus_id\": " << json_string(pci_bus_id)
           << ",\n  \"compute_capability\": " << json_string(std::to_string(properties.major) + "." + std::to_string(properties.minor))
           << ",\n  \"total_global_memory_bytes\": " << properties.totalGlobalMem
           << ",\n  \"cuda_driver_api_version\": " << driver
           << ",\n  \"cuda_runtime_version\": " << runtime
           << ",\n  \"cuda_compile_version\": " << CUDART_VERSION
           << ",\n  \"cufft_version\": " << cufft_version << "\n}\n";
}

static std::vector<float> make_alphas(const Options& opt) {
    std::vector<float> a;
    if (opt.alpha_sweep) {
        for (int e = opt.alpha_exp_min; e <= opt.alpha_exp_max; ++e) {
            a.push_back(std::ldexp(1.0f, e));
        }
    } else {
        a.push_back(opt.alpha);
    }
    return a;
}

static void generate_input_fp32(std::vector<float2>& h_input,
                                int N,
                                int batch,
                                InputMode mode,
                                float alpha,
                                uint64_t seed) {
    h_input.assign(static_cast<size_t>(N) * static_cast<size_t>(batch), make_float2(0.0f, 0.0f));

    if (mode == InputMode::AllOnes) {
        for (int b = 0; b < batch; ++b) {
            for (int j = 0; j < N; ++j) {
                h_input[static_cast<size_t>(b) * N + j] = make_float2(alpha, 0.0f);
            }
        }
        return;
    }

    if (mode == InputMode::Impulse) {
        for (int b = 0; b < batch; ++b) {
            h_input[static_cast<size_t>(b) * N] = make_float2(alpha, 0.0f);
        }
        return;
    }

    // Gaussian mode: normalize per FFT instance so ||x||_2 / sqrt(N) = 1, then scale by alpha.
    std::normal_distribution<float> normal(0.0f, 1.0f);
    for (int b = 0; b < batch; ++b) {
        std::mt19937_64 rng(seed + static_cast<uint64_t>(b));
        double norm2 = 0.0;
        for (int j = 0; j < N; ++j) {
            float re = normal(rng);
            float im = normal(rng);
            h_input[static_cast<size_t>(b) * N + j] = make_float2(re, im);
            norm2 += static_cast<double>(re) * re + static_cast<double>(im) * im;
        }
        if (norm2 <= 0.0) norm2 = 1.0;
        const double target = std::sqrt(static_cast<double>(N));
        const double scale = (target / std::sqrt(norm2)) * static_cast<double>(alpha);
        for (int j = 0; j < N; ++j) {
            float2& v = h_input[static_cast<size_t>(b) * N + j];
            v.x = static_cast<float>(v.x * scale);
            v.y = static_cast<float>(v.y * scale);
        }
    }
}

static void pack_half2(const std::vector<float2>& in, std::vector<half2>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = __floats2half2_rn(in[i].x, in[i].y);
    }
}

static void reference_fft_double(const std::vector<float2>& h_input,
                                 int N,
                                 int batch,
                                 std::vector<cufftDoubleComplex>& h_ref_out) {
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(batch);
    h_ref_out.resize(elems);

    std::vector<cufftDoubleComplex> h_in(elems);
    for (size_t i = 0; i < elems; ++i) {
        h_in[i].x = static_cast<double>(h_input[i].x);
        h_in[i].y = static_cast<double>(h_input[i].y);
    }

    cufftDoubleComplex* d_in = nullptr;
    cufftDoubleComplex* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftDoubleComplex) * elems));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftDoubleComplex) * elems));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), sizeof(cufftDoubleComplex) * elems, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_Z2Z, batch));
    CHECK_CUFFT(cufftExecZ2Z(plan, d_in, d_out, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ref_out.data(), d_out, sizeof(cufftDoubleComplex) * elems, cudaMemcpyDeviceToHost));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

static void reference_fft_float(const std::vector<float2>& h_input,
                                int N,
                                int batch,
                                std::vector<float2>& h_ref_out) {
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(batch);
    h_ref_out.resize(elems);

    cufftComplex* d_in = nullptr;
    cufftComplex* d_out = nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(cufftComplex) * elems));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(cufftComplex) * elems));
    CHECK_CUDA(cudaMemcpy(d_in, h_input.data(), sizeof(cufftComplex) * elems, cudaMemcpyHostToDevice));

    cufftHandle plan;
    CHECK_CUFFT(cufftPlan1d(&plan, N, CUFFT_C2C, batch));
    CHECK_CUFFT(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_ref_out.data(), d_out, sizeof(cufftComplex) * elems, cudaMemcpyDeviceToHost));
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
}

template <typename T>
static void copy_device_to_float2(const vec2_t<T>* d_data,
                                  size_t elems,
                                  std::vector<float2>& out) {
    out.resize(elems);
    if constexpr (std::is_same_v<T, float>) {
        CHECK_CUDA(cudaMemcpy(out.data(), d_data, sizeof(float2) * elems, cudaMemcpyDeviceToHost));
    } else {
        std::vector<half2> tmp(elems);
        CHECK_CUDA(cudaMemcpy(tmp.data(), d_data, sizeof(half2) * elems, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < elems; ++i) {
            out[i].x = __half2float(__low2half(tmp[i]));
            out[i].y = __half2float(__high2half(tmp[i]));
        }
    }
}

template <typename T, int N>
static void run_thunderfft_reg_e2e(const std::vector<vec2_t<T>>& h_input,
                                   int batch,
                                   std::vector<float2>& h_out_f32) {
    using T2 = vec2_t<T>;
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(batch);

    if (batch % BPB != 0) {
        std::cerr << "batch(" << batch << ") must be divisible by batch_per_block(" << BPB
                  << ") for N=" << N << "\n";
        std::exit(EXIT_FAILURE);
    }

    T2* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(T2) * elems));
    CHECK_CUDA(cudaMemcpy(d_data, h_input.data(), sizeof(T2) * elems, cudaMemcpyHostToDevice));

    const dim3 grid(batch / BPB);
    const dim3 block(thunderfft::threads_per_warp, WPB);
    const size_t shmem_bytes = sizeof(T2) * (N + pad_h(N)) * BPB;

    CHECK_CUDA(cudaFuncSetAttribute(
        thunderfft::ThunderFFT_benchmark_reg_e2e<T, N, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));

    thunderfft::ThunderFFT_benchmark_reg_e2e<T, N, true>
        <<<grid, block, shmem_bytes>>>(d_data, d_data, nullptr);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    copy_device_to_float2<T>(d_data, elems, h_out_f32);
    CHECK_CUDA(cudaFree(d_data));
}

#ifdef THFF_WITH_CUFFTDX
// Two FFTs share half2 lanes; the FP32-generated input is rounded to FP16 here,
// as in the ThunderFFT FP16 path.
template <class FFT>
__global__ __launch_bounds__(FFT::max_threads_per_block)
void cufftdx_fp16_kernel(float2* data, int batch) {
    constexpr int N = cufftdx::size_of<FFT>::value;
    const size_t first = static_cast<size_t>(blockIdx.x) * FFT::ffts_per_block
                       + threadIdx.y * FFT::implicit_type_batching;
    extern __shared__ __align__(16) unsigned char shared_mem[];
    typename FFT::value_type values[FFT::storage_size] = {};

    #pragma unroll
    for (int i = 0; i < FFT::elements_per_thread; ++i) {
        const int j = threadIdx.x + i * FFT::stride;
        if (j < N) {
            const float2 a = first < batch ? data[first * N + j] : make_float2(0, 0);
            const float2 b = first + 1 < batch ? data[(first + 1) * N + j] : make_float2(0, 0);
            values[i].x = __floats2half2_rn(a.x, b.x);
            values[i].y = __floats2half2_rn(a.y, b.y);
        }
    }
    // All threads participate, including a zero-padded final batch lane.
    FFT().execute(values, shared_mem);
    #pragma unroll
    for (int i = 0; i < FFT::elements_per_thread; ++i) {
        const int j = threadIdx.x + i * FFT::stride;
        if (j < N) {
            if (first < batch)
                data[first * N + j] = make_float2(__low2float(values[i].x), __low2float(values[i].y));
            if (first + 1 < batch)
                data[(first + 1) * N + j] = make_float2(__high2float(values[i].x), __high2float(values[i].y));
        }
    }
}

template <int N>
static void run_cufftdx_fp16(const std::vector<float2>& input, int batch, std::vector<float2>& output) {
    using FFT = decltype(cufftdx::Size<N>() + cufftdx::Precision<__half>()
                       + cufftdx::Type<cufftdx::fft_type::c2c>()
                       + cufftdx::Direction<cufftdx::fft_direction::forward>()
                       + cufftdx::ElementsPerThread<8>() + cufftdx::FFTsPerBlock<2>()
                       + cufftdx::SM<THFF_CUFFTDX_SM>() + cufftdx::Block());
    static_assert(!FFT::requires_workspace, "Selected cuFFTDx FFT requires a workspace");
    float2* data = nullptr;
    const size_t bytes = input.size() * sizeof(float2);
    CHECK_CUDA(cudaMalloc(&data, bytes));
    CHECK_CUDA(cudaMemcpy(data, input.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaFuncSetAttribute(cufftdx_fp16_kernel<FFT>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   FFT::shared_memory_size));
    const unsigned blocks = static_cast<unsigned>(
        (static_cast<size_t>(batch) + FFT::ffts_per_block - 1) / FFT::ffts_per_block);
    cufftdx_fp16_kernel<FFT><<<blocks, FFT::block_dim, FFT::shared_memory_size>>>(data, batch);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    output.resize(input.size());
    CHECK_CUDA(cudaMemcpy(output.data(), data, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(data));
}
#endif

static Metrics compute_metrics(const std::vector<float2>& y,
                               const std::vector<cufftDoubleComplex>& y_ref) {
    Metrics m;
    double err2 = 0.0;
    double ref2 = 0.0;
    double max_err = 0.0;
    double max_ref = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        const double yr = static_cast<double>(y[i].x);
        const double yi = static_cast<double>(y[i].y);
        const double rr = y_ref[i].x;
        const double ri = y_ref[i].y;

        if (!std::isfinite(yr) || !std::isfinite(yi)) m.all_finite = false;

        const double dx = yr - rr;
        const double dy = yi - ri;
        const double ae = std::sqrt(dx * dx + dy * dy);
        const double ar = std::sqrt(rr * rr + ri * ri);

        err2 += dx * dx + dy * dy;
        ref2 += rr * rr + ri * ri;
        if (ae > max_err) max_err = ae;
        if (ar > max_ref) max_ref = ar;
    }

    m.max_abs = max_err;
    if (ref2 > 0.0) {
        m.rel_l2 = std::sqrt(err2 / ref2);
    } else {
        m.rel_l2 = (err2 == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    if (max_ref > 0.0) {
        m.rel_linf = max_err / max_ref;
    } else {
        m.rel_linf = (max_err == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return m;
}

static Metrics compute_metrics(const std::vector<float2>& y,
                               const std::vector<float2>& y_ref) {
    Metrics m;
    double err2 = 0.0;
    double ref2 = 0.0;
    double max_err = 0.0;
    double max_ref = 0.0;

    for (size_t i = 0; i < y.size(); ++i) {
        const double yr = static_cast<double>(y[i].x);
        const double yi = static_cast<double>(y[i].y);
        const double rr = static_cast<double>(y_ref[i].x);
        const double ri = static_cast<double>(y_ref[i].y);

        if (!std::isfinite(yr) || !std::isfinite(yi)) m.all_finite = false;

        const double dx = yr - rr;
        const double dy = yi - ri;
        const double ae = std::sqrt(dx * dx + dy * dy);
        const double ar = std::sqrt(rr * rr + ri * ri);

        err2 += dx * dx + dy * dy;
        ref2 += rr * rr + ri * ri;
        if (ae > max_err) max_err = ae;
        if (ar > max_ref) max_ref = ar;
    }

    m.max_abs = max_err;
    if (ref2 > 0.0) {
        m.rel_l2 = std::sqrt(err2 / ref2);
    } else {
        m.rel_l2 = (err2 == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    if (max_ref > 0.0) {
        m.rel_linf = max_err / max_ref;
    } else {
        m.rel_linf = (max_err == 0.0) ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return m;
}

static bool all_finite_only(const std::vector<float2>& y) {
    for (const auto& v : y) {
        if (!std::isfinite(static_cast<double>(v.x)) || !std::isfinite(static_cast<double>(v.y))) {
            return false;
        }
    }
    return true;
}

static void print_row(const Row& r) {
    std::cout << "N=" << r.N
              << " batch=" << r.batch
              << " mode=" << r.mode
              << " alpha=" << std::scientific << std::setprecision(4) << r.alpha
              << " precision=" << r.precision
              << " rel_l2=" << std::scientific << std::setprecision(6) << r.m.rel_l2
              << " max_abs=" << r.m.max_abs
              << " rel_linf=" << r.m.rel_linf
              << " all_finite=" << (r.m.all_finite ? 1 : 0)
              << "\n";
}

static void write_csv(const std::string& path, const std::vector<Row>& rows) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "failed to open csv path: " << path << "\n";
        std::exit(EXIT_FAILURE);
    }
    ofs << "N,batch,mode,alpha,precision,rel_l2,max_abs,rel_linf,all_finite\n";
    for (const auto& r : rows) {
        ofs << r.N << ','
            << r.batch << ','
            << r.mode << ','
            << std::setprecision(9) << r.alpha << ','
            << r.precision << ','
            << std::setprecision(12) << r.m.rel_l2 << ','
            << r.m.max_abs << ','
            << r.m.rel_linf << ','
            << (r.m.all_finite ? 1 : 0) << '\n';
    }
}

template <int N>
static void run_for_n(const Options& opt,
                      const std::vector<float>& alphas,
                      std::vector<Row>& rows) {
    const int batch = opt.batch;
    const std::string mode_s = mode_to_string(opt.mode);
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(batch);

    for (float alpha : alphas) {
        std::vector<float2> h_input_f32(elems);
        std::vector<half2> h_input_half;
        generate_input_fp32(h_input_f32, N, batch, opt.mode, alpha, opt.seed);
        const bool run_tf32 = (opt.precision == PrecisionMode::Both || opt.precision == PrecisionMode::TF32);
        const bool run_fp16 = (opt.precision == PrecisionMode::Both || opt.precision == PrecisionMode::FP16);
        const bool run_fp32 = (opt.precision == PrecisionMode::FP32);
        const bool run_cufftdx_fp16_path = (opt.precision == PrecisionMode::CufftdxFP16);

        if (run_fp16) {
            pack_half2(h_input_f32, h_input_half);
        }

        std::vector<cufftDoubleComplex> h_ref;
        std::vector<float2> h_ref_f32;
        const bool need_ref = !opt.fast_finite_only;
        if (need_ref) {
            if (opt.reference == ReferenceMode::FP64) {
                reference_fft_double(h_input_f32, N, batch, h_ref);
            } else {
                reference_fft_float(h_input_f32, N, batch, h_ref_f32);
            }
        }

        if (run_tf32) {
            std::vector<float2> y_tf32;
            thunderfft::ThunderFFTInitialize<float>(N);
            run_thunderfft_reg_e2e<float, N>(h_input_f32, batch, y_tf32);
            thunderfft::ThunderFFTFinalize<float>();

            Row r_tf32;
            r_tf32.N = N;
            r_tf32.batch = batch;
            r_tf32.mode = mode_s;
            r_tf32.alpha = alpha;
            r_tf32.precision = "tf32";
            if (need_ref) {
                if (opt.reference == ReferenceMode::FP64) {
                    r_tf32.m = compute_metrics(y_tf32, h_ref);
                } else {
                    r_tf32.m = compute_metrics(y_tf32, h_ref_f32);
                }
            } else {
                r_tf32.m.rel_l2 = std::numeric_limits<double>::quiet_NaN();
                r_tf32.m.max_abs = std::numeric_limits<double>::quiet_NaN();
                r_tf32.m.rel_linf = std::numeric_limits<double>::quiet_NaN();
                r_tf32.m.all_finite = all_finite_only(y_tf32);
            }
            rows.push_back(r_tf32);
            print_row(r_tf32);
        }

        if (run_fp16) {
            std::vector<float2> y_fp16;
            thunderfft::ThunderFFTInitialize<half>(N);
            run_thunderfft_reg_e2e<half, N>(h_input_half, batch, y_fp16);
            thunderfft::ThunderFFTFinalize<half>();

            Row r_fp16;
            r_fp16.N = N;
            r_fp16.batch = batch;
            r_fp16.mode = mode_s;
            r_fp16.alpha = alpha;
            r_fp16.precision = "fp16";
            if (need_ref) {
                if (opt.reference == ReferenceMode::FP64) {
                    r_fp16.m = compute_metrics(y_fp16, h_ref);
                } else {
                    r_fp16.m = compute_metrics(y_fp16, h_ref_f32);
                }
            } else {
                r_fp16.m.rel_l2 = std::numeric_limits<double>::quiet_NaN();
                r_fp16.m.max_abs = std::numeric_limits<double>::quiet_NaN();
                r_fp16.m.rel_linf = std::numeric_limits<double>::quiet_NaN();
                r_fp16.m.all_finite = all_finite_only(y_fp16);
            }
            rows.push_back(r_fp16);
            print_row(r_fp16);
        }

        if (run_fp32) {
            std::vector<float2> y_fp32;
            reference_fft_float(h_input_f32, N, batch, y_fp32);  // cuFFT C2C as FP32 path

            Row r_fp32;
            r_fp32.N = N;
            r_fp32.batch = batch;
            r_fp32.mode = mode_s;
            r_fp32.alpha = alpha;
            r_fp32.precision = "fp32";
            if (need_ref) {
                if (opt.reference == ReferenceMode::FP64) {
                    r_fp32.m = compute_metrics(y_fp32, h_ref);
                } else {
                    r_fp32.m = compute_metrics(y_fp32, h_ref_f32);
                }
            } else {
                r_fp32.m.rel_l2 = std::numeric_limits<double>::quiet_NaN();
                r_fp32.m.max_abs = std::numeric_limits<double>::quiet_NaN();
                r_fp32.m.rel_linf = std::numeric_limits<double>::quiet_NaN();
                r_fp32.m.all_finite = all_finite_only(y_fp32);
            }
            rows.push_back(r_fp32);
            print_row(r_fp32);
        }

#ifdef THFF_WITH_CUFFTDX
        if (run_cufftdx_fp16_path) {
            std::vector<float2> y_dx;
            run_cufftdx_fp16<N>(h_input_f32, batch, y_dx);

            Row r_dx;
            r_dx.N = N;
            r_dx.batch = batch;
            r_dx.mode = mode_s;
            r_dx.alpha = alpha;
            r_dx.precision = "fp16_cufftdx";
            if (need_ref) {
                if (opt.reference == ReferenceMode::FP64) {
                    r_dx.m = compute_metrics(y_dx, h_ref);
                } else {
                    r_dx.m = compute_metrics(y_dx, h_ref_f32);
                }
            } else {
                r_dx.m.rel_l2 = std::numeric_limits<double>::quiet_NaN();
                r_dx.m.max_abs = std::numeric_limits<double>::quiet_NaN();
                r_dx.m.rel_linf = std::numeric_limits<double>::quiet_NaN();
                r_dx.m.all_finite = all_finite_only(y_dx);
            }
            rows.push_back(r_dx);
            print_row(r_dx);
        }
#else
        (void)run_cufftdx_fp16_path;
#endif
    }
}

static void dispatch_n(const Options& opt,
                       int N,
                       const std::vector<float>& alphas,
                       std::vector<Row>& rows) {
    switch (N) {
        case 64: run_for_n<64>(opt, alphas, rows); break;
        case 128: run_for_n<128>(opt, alphas, rows); break;
        case 256: run_for_n<256>(opt, alphas, rows); break;
        case 512: run_for_n<512>(opt, alphas, rows); break;
        case 1024: run_for_n<1024>(opt, alphas, rows); break;
        case 2048: run_for_n<2048>(opt, alphas, rows); break;
        case 4096: run_for_n<4096>(opt, alphas, rows); break;
        default:
            std::cerr << "Unsupported N=" << N
                      << " (supported: 64,128,256,512,1024,2048,4096)\n";
            std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main(int argc, char** argv) {
    const Options opt = parse_args(argc, argv);
    write_environment(opt.environment_path);
    const auto alphas = make_alphas(opt);

    std::vector<Row> rows;
    const size_t per_alpha = (opt.precision == PrecisionMode::Both) ? 2u : 1u;
    rows.reserve(opt.n_list.size() * alphas.size() * per_alpha);

    for (int N : opt.n_list) {
        dispatch_n(opt, N, alphas, rows);
    }

    write_csv(opt.csv_path, rows);
    std::cout << "saved: " << opt.csv_path << "\n";
    return 0;
}
