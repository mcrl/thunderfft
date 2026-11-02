// Correctness and memory-boundary checks only; this program does not time FFTs.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thunderfft/thunderfft.cuh>
#include <thunderfft/thunderfft_layout_tuned.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int kSharedGuard = 64;
constexpr int kOutputGuard = 16;
constexpr double kRelativeL2Limit = 5e-3;
// Evaluate this host-only constexpr helper outside device code. NVCC 12.9
// otherwise requires --expt-relaxed-constexpr for a device-side pad_h call.
template <int N>
inline constexpr int kWorkspaceElements = (N + pad_h(N)) * thunderfft::batch_per_block<N>;

void check_cuda(cudaError_t error, const char* expression, int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(error) + " (line " +
                                 std::to_string(line) + ")");
}
void check_cufft(cufftResult error, const char* expression, int line) {
    if (error != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(expression) + ": cuFFT error " +
                                 std::to_string(int(error)) + " (line " +
                                 std::to_string(line) + ")");
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression, __LINE__)
#define CUFFT_CHECK(expression) check_cufft((expression), #expression, __LINE__)

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) { CUDA_CHECK(cudaMalloc(&data, count * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct ReferencePlan {
    cufftHandle handle = 0;
    ReferencePlan(int n, int batch) { CUFFT_CHECK(cufftPlan1d(&handle, n, CUFFT_Z2Z, batch)); }
    ~ReferencePlan() { if (handle) cufftDestroy(handle); }
    ReferencePlan(const ReferencePlan&) = delete;
    ReferencePlan& operator=(const ReferencePlan&) = delete;
};

template <class T> struct TwiddleScope {
    explicit TwiddleScope(int n) { thunderfft::ThunderFFTInitialize<T>(n); }
    ~TwiddleScope() { thunderfft::ThunderFFTFinalize<T>(); }
    TwiddleScope(const TwiddleScope&) = delete;
    TwiddleScope& operator=(const TwiddleScope&) = delete;
};

template <class T> __host__ __device__ vec2_t<T> encode(float2 value) {
    if constexpr (std::is_same_v<T, half>)
        return __floats2half2_rn(value.x, value.y);
    else
        return value;
}
template <class T> __host__ __device__ float2 decode(vec2_t<T> value) {
    if constexpr (std::is_same_v<T, half>)
        return __half22float2(value);
    else
        return value;
}

template <class T, int N, bool Forward>
__global__ void validate_core(const vec2_t<T>* input, vec2_t<T>* output,
                              int* guard_changes) {
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int EPT = N * BPB / (32 * WPB);
    constexpr int workspace_elements = kWorkspaceElements<N>;
    extern __shared__ __align__(16) unsigned char raw[];
    auto* workspace = reinterpret_cast<vec2_t<T>*>(raw);
    vec2_t<T> reg[EPT], coefficients[36];
    const int tid = threadIdx.x + 32 * threadIdx.y;
    for (int j = tid; j < kSharedGuard; j += 32 * WPB)
        workspace[workspace_elements + j] = encode<T>(make_float2(299.f, -299.f));
    __syncthreads();

    if constexpr (std::is_same_v<T, half>)
        thunderfft::unit_fp16::make_reg_b_precompute<N, Forward>(coefficients);
    // Natural-order logical input/output. Padding is benchmark-specific; the
    // inverse specializations preserve the same fragment mapping as forward.
    using L_in = typename thunderfft::bench_layout<T, N, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<T, N, BPB>::L_out;
    thunderfft::ThunderFFT_gmem2smem<T, L_in>(workspace, input + blockIdx.x * BPB * N);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<T, L_in>(reg, workspace);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<T, N, BPB, Forward>(reg, coefficients, workspace);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<T, L_out>(workspace, reg);
    __syncthreads();
    thunderfft::ThunderFFT_smem2gmem<T, L_out>(output + blockIdx.x * BPB * N, workspace);
    __syncthreads();

    if (tid == 0) {
        int changed = 0;
        for (int j = 0; j < kSharedGuard; ++j) {
            const auto value = decode<T>(workspace[workspace_elements + j]);
            changed += value.x != 299.f || value.y != -299.f;
        }
        guard_changes[blockIdx.x] = changed;
    }
}

const char* input_name(int kind) {
    constexpr const char* names[] = {"gaussian_rms1", "offset_impulse", "complex_tone", "constant"};
    return names[kind];
}

std::vector<float2> make_input(int n, int batch, int kind) {
    std::vector<float2> input(size_t(n) * batch, make_float2(0.f, 0.f));
    std::mt19937 random(1234 + n);
    std::normal_distribution<float> normal(0.f, 1.f);
    constexpr double tau = 6.283185307179586476925286766559;
    for (int b = 0; b < batch; ++b) {
        if (kind == 0) {
            double squared_norm = 0;
            for (int i = 0; i < n; ++i) {
                auto value = make_float2(normal(random), normal(random));
                input[size_t(b) * n + i] = value;
                squared_norm += double(value.x) * value.x + double(value.y) * value.y;
            }
            const double scale = std::sqrt(n / squared_norm);
            for (int i = 0; i < n; ++i) {
                auto& value = input[size_t(b) * n + i];
                value.x *= scale;
                value.y *= scale;
            }
        } else if (kind == 1) {
            input[size_t(b) * n + (13 * b + 7) % n] = make_float2(.75f, .25f);
        } else if (kind == 2) {
            const int frequency = 1 + (7 * b) % (n - 1);
            for (int i = 0; i < n; ++i) {
                const double phase = tau * frequency * i / n + .13 * b;
                input[size_t(b) * n + i] = make_float2(.5 * std::cos(phase), .5 * std::sin(phase));
            }
        } else {
            std::fill(input.begin() + size_t(b) * n,
                      input.begin() + size_t(b + 1) * n, make_float2(.25f, -.125f));
        }
    }
    return input;
}

struct Options {
    int device = 0;
    int batch = 64;
    bool quick = false;
    bool host_only = false;
    bool batch_set = false;
    std::string csv;
};

struct Results {
    std::ofstream file;
    int cases = 0;
    int failures = 0;
    explicit Results(const std::string& path) {
        if (!path.empty()) {
            file.open(path);
            if (!file) throw std::runtime_error("Cannot open CSV output: " + path);
        }
        write("path,precision,N,direction,input,batch,relative_l2,max_abs,relative_linf,nonfinite_outputs,shared_guard_changed_max,output_guard_changed,pass");
    }
    void write(const std::string& line) {
        std::cout << line << '\n' << std::flush;
        if (file.is_open()) {
            file << line << '\n' << std::flush;
            if (!file) throw std::runtime_error("Cannot write CSV output");
        }
    }
};

template <class T, int N, bool Forward, bool Public>
void check_case(Results& results, int batch, int kind) {
    using Value = vec2_t<T>;
    const size_t total = size_t(N) * batch;
    const auto base = make_input(N, batch, kind);
    std::vector<Value> input(total);
    std::vector<Value> output(total + kOutputGuard,
                              encode<T>(make_float2(std::numeric_limits<float>::quiet_NaN(), 0.f)));
    std::vector<cufftDoubleComplex> reference(total);
    for (size_t i = 0; i < total; ++i) {
        input[i] = encode<T>(base[i]);
        // Compare against the values actually represented by the input dtype.
        const auto actual = decode<T>(input[i]);
        reference[i] = {double(actual.x), double(actual.y)};
    }
    for (size_t i = total; i < output.size(); ++i)
        output[i] = encode<T>(make_float2(-271.f, 137.f));

    // The input allocation has exactly batch*N values, so a sanitizer also
    // detects reads past the requested final FFT in public tail cases.
    DeviceBuffer<Value> device_input(total), device_output(output.size());
    DeviceBuffer<cufftDoubleComplex> device_reference(total);
    CUDA_CHECK(cudaMemcpy(device_input.data, input.data(), total * sizeof(Value), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_output.data, output.data(), output.size() * sizeof(Value), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_reference.data, reference.data(), total * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    ReferencePlan plan(N, batch);
    CUFFT_CHECK(cufftExecZ2Z(plan.handle, device_reference.data, device_reference.data,
                           Forward ? CUFFT_FORWARD : CUFFT_INVERSE));
    CUDA_CHECK(cudaDeviceSynchronize());

    int shared_changes = 0;
    if constexpr (Public) {
        thunderfft::ThunderFFT_global<T, N, Forward>(device_input.data, device_output.data, batch);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    } else {
        constexpr int BPB = thunderfft::batch_per_block<N>;
        constexpr int WPB = thunderfft::warp_per_block<N>;
        constexpr int shared_bytes = sizeof(Value) * ((N + pad_h(N)) * BPB + kSharedGuard);
        DeviceBuffer<int> device_guard(batch / BPB);
        CUDA_CHECK(cudaFuncSetAttribute(validate_core<T, N, Forward>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        validate_core<T, N, Forward><<<batch / BPB, dim3(32, WPB), shared_bytes>>>(
            device_input.data, device_output.data, device_guard.data);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<int> guard(batch / BPB);
        CUDA_CHECK(cudaMemcpy(guard.data(), device_guard.data, guard.size() * sizeof(int), cudaMemcpyDeviceToHost));
        shared_changes = *std::max_element(guard.begin(), guard.end());
    }
    CUDA_CHECK(cudaMemcpy(output.data(), device_output.data, output.size() * sizeof(Value), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(reference.data(), device_reference.data, total * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));

    double squared_error = 0, squared_reference = 0, max_error = 0, max_reference = 0;
    int nonfinite = 0;
    for (size_t i = 0; i < total; ++i) {
        const auto value = decode<T>(output[i]);
        if (!std::isfinite(value.x) || !std::isfinite(value.y)) ++nonfinite;
        const double dx = double(value.x) - reference[i].x;
        const double dy = double(value.y) - reference[i].y;
        squared_error += dx * dx + dy * dy;
        squared_reference += reference[i].x * reference[i].x + reference[i].y * reference[i].y;
        max_error = std::max(max_error, std::hypot(dx, dy));
        max_reference = std::max(max_reference, std::hypot(reference[i].x, reference[i].y));
    }
    int output_changes = 0;
    for (size_t i = total; i < output.size(); ++i) {
        const auto value = decode<T>(output[i]);
        output_changes += value.x != -271.f || value.y != 137.f;
    }
    const double relative_l2 = std::sqrt(squared_error / squared_reference);
    const bool pass = nonfinite == 0 && shared_changes == 0 && output_changes == 0 &&
                      std::isfinite(relative_l2) && relative_l2 <= kRelativeL2Limit;
    std::ostringstream row;
    row << std::setprecision(12) << (Public ? "public_global" : "direct_core_guarded") << ','
        << (std::is_same_v<T, half> ? "fp16" : "tf32") << ',' << N << ','
        << (Forward ? "forward" : "inverse") << ',' << input_name(kind) << ',' << batch << ','
        << relative_l2 << ',' << max_error << ',' << max_error / max_reference << ','
        << nonfinite << ',' << shared_changes << ',' << output_changes << ',' << (pass ? 1 : 0);
    results.write(row.str());
    ++results.cases;
    if (!pass) ++results.failures;
}

template <class T, int N> void check_size(Results& results, const Options& options) {
    TwiddleScope<T> twiddles(N);
    constexpr int BPB = thunderfft::batch_per_block<N>;
    std::vector<int> tails = options.quick ? std::vector<int>{1} : std::vector<int>{1, BPB - 1, BPB + 1};
    tails.erase(std::remove_if(tails.begin(), tails.end(), [](int b) { return b <= 0; }), tails.end());
    std::sort(tails.begin(), tails.end());
    tails.erase(std::unique(tails.begin(), tails.end()), tails.end());
    for (int kind = 0; kind < (options.quick ? 1 : 4); ++kind) {
        check_case<T, N, true, false>(results, options.batch, kind);
        check_case<T, N, false, false>(results, options.batch, kind);
        check_case<T, N, true, true>(results, options.batch, kind);
        check_case<T, N, false, true>(results, options.batch, kind);
        for (int batch : tails) {
            if (batch == options.batch) continue;
            check_case<T, N, true, true>(results, batch, kind);
            check_case<T, N, false, true>(results, batch, kind);
        }
    }
}

template <class T> void check_type(Results& results, const Options& options) {
    check_size<T, 64>(results, options);
    check_size<T, 128>(results, options);
    check_size<T, 256>(results, options);
    check_size<T, 512>(results, options);
    check_size<T, 1024>(results, options);
    check_size<T, 2048>(results, options);
    check_size<T, 4096>(results, options);
    // Reinitializing after Finalize and after several sizes also checks that
    // no stale FP16 unit coefficient pointer survives cache teardown.
    TwiddleScope<T> reinitialized(64);
    check_case<T, 64, true, true>(results, 1, 0);
    check_case<T, 64, false, true>(results, 1, 0);
}

int parse_int(const std::string& text, const char* name) {
    int value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    return value;
}

template <class Operation>
void expect_invalid_argument(Operation operation, const char* description) {
    try {
        operation();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string("Host API check did not reject ") + description);
}

template <class T> void check_host_arguments() {
    // These arguments must be handled before any CUDA calls. Host pointers
    // appear only in calls that must reject them for nullness or aliasing.
    vec2_t<T> value{};
    expect_invalid_argument([] {
        thunderfft::ThunderFFT_global<T, 64>(nullptr, nullptr, -1);
    }, "negative batch");
    expect_invalid_argument([&] {
        thunderfft::ThunderFFT_global<T, 64>(nullptr, &value, 1);
    }, "null input");
    expect_invalid_argument([&] {
        thunderfft::ThunderFFT_global<T, 64>(&value, nullptr, 1);
    }, "null output");
    expect_invalid_argument([&] {
        thunderfft::ThunderFFT_global<T, 64>(&value, &value, 1);
    }, "aliased input/output");
    thunderfft::ThunderFFT_global<T, 64>(nullptr, nullptr, 0);
    for (int n : {-1, 0, 32, 8192})
        expect_invalid_argument([n] {
            thunderfft::ThunderFFTInitialize<T>(n);
        }, "unsupported initialization length");
}

void help(const char* executable) {
    std::cout << "Usage: " << executable << " [--device N] [--batch N] [--csv PATH] [--quick] [--host-only]\n"
              << "Checks all seven FFT sizes, FP16/TF32, forward/inverse, direct/public paths.\n"
              << "Default batch: 64; --quick: batch 16, Gaussian input and public batch 1.\n"
              << "--batch must be positive and a multiple of 16 (all direct-core tiles).\n"
              << "Full mode also checks impulse, tone, constant and public tail batches.\n"
              << "Reference: cuFFT FP64 of the actual represented input.\n"
              << "Pass: finite output, relative L2 <= 0.005 and unchanged memory guards.\n"
              << "--host-only: check invalid public API arguments and zero-batch handling without CUDA.\n"
              << "CSV is always written to stdout; --csv also writes a file. No timing.\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") { help(argv[0]); return 0; }
            if (argument == "--quick") { options.quick = true; continue; }
            if (argument == "--host-only") { options.host_only = true; continue; }
            if (argument != "--device" && argument != "--batch" && argument != "--csv")
                throw std::runtime_error("Unknown option: " + argument);
            if (++i >= argc) throw std::runtime_error("Missing value for " + argument);
            if (argument == "--device") options.device = parse_int(argv[i], "device");
            else if (argument == "--batch") {
                options.batch = parse_int(argv[i], "batch");
                options.batch_set = true;
            } else options.csv = argv[i];
        }
        if (options.quick && !options.batch_set) options.batch = 16;
        if (options.device < 0) throw std::runtime_error("Device must be nonnegative");
        if (options.batch <= 0 || options.batch % 16 != 0)
            throw std::runtime_error("Batch must be positive and a multiple of 16");
        // Keep the default validation bounded; callers can explicitly increase
        // batch without changing this correctness program into a benchmark.
        if (options.batch > 65536) throw std::runtime_error("Batch must not exceed 65536");
        if (options.host_only && !options.csv.empty())
            throw std::runtime_error("--csv requires GPU validation; omit it with --host-only");
        check_host_arguments<float>();
        check_host_arguments<half>();
        std::cerr << "Host API validation: 18 checks passed before any CUDA calls\n";
        if (options.host_only) return 0;
        CUDA_CHECK(cudaSetDevice(options.device));
        Results results(options.csv);
        check_type<float>(results, options);
        check_type<half>(results, options);
        std::cerr << "Validation: " << results.cases << " cases, " << results.failures
                  << " failures; relative L2 limit " << kRelativeL2Limit << '\n';
        return results.failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Validation error: " << error.what() << '\n';
        return 2;
    }
}
