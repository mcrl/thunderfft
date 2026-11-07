// Native driver for Table 7 candidates. See kernels/provenance.json.
// Repeated FFTs are a compute-throughput workload; their outputs can overflow.
// Default runs validate R=1. Explicit --timing-only runs are unvalidated diagnostics.
#include "kernels/recovered.cuh"
#include "kernels/historical_branch_51e346c.cuh"
#include "kernels/historical_uniform_5e37821.cuh"
#include "kernels/historical_e4aa1b0.cuh"
#include "kernels/historical_652808b.cuh"
#include "kernels/radix16.cuh"
#include "kernels/radix4_shared.cuh"

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int kN = 64;
constexpr int kTransformsPerBlock = 16;
constexpr int kThreadsPerBlock = 32;
constexpr int kGuardElements = 16;
constexpr int kCheckBatch = 64;
constexpr int kWarmups = 3;
constexpr int kUniformSharedBytes = 8704;  // Original 5e37821 host launch.
constexpr double kErrorThreshold = 0.005;
constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::uint32_t kSeed = 1234567;

void check_cuda(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(expression) + ": " +
                                 cudaGetErrorString(status));
    }
}
#define CUDA_CHECK(expression) check_cuda((expression), #expression)

enum class Variant { radix16, radix4_shared, branch, uniform, historical_e4aa1b0, historical_652808b };

struct DeviceBuffer {
    cuFloatComplex* pointer = nullptr;
    std::size_t elements = 0;
    explicit DeviceBuffer(std::size_t count) : elements(count) {
        CUDA_CHECK(cudaMalloc(&pointer, count * sizeof(cuFloatComplex)));
    }
    ~DeviceBuffer() { if (pointer) cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    cuFloatComplex* data() const { return pointer + kGuardElements; }
    std::size_t bytes() const { return elements * sizeof(cuFloatComplex); }
};

struct Event {
    cudaEvent_t value = nullptr;
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { if (value) cudaEventDestroy(value); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};

int parse_int(const char* text, int minimum, int maximum, const char* name) {
    if (!text[0]) throw std::runtime_error(std::string("invalid ") + name);
    for (const char* p = text; *p; ++p) {
        if (*p < '0' || *p > '9')
            throw std::runtime_error(std::string("invalid ") + name);
    }
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (!end || *end || value < minimum || value > maximum)
        throw std::runtime_error(std::string("invalid ") + name);
    return static_cast<int>(value);
}

void help() {
    std::printf(
        "Usage: benchmark VARIANT [batch=65536] [R=1000] [trials=3] [--profile|--timing-only]\n"
        "VARIANT: register_branch | register_uniform\n"
        "Timing-only historical microbenchmarks: historical_e4aa1b0 | historical_652808b\n"
        "Experimental only: reconstructed_radix16 | reconstructed_radix4_compute\n"
        "N=64, forward, unnormalized TF32, batch must be a multiple of 16.\n"
        "Default: validate R=1 Gaussian and tone inputs (batch64) against CPU FP64 DFT.\n"
        "Timing: reset immutable input before each launch; report (t_2R-t_R)/R.\n"
        "Repeated outputs may overflow and are not numerical validation.\n"
        "--timing-only: skip accuracy/guard checks; one warmup per repeat count;\n"
        "               constant input (0.125,-0.25); NaN errors, -1 check counts, passed=0.\n"
        "--profile: after the two validation launches, launch R once and exit;\n"
        "           no warmups or timing rows (for an external profiler).\n"
        "CSV: variant,N,batch,precision,trial,inside_repeats,t_R_us,t_2R_us,"
        "compute_us,relative_l2,max_transform_relative_l2,nonfinite_outputs,"
        "output_guard_changed,passed\n");
}

void launch(Variant variant, cuFloatComplex* data, int batch, unsigned repeats,
            const cuFloatComplex* twiddles) {
    const unsigned blocks = static_cast<unsigned>(batch / kTransformsPerBlock);
    if (variant == Variant::radix16) {
        thff_table7_legacy::fft_kernel_radix16_batch16<<<
            blocks, kThreadsPerBlock>>>(data, nullptr, repeats);
    } else if (variant == Variant::radix4_shared) {
        thff_table7_legacy::fft_kernel_radix4_shared_batch16<<<
            blocks, kThreadsPerBlock>>>(data, nullptr, repeats);
    } else if (variant == Variant::uniform) {
        thff_table7_5e37821::fft_kernel_radix64_batch16<<<
            blocks, kThreadsPerBlock, kUniformSharedBytes>>>(data, twiddles, repeats);
    } else if (variant == Variant::historical_e4aa1b0) {
        thff_table7_e4aa1b0::fft_kernel_radix64_batch16<<<
            blocks, kThreadsPerBlock>>>(data, twiddles, repeats);
    } else if (variant == Variant::historical_652808b) {
        thff_table7_652808b::fft_kernel_radix64_batch16<<<
            blocks, kThreadsPerBlock, kUniformSharedBytes>>>(data, twiddles, repeats);
    } else {
        thff_table7_51e346c::fft_kernel_radix64_batch16<<<
            blocks, kThreadsPerBlock>>>(data, twiddles, repeats);
    }
    CUDA_CHECK(cudaGetLastError());
}

cuFloatComplex guard_value() {
    const std::uint32_t bits[2] = {0x4e123456u, 0xce654321u};
    cuFloatComplex guard;
    static_assert(sizeof(guard) == sizeof(bits), "complex guard layout");
    std::memcpy(&guard, bits, sizeof(guard));
    return guard;
}

std::vector<cuFloatComplex> guarded_input(int batch, bool tone) {
    const std::size_t count = static_cast<std::size_t>(batch) * kN;
    std::vector<cuFloatComplex> values(count + 2 * kGuardElements, guard_value());
    std::mt19937 generator(kSeed);
    // Explicit Box-Muller avoids implementation-defined normal_distribution.
    const auto uniform = [&]() {
        return (static_cast<double>(generator()) + 0.5) / 4294967296.0;
    };
    for (int b = 0; b < batch; ++b) {
        const int frequency = (7 * b + 3) % kN;
        for (int n = 0; n < kN; ++n) {
            double real, imag;
            if (tone) {
                const double phase = 2 * kPi * frequency * n / kN + 0.123 * b;
                const double amplitude = 0.25 + 0.005 * (b % 8);
                real = amplitude * std::cos(phase);
                imag = amplitude * std::sin(phase);
            } else {
                const double radius = 0.2 * std::sqrt(-2 * std::log(uniform()));
                const double angle = 2 * kPi * uniform();
                real = std::max(-0.75, std::min(0.75, radius * std::cos(angle)));
                imag = std::max(-0.75, std::min(0.75, radius * std::sin(angle)));
            }
            values[kGuardElements + static_cast<std::size_t>(b) * kN + n] =
                make_cuFloatComplex(static_cast<float>(real), static_cast<float>(imag));
        }
    }
    return values;
}

std::uint64_t host_guard_changes(const std::vector<cuFloatComplex>& values) {
    const auto expected = guard_value();
    std::uint64_t changed = 0;
    for (int i = 0; i < kGuardElements; ++i) {
        changed += std::memcmp(&values[i], &expected, sizeof(expected)) != 0;
        changed += std::memcmp(&values[values.size() - kGuardElements + i],
                               &expected, sizeof(expected)) != 0;
    }
    return changed;
}

std::uint64_t device_guard_changes(const DeviceBuffer& work) {
    std::vector<cuFloatComplex> guards(2 * kGuardElements);
    CUDA_CHECK(cudaMemcpy(guards.data(), work.pointer,
                          kGuardElements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(guards.data() + kGuardElements,
                          work.pointer + work.elements - kGuardElements,
                          kGuardElements * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
    return host_guard_changes(guards);
}

struct Validation {
    double error_squared = 0;
    double reference_squared = 0;
    double max_transform_relative_l2 = 0;
    std::uint64_t nonfinite_outputs = 0;
    std::uint64_t guard_changed = 0;
    double relative_l2() const {
        return std::sqrt(error_squared / reference_squared);
    }
    bool passed() const {
        return reference_squared > 0 && std::isfinite(relative_l2()) &&
            relative_l2() <= kErrorThreshold &&
            std::isfinite(max_transform_relative_l2) &&
            max_transform_relative_l2 <= kErrorThreshold &&
            nonfinite_outputs == 0 && guard_changed == 0;
    }
};

Validation validate_r1(Variant variant, bool tone, const cuFloatComplex* twiddles) {
    const auto input = guarded_input(kCheckBatch, tone);
    auto output = input;
    DeviceBuffer work(input.size());
    CUDA_CHECK(cudaMemcpy(work.pointer, input.data(), work.bytes(), cudaMemcpyHostToDevice));
    launch(variant, work.data(), kCheckBatch, 1, twiddles);
    CUDA_CHECK(cudaMemcpy(output.data(), work.pointer, work.bytes(), cudaMemcpyDeviceToHost));

    std::complex<double> weights[kN][kN];
    for (int k = 0; k < kN; ++k) {
        for (int n = 0; n < kN; ++n) {
            const double angle = -2 * kPi * k * n / kN;
            weights[k][n] = {std::cos(angle), std::sin(angle)};
        }
    }
    Validation result;
    result.guard_changed = host_guard_changes(output);
    for (int b = 0; b < kCheckBatch; ++b) {
        double transform_error = 0, transform_reference = 0;
        for (int k = 0; k < kN; ++k) {
            std::complex<double> reference(0, 0);
            for (int n = 0; n < kN; ++n) {
                const auto x = input[kGuardElements + b * kN + n];
                reference += std::complex<double>(x.x, x.y) * weights[k][n];
            }
            const auto actual = output[kGuardElements + b * kN + k];
            transform_reference += std::norm(reference);
            if (!std::isfinite(actual.x) || !std::isfinite(actual.y)) {
                ++result.nonfinite_outputs;
                transform_error = INFINITY;
            } else {
                transform_error += std::norm(std::complex<double>(actual.x, actual.y) - reference);
            }
        }
        result.error_squared += transform_error;
        result.reference_squared += transform_reference;
        const double error = std::sqrt(transform_error / transform_reference);
        result.max_transform_relative_l2 = std::max(result.max_transform_relative_l2, error);
    }
    std::fprintf(stderr,
        "validation_input=%s validation_batch=%d R=1 reference=cpu_fp64_dft "
        "relative_l2=%.12g max_transform_relative_l2=%.12g nonfinite_outputs=%llu "
        "output_guard_changed=%llu threshold=%.6g passed=%d\n",
        tone ? "complex_tone" : "bounded_complex_gaussian", kCheckBatch,
        result.relative_l2(), result.max_transform_relative_l2,
        static_cast<unsigned long long>(result.nonfinite_outputs),
        static_cast<unsigned long long>(result.guard_changed), kErrorThreshold,
        result.passed() ? 1 : 0);
    return result;
}

void reset(const DeviceBuffer& immutable, const DeviceBuffer& work) {
    CUDA_CHECK(cudaMemcpy(work.pointer, immutable.pointer, work.bytes(), cudaMemcpyDeviceToDevice));
}

double timed_launch(Variant variant, const DeviceBuffer& immutable,
                    const DeviceBuffer& work, int batch, unsigned repeats,
                    const Event& start, const Event& stop,
                    const cuFloatComplex* twiddles) {
    // The default-stream reset precedes the start event and is not timed.
    reset(immutable, work);
    CUDA_CHECK(cudaEventRecord(start.value));
    launch(variant, work.data(), batch, repeats, twiddles);
    CUDA_CHECK(cudaEventRecord(stop.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start.value, stop.value));
    if (!(elapsed_ms > 0) || !std::isfinite(elapsed_ms))
        throw std::runtime_error("invalid CUDA event elapsed time");
    return static_cast<double>(elapsed_ms) * 1000;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        // This path intentionally precedes every CUDA API call.
        if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                          std::strcmp(argv[1], "-h") == 0)) {
            help();
            return 0;
        }
        const bool profile = argc >= 3 && std::strcmp(argv[argc - 1], "--profile") == 0;
        const bool timing_only = argc >= 3 && std::strcmp(argv[argc - 1], "--timing-only") == 0;
        const int warmups = profile ? 0 : (timing_only ? 1 : kWarmups);
        const int positional_count = argc - ((profile || timing_only) ? 1 : 0);
        if (positional_count < 2 || positional_count > 5)
            throw std::runtime_error("usage: benchmark VARIANT [batch] [R] [trials] [--profile|--timing-only]");
        const std::string variant_name = argv[1];
        Variant variant;
        if (variant_name == "reconstructed_radix16") variant = Variant::radix16;
        else if (variant_name == "reconstructed_radix4_compute") variant = Variant::radix4_shared;
        else if (variant_name == "register_branch") variant = Variant::branch;
        else if (variant_name == "register_uniform") variant = Variant::uniform;
        else if (variant_name == "historical_e4aa1b0") variant = Variant::historical_e4aa1b0;
        else if (variant_name == "historical_652808b") variant = Variant::historical_652808b;
        else throw std::runtime_error("unknown or unrecovered variant; see --help for historical and experimental candidates");
        if ((variant == Variant::historical_e4aa1b0 || variant == Variant::historical_652808b) && !timing_only)
            throw std::runtime_error("historical partial kernels require --timing-only; they do not implement a validated complete FFT");
        const int batch = positional_count > 2 ? parse_int(argv[2], 16, 1048576, "batch") : 65536;
        const int repeats = positional_count > 3 ? parse_int(argv[3], 1, 1000000, "R") : 1000;
        const int trials = positional_count > 4 ? parse_int(argv[4], 1, 100, "trials") : 3;
        if (batch % kTransformsPerBlock)
            throw std::runtime_error("batch must be a positive multiple of 16");

        CUDA_CHECK(cudaSetDevice(0));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, 0));
        if (properties.major < 8)
            throw std::runtime_error("TF32 MMA requires compute capability >= 8.0");
        int driver = 0, runtime = 0;
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        cudaFuncAttributes attributes{};
        if (variant == Variant::radix16) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_legacy::fft_kernel_radix16_batch16));
        } else if (variant == Variant::radix4_shared) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_legacy::fft_kernel_radix4_shared_batch16));
        } else if (variant == Variant::uniform) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_5e37821::fft_kernel_radix64_batch16));
        } else if (variant == Variant::historical_e4aa1b0) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_e4aa1b0::fft_kernel_radix64_batch16));
        } else if (variant == Variant::historical_652808b) {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_652808b::fft_kernel_radix64_batch16));
        } else {
            CUDA_CHECK(cudaFuncGetAttributes(&attributes, thff_table7_51e346c::fft_kernel_radix64_batch16));
        }
        std::fprintf(stderr,
            "gpu=%s device=0 sm=%d.%d sm_count=%d driver=%d runtime=%d "
            "cuda_compiler=%d.%d.%d source=artifact/table7/ablation/benchmark.cu\n"
            "variant=%s N=64 precision=tf32 direction=forward normalization=none "
            "batch=%d R=%d trials=%d warmups_per_repeat=%d block=32 grid=%d "
            "dynamic_shared_bytes=%d static_shared_bytes=%zu registers=%d local_bytes=%zu "
            "input_seed=%u guard_elements_each_side=%d mode=%s\n"
            "timing=(t_2R-t_R)/R immutable_input_reset_before_every_launch=1 "
            "clock_policy=unchanged "
            "repeated_output_may_overflow=1 repeated_output_accuracy_claim=0\n",
            properties.name, properties.major, properties.minor,
            properties.multiProcessorCount, driver, runtime,
            __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__,
            variant_name.c_str(), batch, repeats, trials, warmups,
            batch / kTransformsPerBlock,
            (variant == Variant::uniform || variant == Variant::historical_652808b) ? kUniformSharedBytes : 0,
            attributes.sharedSizeBytes, attributes.numRegs, attributes.localSizeBytes,
            kSeed, kGuardElements, profile ? "profile" : (timing_only ? "timing_only" : "timing"));
        std::fprintf(stderr,
            "accuracy_status=%s timing_only=%d R1_validation=%d output_guard_checks=%d\n",
            timing_only ? "unchecked" : "validation_requested", timing_only ? 1 : 0,
            timing_only ? 0 : 1, timing_only ? 0 : 1);
        std::fprintf(stderr, "timing_input=%s\n",
                     timing_only ? "constant_0.125_minus0.25" : "bounded_complex_gaussian_seed1234567");
        std::fprintf(stderr, "gpu_uuid=GPU-");
        for (int byte = 0; byte < 16; ++byte) {
            if (byte == 4 || byte == 6 || byte == 8 || byte == 10)
                std::fprintf(stderr, "-");
            std::fprintf(stderr, "%02x", static_cast<unsigned char>(properties.uuid.bytes[byte]));
        }
        std::fprintf(stderr, "\n");

        // Preserve the original host expressions: float PI in 51e346c,
        // double PI followed by conversion to float theta in 5e37821.
        // Both LUTs are initialized outside the timed interval.
        DeviceBuffer twiddles(kN);
        cuFloatComplex host_twiddles[kN];
        for (int i = 0; i < kN; ++i) {
            float theta = variant == Variant::uniform
                ? static_cast<float>(-2.0f * 3.14159265358979323846 * i / 64.0f)
                : -2.0f * 3.14159265358979323846f * i / 64.0f;
            host_twiddles[i] = make_cuFloatComplex(cosf(theta), sinf(theta));
        }
        CUDA_CHECK(cudaMemcpy(twiddles.pointer, host_twiddles,
                              sizeof(host_twiddles), cudaMemcpyHostToDevice));
        Validation validation;
        if (!timing_only) {
            const Validation gaussian = validate_r1(variant, false, twiddles.pointer);
            const Validation tone = validate_r1(variant, true, twiddles.pointer);
            if (!gaussian.passed() || !tone.passed())
                throw std::runtime_error("independent R=1 accuracy gate failed; no timing performed");
            validation = gaussian;
            validation.error_squared += tone.error_squared;
            validation.reference_squared += tone.reference_squared;
            validation.max_transform_relative_l2 = std::max(
                gaussian.max_transform_relative_l2, tone.max_transform_relative_l2);
            validation.nonfinite_outputs += tone.nonfinite_outputs;
            validation.guard_changed += tone.guard_changed;
        }

        const auto input = timing_only
            ? std::vector<cuFloatComplex>(static_cast<std::size_t>(batch) * kN + 2 * kGuardElements,
                                         make_cuFloatComplex(0.125f, -0.25f))
            : guarded_input(batch, false);
        DeviceBuffer immutable(input.size()), work(input.size());
        CUDA_CHECK(cudaMemcpy(immutable.pointer, input.data(), immutable.bytes(), cudaMemcpyHostToDevice));
        std::printf("variant,N,batch,precision,trial,inside_repeats,t_R_us,t_2R_us,"
                    "compute_us,relative_l2,max_transform_relative_l2,nonfinite_outputs,"
                    "output_guard_changed,passed\n");
        std::fflush(stdout);
        if (profile) {
            // Exactly the third recovered-kernel launch in this process. An
            // external profiler can skip the two R=1 validation launches.
            reset(immutable, work);
            launch(variant, work.data(), batch, static_cast<unsigned>(repeats), twiddles.pointer);
            CUDA_CHECK(cudaDeviceSynchronize());
            if (device_guard_changes(work))
                throw std::runtime_error("profile launch changed output guards");
            std::fprintf(stderr, "profile_launch_complete=1 profile_R=%d profile_warmups=0 "
                                 "profile_timing_rows=0\n", repeats);
            return 0;
        }

        for (int warmup = 0; warmup < warmups; ++warmup) {
            reset(immutable, work);
            launch(variant, work.data(), batch, static_cast<unsigned>(repeats), twiddles.pointer);
            reset(immutable, work);
            launch(variant, work.data(), batch, static_cast<unsigned>(2 * repeats), twiddles.pointer);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!timing_only && device_guard_changes(work))
            throw std::runtime_error("warmup launch changed output guards");
        Event start, stop;
        for (int trial = 0; trial < trials; ++trial) {
            const double t_r = timed_launch(variant, immutable, work, batch,
                                            repeats, start, stop, twiddles.pointer);
            std::uint64_t guard_changed = timing_only ? 0 : device_guard_changes(work);
            const double t_2r = timed_launch(variant, immutable, work, batch,
                                             2 * repeats, start, stop, twiddles.pointer);
            if (!timing_only) guard_changed += device_guard_changes(work);
            const double compute_us = (t_2r - t_r) / repeats;
            const bool timing_passed = std::isfinite(compute_us) && compute_us > 0;
            const bool passed = !timing_only && validation.passed() && guard_changed == 0 && timing_passed;
            std::printf("%s,64,%d,tf32,%d,%d,%.12g,%.12g,%.12g,%.12g,%.12g,%lld,%lld,%d\n",
                variant_name.c_str(), batch, trial, repeats, t_r, t_2r, compute_us,
                timing_only ? NAN : validation.relative_l2(),
                timing_only ? NAN : validation.max_transform_relative_l2,
                timing_only ? -1LL : static_cast<long long>(validation.nonfinite_outputs),
                timing_only ? -1LL : static_cast<long long>(validation.guard_changed + guard_changed),
                passed ? 1 : 0);
            std::fflush(stdout);
            if (!timing_passed || (!timing_only && !passed))
                throw std::runtime_error("timing difference nonpositive or output guards changed");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 1;
    }
}
