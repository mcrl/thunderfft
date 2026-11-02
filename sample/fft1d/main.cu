#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <thunderfft/thunderfft.cuh>

namespace {

constexpr int kN = 64;
constexpr int kBatchPerBlock = 16;
// A regression threshold for this deterministic float forward example, not an
// accuracy guarantee for every input, FFT size, direction, or data type.
constexpr double kTolerance = 5e-3;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
    }
}

void check_cufft(cufftResult status, const char* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": cuFFT status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

void usage(std::ostream& out, const char* executable) {
    out << "Usage: " << executable << " [batch] [device]\n"
        << "  batch   Positive multiple of 16 (default: 16).\n"
        << "  device  Nonnegative CUDA device ordinal (default: 0).\n"
        << "Checks an unscaled, out-of-place, complex float forward FFT "
           "(N=64) against cuFFT.\n"
        << "PASS requires finite outputs and relative L2/Linf <= 5e-3 "
           "for this example input.\n";
}

int parse_nonnegative(const char* text, const char* name) {
    int value = 0;
    const char* end = text + std::strlen(text);
    const auto parsed = std::from_chars(text, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end || value < 0) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a nonnegative integer in int range");
    }
    return value;
}

// Bounded, bin-aligned complex tones; repeating the pattern keeps larger batches
// deterministic without growing the amplitude or frequency with the batch size.
void make_input(std::vector<float2>& input, int batch) {
    constexpr double two_pi = 6.28318530717958647692;
    for (int b = 0; b < batch; ++b) {
        const double frequency = 1 + b % 13;
        const double amplitude = 0.5 + 0.1 * (b % 8);
        for (int i = 0; i < kN; ++i) {
            const double phase = two_pi * i / kN;
            input[static_cast<size_t>(b) * kN + i] = make_float2(
                static_cast<float>(amplitude * std::cos(frequency * phase) +
                                   0.1 * std::cos(7.0 * phase)),
                static_cast<float>(amplitude * std::sin(frequency * phase) +
                                   0.1 * std::sin(7.0 * phase)));
        }
    }
}

struct Resources {
    float2* input = nullptr;
    float2* output = nullptr;
    float2* reference = nullptr;
    cufftHandle plan = 0;
    bool twiddles_started = false;

    // This also runs when an allocation, CUDA operation, or comparison throws.
    ~Resources() {
        if (plan) cufftDestroy(plan);
        if (twiddles_started) thunderfft::ThunderFFTFinalize<float>();
        if (reference) cudaFree(reference);
        if (output) cudaFree(output);
        if (input) cudaFree(input);
    }
};

bool run(int batch, int device) {
    check_cuda(cudaSetDevice(device), "cudaSetDevice");
    const size_t count = static_cast<size_t>(kN) * batch;
    const size_t bytes = count * sizeof(float2);
    std::vector<float2> input(count), output(count), reference(count);
    make_input(input, batch);

    Resources resources;
    check_cuda(cudaMalloc(&resources.input, bytes), "cudaMalloc(input)");
    check_cuda(cudaMalloc(&resources.output, bytes), "cudaMalloc(output)");
    check_cuda(cudaMalloc(&resources.reference, bytes), "cudaMalloc(reference)");
    check_cuda(cudaMemcpy(resources.input, input.data(), bytes,
                          cudaMemcpyHostToDevice), "cudaMemcpy(input)");

    resources.twiddles_started = true;
    thunderfft::ThunderFFTInitialize<float>(kN);
    check_cuda(cudaGetLastError(), "ThunderFFTInitialize");
    check_cuda(cudaDeviceSynchronize(), "ThunderFFTInitialize synchronization");

    // Each signal is contiguous: input[b * kN + i]. Input and output are distinct.
    thunderfft::ThunderFFT_global<float, kN, true>(
        resources.input, resources.output, batch, /*stream=*/0);
    check_cuda(cudaGetLastError(), "ThunderFFT launch");
    // The current launch wrapper discards its synchronization status, so check
    // explicitly before accepting or copying its result.
    check_cuda(cudaDeviceSynchronize(), "ThunderFFT synchronization");

    check_cufft(cufftPlan1d(&resources.plan, kN, CUFFT_C2C, batch), "cufftPlan1d");
    check_cufft(cufftExecC2C(resources.plan,
                           reinterpret_cast<cufftComplex*>(resources.input),
                           reinterpret_cast<cufftComplex*>(resources.reference),
                           CUFFT_FORWARD), "cufftExecC2C");
    check_cuda(cudaGetLastError(), "cuFFT launch");
    check_cuda(cudaDeviceSynchronize(), "cuFFT synchronization");
    check_cuda(cudaMemcpy(output.data(), resources.output, bytes,
                          cudaMemcpyDeviceToHost), "cudaMemcpy(output)");
    check_cuda(cudaMemcpy(reference.data(), resources.reference, bytes,
                          cudaMemcpyDeviceToHost), "cudaMemcpy(reference)");

    double squared_error = 0.0, squared_reference = 0.0;
    double max_error = 0.0, max_reference = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const float2 a = output[i], b = reference[i];
        if (!std::isfinite(a.x) || !std::isfinite(a.y) ||
            !std::isfinite(b.x) || !std::isfinite(b.y)) {
            throw std::runtime_error("nonfinite output at complex element " +
                                     std::to_string(i));
        }
        const double error = std::hypot(static_cast<double>(a.x) - b.x,
                                        static_cast<double>(a.y) - b.y);
        const double magnitude = std::hypot(static_cast<double>(b.x),
                                            static_cast<double>(b.y));
        squared_error += error * error;
        squared_reference += magnitude * magnitude;
        max_error = std::max(max_error, error);
        max_reference = std::max(max_reference, magnitude);
    }
    const double l2 = squared_reference > 0.0
        ? std::sqrt(squared_error / squared_reference) : std::sqrt(squared_error);
    const double linf = max_reference > 0.0 ? max_error / max_reference : max_error;
    const bool passed = std::isfinite(l2) && std::isfinite(linf) &&
                        l2 <= kTolerance && linf <= kTolerance;
    std::cout << "ThunderFFT vs cuFFT: float2, N=" << kN
              << ", forward (unscaled), batch=" << batch
              << ", device=" << device << '\n'
              << std::scientific << std::setprecision(6)
              << "Relative L2:   " << l2 << '\n'
              << "Relative Linf: " << linf << '\n'
              << "Tolerance:     " << kTolerance << " (this example input)\n"
              << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && (std::strcmp(argv[1], "--help") == 0 ||
                      std::strcmp(argv[1], "-h") == 0)) {
        usage(std::cout, argv[0]);
        return EXIT_SUCCESS;
    }
    try {
        if (argc > 3) throw std::invalid_argument("too many arguments");
        const int batch = argc > 1 ? parse_nonnegative(argv[1], "batch") : 16;
        const int device = argc > 2 ? parse_nonnegative(argv[2], "device") : 0;
        if (batch <= 0 || batch % kBatchPerBlock != 0) {
            throw std::invalid_argument("batch must be a positive multiple of 16; "
                                        "partial FFT blocks are unsupported");
        }
        if (static_cast<size_t>(batch) >
            std::numeric_limits<size_t>::max() / (kN * sizeof(float2))) {
            throw std::invalid_argument("batch allocation size overflows size_t");
        }
        return run(batch, device) ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
