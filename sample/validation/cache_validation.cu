// Deliberately avoid thunderfft.cuh: the cache tests need no FFT/MMA kernels,
// and can run on older CUDA devices when built with CUDA 12.
#include <thunderfft/detail/thunderfft_impl_twiddle.cuh>

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef THUNDERFFT_CACHE_MIN_CC
#define THUNDERFFT_CACHE_MIN_CC 70
#endif

namespace {
int checks = 0;

void check_cuda(cudaError_t error, const char* expression) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(error));
}
#define CACHE_CUDA(expression) check_cuda((expression), #expression)

void require(bool condition, const std::string& description) {
    if (!condition) throw std::runtime_error("Cache check failed: " + description);
    ++checks;
}

template <class T> using Pair = typename thunderfft::TwiddleCache<T>::TwiddleT;
template <class T> struct Snapshot {
    const Pair<T>* pointer = nullptr;
    int n = 0;
};

template <class T> Snapshot<T> snapshot() {
    Snapshot<T> value;
    if constexpr (std::is_same_v<T, float>) {
        CACHE_CUDA(cudaMemcpyFromSymbol(&value.pointer, thunderfft::detail::d_twiddle_f,
                                        sizeof(value.pointer)));
        CACHE_CUDA(cudaMemcpyFromSymbol(&value.n, thunderfft::detail::d_twiddle_N_f,
                                        sizeof(value.n)));
    } else {
        CACHE_CUDA(cudaMemcpyFromSymbol(&value.pointer, thunderfft::detail::d_twiddle_h,
                                        sizeof(value.pointer)));
        CACHE_CUDA(cudaMemcpyFromSymbol(&value.n, thunderfft::detail::d_twiddle_N_h,
                                        sizeof(value.n)));
    }
    return value;
}

template <class T> void check_unit_coefficients() {
    if constexpr (std::is_same_v<T, half>) {
        std::vector<half2> expected;
        thunderfft::build_unit_twiddle_host(expected);
        std::vector<half2> actual(expected.size());
        CACHE_CUDA(cudaMemcpyFromSymbol(actual.data(), thunderfft::detail::d_unit_twiddle_half2,
                                        actual.size() * sizeof(half2)));
        require(std::memcmp(actual.data(), expected.data(), actual.size() * sizeof(half2)) == 0,
                "FP16 unit coefficient table must match its host builder");
    }
}

template <class T> Snapshot<T> check_live(int device, int n, const char* stage) {
    const auto value = snapshot<T>();
    require(value.pointer != nullptr, "live twiddle symbol must be non-null");
    require(value.n == n, "twiddle length symbol must match initialization");
    cudaPointerAttributes attributes{};
    CACHE_CUDA(cudaPointerGetAttributes(&attributes, value.pointer));
    require(attributes.type == cudaMemoryTypeDevice, "twiddle pointer must name device memory");
    require(attributes.device == device, "twiddle allocation must belong to the current device");

    Pair<T> first[2];
    CACHE_CUDA(cudaMemcpy(first, value.pointer, sizeof(first), cudaMemcpyDeviceToHost));
    float2 zero, one;
    if constexpr (std::is_same_v<T, half>) {
        zero = __half22float2(first[0]);
        one = __half22float2(first[1]);
    } else {
        zero = first[0];
        one = first[1];
    }
    constexpr double tau = 6.283185307179586476925286766559;
    const double tolerance = std::is_same_v<T, half> ? 1e-3 : 1e-6;
    require(zero.x == 1.f && zero.y == 0.f, "twiddle W[0] must equal 1+0i");
    require(std::abs(one.x - std::cos(tau / n)) <= tolerance &&
            std::abs(one.y - std::sin(tau / n)) <= tolerance,
            "twiddle W[1] must use the current transform length");
    check_unit_coefficients<T>();
    std::cout << (std::is_same_v<T, half> ? "fp16" : "tf32") << ',' << device << ','
              << stage << ',' << n << ',' << value.pointer << ",1\n";
    return value;
}

template <class T> void check_released(int device, const char* stage) {
    const auto value = snapshot<T>();
    require(value.pointer == nullptr, "Finalize must clear the device pointer symbol");
    require(value.n == 0, "Finalize must clear the device length symbol");
    std::cout << (std::is_same_v<T, half> ? "fp16" : "tf32") << ',' << device << ','
              << stage << ",0,0,1\n";
}

template <class T> void check_lifecycle(int device) {
    CACHE_CUDA(cudaSetDevice(device));
    thunderfft::ThunderFFTInitialize<T>(64);
    const auto first = check_live<T>(device, 64, "initialize");
    thunderfft::ThunderFFTInitialize<T>(64);
    const auto hit = check_live<T>(device, 64, "same_length_cache_hit");
    require(first.pointer == hit.pointer, "same-length initialization must retain the allocation");
    thunderfft::ThunderFFTInitialize<T>(128);
    check_live<T>(device, 128, "replace_length");
    thunderfft::ThunderFFTFinalize<T>();
    check_released<T>(device, "finalize");
    thunderfft::ThunderFFTFinalize<T>();
    check_released<T>(device, "repeated_finalize");
    thunderfft::ThunderFFTInitialize<T>(64);
    check_live<T>(device, 64, "reinitialize_after_finalize");
    thunderfft::ThunderFFTFinalize<T>();
    check_released<T>(device, "final_cleanup");
}

template <class T> void check_type(const std::vector<int>& devices) {
    for (int device : devices) check_lifecycle<T>(device);
    // Keep all devices' caches live together, using different transform sizes.
    // Returning to an earlier device must recover its original pointer/symbols.
    std::vector<Snapshot<T>> live;
    for (size_t index = 0; index < devices.size(); ++index) {
        CACHE_CUDA(cudaSetDevice(devices[index]));
        const int n = index % 2 == 0 ? 64 : 128;
        thunderfft::ThunderFFTInitialize<T>(n);
        live.push_back(check_live<T>(devices[index], n, "cross_device_initialize"));
    }
    for (size_t index = 0; index < devices.size(); ++index) {
        CACHE_CUDA(cudaSetDevice(devices[index]));
        const auto returned = check_live<T>(devices[index], live[index].n, "return_to_device");
        require(returned.pointer == live[index].pointer,
                "another device's initialization must not replace this device's allocation");
        thunderfft::ThunderFFTInitialize<T>(live[index].n);
        require(snapshot<T>().pointer == live[index].pointer,
                "cache hit after device switch must retain this device's allocation");
    }
    for (int device : devices) {
        CACHE_CUDA(cudaSetDevice(device));
        thunderfft::ThunderFFTFinalize<T>();
        check_released<T>(device, "cross_device_cleanup");
    }
}

void cleanup(const std::vector<int>& devices, int original_device) noexcept {
    if (devices.empty()) return;
    for (int device : devices) {
        if (cudaSetDevice(device) != cudaSuccess) continue;
        try { thunderfft::ThunderFFTFinalize<float>(); } catch (...) {}
        try { thunderfft::ThunderFFTFinalize<half>(); } catch (...) {}
    }
    cudaSetDevice(original_device);
}
} // namespace

int main(int argc, char** argv) {
    std::vector<int> devices;
    int original_device = 0;
    try {
        if (argc == 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
            std::cout << "Usage: " << argv[0] << "\n"
                      << "Validate twiddle lifecycle and device ownership on every supported visible GPU.\n"
                      << "No FFT, MMA or other GPU kernels are launched.\n";
            return 0;
        }
        if (argc != 1) throw std::runtime_error("This program accepts only --help");
        int count = 0;
        CACHE_CUDA(cudaGetDeviceCount(&count));
        CACHE_CUDA(cudaGetDevice(&original_device));
        for (int device = 0; device < count; ++device) {
            cudaDeviceProp properties{};
            CACHE_CUDA(cudaGetDeviceProperties(&properties, device));
            if (properties.major * 10 + properties.minor < THUNDERFFT_CACHE_MIN_CC) {
                std::cerr << "Skipping device " << device << " (" << properties.name
                          << "): this compiler build requires SM " << THUNDERFFT_CACHE_MIN_CC << "+\n";
                continue;
            }
            devices.push_back(device);
        }
        if (devices.empty()) throw std::runtime_error("No GPU supported by this cache-validation build");
        std::cout << "precision,device,stage,N,pointer,pass\n";
        check_type<float>(devices);
        check_type<half>(devices);
        cleanup(devices, original_device);
        std::cerr << "Cache validation: " << checks << " checks passed across " << devices.size()
                  << " visible GPUs; no GPU kernels launched\n";
        if (devices.size() < 2)
            std::cerr << "Cross-device isolation needs at least two supported visible GPUs.\n";
        return 0;
    } catch (const std::exception& error) {
        cleanup(devices, original_device);
        std::cerr << "Cache validation error: " << error.what() << '\n';
        return 2;
    }
}
