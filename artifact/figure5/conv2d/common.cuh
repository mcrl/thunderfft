// Shared definitions for the Figure 5 2D convolution driver: single-channel FP32 valid
// cross-correlation out[i][j] = sum_{a,b < f} in[i + a][j + b] * filter[a][b], 0 <= i, j < N - f + 1.
#pragma once

#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace figure5 {

inline void check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(expression) + ": " + cudaGetErrorString(error) + " (" + file + ":" +
                                 std::to_string(line) + ")");
}
inline void check_cufft(cufftResult error, const char* expression, const char* file, int line) {
    if (error != CUFFT_SUCCESS)
        throw std::runtime_error(std::string(expression) + ": cuFFT error " + std::to_string(int(error)) + " (" + file +
                                 ":" + std::to_string(line) + ")");
}
#define FIG5_CUDA(expression) ::figure5::check_cuda((expression), #expression, __FILE__, __LINE__)
#define FIG5_CUFFT(expression) ::figure5::check_cufft((expression), #expression, __FILE__, __LINE__)

template <class T> struct DeviceBuffer {
    T* data = nullptr;
    size_t count = 0;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t n) : count(n) { FIG5_CUDA(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Problem {
    int N = 0;
    int f = 0;
    int M = 0;
    const float* d_input = nullptr;
    const float* d_filter = nullptr;
    cudaStream_t stream = nullptr;
};

// run() enqueues one timed iteration on problem.stream, including any filter transform.
class Method {
public:
    virtual ~Method() = default;
    virtual void run() = 0;
    virtual const float* output() const = 0;
    virtual size_t output_pitch() const = 0;
    virtual bool filter_fft_included() const = 0;
    virtual std::string details() const { return ""; }
};

std::unique_ptr<Method> make_cudnn(const Problem&);
std::unique_ptr<Method> make_cufft(const Problem&);
std::unique_ptr<Method> make_cufftdx_c2c(const Problem&);
std::unique_ptr<Method> make_thunderfft_c2c(const Problem&);

}  // namespace figure5
