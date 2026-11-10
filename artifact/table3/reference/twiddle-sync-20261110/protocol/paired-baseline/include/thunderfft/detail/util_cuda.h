#pragma once
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>


#define PI 3.14159265358979323846

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

static inline void tf_check_cuda(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string("[ThunderFFT] ") + msg + ": " + cudaGetErrorString(e));
    }
}

template <typename T> constexpr const char *type_cstr() {
    using U = std::remove_cv_t<T>;
    if constexpr (std::is_same_v<U, float>)
        return "float";
    else if constexpr (std::is_same_v<U, double>)
        return "double";
    else if constexpr (std::is_same_v<U, int>)
        return "int";
    else if constexpr (std::is_same_v<U, float2>)
        return "float2";
    else if constexpr (std::is_same_v<U, float2>)
        return "float2";
    else if constexpr (std::is_same_v<U, half>)
        return "half";
    else if constexpr (std::is_same_v<U, half2>)
        return "half2";
    else
        return "unknown";
}

template <class T>
using vec2_t =
    typename std::conditional_t<std::is_same_v<std::remove_cv_t<T>, float>,
                                float2, half2>;
