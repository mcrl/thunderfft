#pragma once

#include <cuda_fp16.h>
#include <iostream>

#define PI 3.14159265358979323846f

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


struct PerfStat {
    double comp_ms;
    double e2e_ms;
    double comm_ms;
};

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

template <unsigned int N>
inline constexpr unsigned LOG2P_builtin = [] {
    static_assert(N && (N & (N - 1)) == 0, "N must be power of two");
    return __builtin_ctz(N); // pow2에서 log2(N)과 동일
}();

template <class T>
using vec2_t =
    typename std::conditional_t<std::is_same_v<std::remove_cv_t<T>, float>,
                                float2, half2>;

template <int r, int N>
__device__ __forceinline__ int reverse_bit_groups(int x) {
    int num_groups = N / r;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (r * i)) & ((1 << r) - 1);
        result |= group << (r * (num_groups - 1 - i));
    }
    for (int i = 0; i < N % r; i++) {
        int bit = (x >> (r * num_groups + i)) & 1;
        result = (result << 1) | bit;
    }
    return result;
}

__device__ __forceinline__ float2 W(int index, int N) {
    return make_float2(__cosf(-2 * PI * index / N),
                       __sinf(-2 * PI * index / N));
    // return make_float2(cos(-2 * PI * index / N),
    //                    sin(-2 * PI * index / N));
}

// template <typename T>
// __device__ __forceinline__ vec2_t<T> cmul(vec2_t<T> a, float2 w);

// // float 버전
// template <> __device__ __forceinline__ float2 cmul<float>(float2 a, float2 w) {
//     return make_float2(a.x * w.x - a.y * w.y, a.y * w.x + a.x * w.y);
// }

__device__ __forceinline__ float2 cmul(float2 a, float2 w) {
    return make_float2(a.x * w.x - a.y * w.y,  // real
                       a.y * w.x + a.x * w.y); // imag
}

// float2용 연산자
__device__ __forceinline__ float2 operator+(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}
__device__ __forceinline__ float2 operator-(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

// half 버전
__device__ __forceinline__ half2 cmul(half2 a, float2 w) {
    // half2 -> float2
    float ax = __half2float(__low2half(a));
    float ay = __half2float(__high2half(a));

    float rx = ax * w.x - ay * w.y;
    float ry = ay * w.x + ax * w.y;

    // float2 -> half2 (반올림)
    return __floats2half2_rn(rx, ry);
}

template <typename T> inline float2 to_float2(const T &v);

template <> inline float2 to_float2<float2>(const float2 &v) { return v; }

template <> inline float2 to_float2<half2>(const half2 &v) {
    return __half22float2(v);
}

template <typename T>
float check_max_abs_err(const float2 *ref, const T *test, int N) {
    float max_abs_err = 0.0f;

    for (int i = 0; i < N; ++i) {
        float2 tf = to_float2<T>(test[i]);
        float dx = ref[i].x - tf.x;
        float dy = ref[i].y - tf.y;
        float abs_err = sqrtf(dx * dx + dy * dy);
        if (abs_err > max_abs_err)
            max_abs_err = abs_err;
    }

    // 요구사항: "maximum absolute error만 출력"
    return max_abs_err;
}

template <typename T> __device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}

constexpr int pad_h(int N) {
    switch (N) {
        case 64: return 4;
        case 256: return 16;
        case 1024: return 64;
        case 4096: return 256;
        default:  return -1; // power of two 아닌 경우
    }
}

__host__ __device__ constexpr int pad(int N) {
    switch (N) {
        case 64: return 4;
        case 256: return 16;
        case 1024: return 64;
        case 4096: return 256;
        default:  return -1;  // not a power of two
    }
}

