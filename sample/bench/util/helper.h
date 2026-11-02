#pragma once
#include "thunderfft/detail/util_cuda.h"
#include "stat.h"
#include <cmath>
#include <iostream>

#define CUDA_CHECK_AND_EXIT(error)                                             \
    {                                                                          \
        auto status = static_cast<cudaError_t>(error);                         \
        if (status != cudaSuccess) {                                           \
            std::cout << cudaGetErrorString(status) << " " << __FILE__ << ":"  \
                      << __LINE__ << std::endl;                                \
            std::exit(status);                                                 \
        }                                                                      \
    }

#ifndef THUNDERFFT_INSIDE_REPEATS
#define THUNDERFFT_INSIDE_REPEATS 1000
#endif

template <typename Kernel>
float measure_execution_ms(Kernel &&kernel, const unsigned int warm_up_runs,
                           const unsigned int runs,
                           const unsigned int inside_repeats) {
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    for (size_t i = 0; i < warm_up_runs; i++) {
        kernel(inside_repeats);
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent));
    for (size_t i = 0; i < runs; i++) {
        kernel(inside_repeats);
    }
    CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    float time;
    CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
    return time / runs;
}

template <typename Kernel>
float measure_execution_ms_once(Kernel &&kernel,
                                const unsigned int warm_up_runs,
                                const unsigned int runs) {
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventCreate(&stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    for (size_t i = 0; i < warm_up_runs; i++) {
        kernel();
    }
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    CUDA_CHECK_AND_EXIT(cudaEventRecord(startEvent));
    for (size_t i = 0; i < runs; i++) {
        kernel();
    }
    CUDA_CHECK_AND_EXIT(cudaEventRecord(stopEvent));
    CUDA_CHECK_AND_EXIT(cudaDeviceSynchronize());

    float time;
    CUDA_CHECK_AND_EXIT(cudaEventElapsedTime(&time, startEvent, stopEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(startEvent));
    CUDA_CHECK_AND_EXIT(cudaEventDestroy(stopEvent));
    return time / runs;
}
struct PerfStat {
    double compute_ms;
    double e2e_ms;
};

template <typename T>
__host__ __device__ constexpr const char *bench_type_cstr() {
    if constexpr (std::is_same_v<T, half>)
        return "fp16";
    if constexpr (std::is_same_v<T, float>)
        return "fp32";
    if constexpr (std::is_same_v<T, double>)
        return "fp64";
    else
        return "unknown";
}


template <typename T>
float check_max_abs_err(const float2 *ref, const T *test, int N) {
    float max_abs_err = 0.0f;

    for (int i = 0; i < N; ++i) {
        float2 tf;
        if constexpr (std::is_same_v<T, float2>) {
            tf = test[i];
        } else if constexpr (std::is_same_v<T, half2>) {
            tf = make_float2(__half2float(test[i].x),
                             __half2float(test[i].y));
        } else {
            static_assert(sizeof(T) == 0,
                          "check_max_abs_err: unsupported type T");
        }
        float dx = ref[i].x - tf.x;
        float dy = ref[i].y - tf.y;
        float abs_err = sqrtf(dx * dx + dy * dy);
        if (!std::isfinite(tf.x) || !std::isfinite(tf.y) || !std::isfinite(abs_err)) {
            return INFINITY;
        }
        if (abs_err > max_abs_err)
            max_abs_err = abs_err;
    }

    return max_abs_err;
}

template <typename T, unsigned int N, typename Kernel, typename KernelE2E>
static inline PerfStat benchmark_perf(Kernel &&kernel, KernelE2E &&kernel_e2e,
                                      int batch) {
    // return {0,0,0};
    static constexpr unsigned int inside_repeats = THUNDERFFT_INSIDE_REPEATS;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 1;

    double t_R =
        measure_execution_ms(kernel, warm_up_runs, kernel_runs, inside_repeats);

    double t_2R = measure_execution_ms(kernel, warm_up_runs, kernel_runs,
                                       2 * inside_repeats);

    double compute_ms = (t_2R - t_R) / static_cast<double>(inside_repeats);

    static constexpr unsigned int e2e_warm_up_runs = 1;
    static constexpr unsigned int e2e_runs = 10;
    double e2e_ms =
        measure_execution_ms_once(kernel_e2e, e2e_warm_up_runs, e2e_runs);

    return {compute_ms, e2e_ms};
}

template <typename T, unsigned int N, typename Kernel>
static inline void benchmark_val(Kernel &&kernel) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 0;

    measure_execution_ms(kernel, warm_up_runs, kernel_runs, inside_repeats);
}

template <typename T, unsigned int N, unsigned int radix, typename Kernel, typename KernelE2E>
void benchmark_run(Kernel &&kernel, KernelE2E &&kernel_e2e,
                   vec2_t<T> *h_data, float2 *baseline,
                   unsigned int B, std::string name = "") {
    using T2 = vec2_t<T>;

    const size_t bytes = sizeof(T2) * N * B;

    // Validation and performance must not share the same mutated device buffer.
    // Repeated in-place FFTs can accidentally make max_err look better than it is.
    T2 *d_validate = nullptr;
    T2 *d_perf = nullptr;
    CHECK_CUDA(cudaMalloc(&d_validate, bytes));
    CHECK_CUDA(cudaMalloc(&d_perf, bytes));
    CHECK_CUDA(cudaMemcpy(d_validate, h_data, bytes,
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_perf, h_data, bytes,
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaDeviceSynchronize());

    auto validation_kernel = [&kernel, d_validate](unsigned int inside_repeats) {
        kernel(d_validate, inside_repeats);
    };
    auto perf_kernel = [&kernel, d_perf](unsigned int inside_repeats) {
        kernel(d_perf, inside_repeats);
    };
    auto perf_kernel_e2e = [&kernel_e2e, d_perf]() { kernel_e2e(d_perf); };

    benchmark_val<T, N>(validation_kernel);

    T2 *h_custom = static_cast<T2 *>(std::malloc(sizeof(T2) * N * B));
    CHECK_CUDA(cudaMemcpy(h_custom, d_validate, bytes,
                          cudaMemcpyDeviceToHost));

    const double max_err =
        static_cast<double>(check_max_abs_err(baseline, h_custom, N * B));
    const PerfStat perf =
        benchmark_perf<T, N>(perf_kernel, perf_kernel_e2e, B);

    std::string label = name + "(" + std::string(bench_type_cstr<T>()) + ")";
    stat::push(stat::RunStat{label, N, radix, B, max_err, perf.compute_ms,
                             perf.e2e_ms});

    CHECK_CUDA(cudaFree(d_validate));
    CHECK_CUDA(cudaFree(d_perf));
    std::free(h_custom);
}
