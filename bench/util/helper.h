#pragma once
#include "thunderfft/detail/utils.h"
#include "stat.h"
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

template <typename T, unsigned int N, typename Kernel>
static inline PerfStat benchmark_perf(Kernel &&kernel, vec2_t<T> *d_data,
                                      int batch) {
    // return {0,0,0};
    static constexpr unsigned int inside_repeats = 1000;
    static constexpr unsigned int kernel_runs = 20;
    static constexpr unsigned int warm_up_runs = 5;

    // R 회와 2R 회를 이용해 per-iteration kernel-only 시간 산출
    double t_R =
        measure_execution_ms(kernel, warm_up_runs, kernel_runs, inside_repeats);

    double t_2R = measure_execution_ms(kernel, warm_up_runs, kernel_runs,
                                       2 * inside_repeats);

    double comp_ms = (t_2R - t_R) / static_cast<double>(inside_repeats);

    double e2e_ms = measure_execution_ms(kernel, warm_up_runs, kernel_runs, 1);

    double comm_ms = measure_execution_ms(kernel, warm_up_runs, kernel_runs, 0);

    return {comp_ms, e2e_ms, comm_ms};
}

template <typename T, unsigned int N, typename Kernel>
static inline void benchmark_val(Kernel &&kernel, vec2_t<T> *d_data,
                                 int batch) {
    static constexpr unsigned int inside_repeats = 1;
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 0;

    measure_execution_ms(kernel, warm_up_runs, kernel_runs, inside_repeats);
}

template <typename T, unsigned int N, unsigned int radix, typename Kernel>
void benchmark_run(Kernel &&kernel, vec2_t<T> *h_data, float2 *baseline,
                   unsigned int B, std::string name = "") {
    using T2 = vec2_t<T>;

    // printf("running %s (type=%s, N=%d)\n", name.c_str(),bench_type_cstr<T>(),
    // N);

    T2 *d_custom = nullptr;
    CHECK_CUDA(cudaMalloc(&d_custom, sizeof(T2) * N * B));
    CHECK_CUDA(cudaMemcpy(d_custom, h_data, sizeof(T2) * N * B,
                          cudaMemcpyHostToDevice));

    auto kernel_wrapper = [&kernel, d_custom, B](unsigned int inside_repeats) {
        kernel(d_custom, inside_repeats);
    };

    benchmark_val<T, N>(kernel_wrapper, d_custom, B);

    T2 *h_custom = static_cast<T2 *>(std::malloc(sizeof(T2) * N * B));
    CHECK_CUDA(cudaMemcpy(h_custom, d_custom, sizeof(T2) * N * B,
                          cudaMemcpyDeviceToHost));

    const double max_err =
        static_cast<double>(check_max_abs_err(baseline, h_custom, N * B));
    const PerfStat perf = benchmark_perf<T, N>(kernel_wrapper, d_custom, B);

    std::string label = name + "(" + std::string(bench_type_cstr<T>()) + ")";
    stat::push(stat::RunStat{label, N, radix, B, max_err, perf.comp_ms,
                             perf.comm_ms, perf.e2e_ms});

    CHECK_CUDA(cudaFree(d_custom));
    std::free(h_custom);
}
