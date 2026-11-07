#pragma once

#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <cstdlib>

#ifndef CHECK_CUDA
#define CHECK_CUDA(stmt)                                                       \
    do {                                                                       \
        cudaError_t err__ = (stmt);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #stmt,        \
                         __FILE__, __LINE__, cudaGetErrorString(err__));       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(expr)                                                       \
    do {                                                                       \
        cufftResult err = (expr);                                              \
        if (err != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "cuFFT Error at %s:%d: %d\n", __FILE__,       \
                         __LINE__, err);                                       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)
#endif

template <typename Kernel>
float measure_execution_ms(Kernel &&kernel, const unsigned int warm_up_runs,
                            const unsigned int runs) {
    cudaEvent_t startEvent, stopEvent;
    CHECK_CUDA(cudaEventCreate(&startEvent));
    CHECK_CUDA(cudaEventCreate(&stopEvent));
    CHECK_CUDA(cudaDeviceSynchronize());

    for (size_t i = 0; i < warm_up_runs; i++) {
        kernel();
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(startEvent));
    for (size_t i = 0; i < runs; i++) {
        kernel();
    }
    CHECK_CUDA(cudaEventRecord(stopEvent));
    CHECK_CUDA(cudaDeviceSynchronize());

    float time;
    CHECK_CUDA(cudaEventElapsedTime(&time, startEvent, stopEvent));
    CHECK_CUDA(cudaEventDestroy(startEvent));
    CHECK_CUDA(cudaEventDestroy(stopEvent));
    return time / runs;
}
// Timed launches per measurement: enough for about 0.2 s of work, at least 3 and at
// most max_runs, so slow cases (e.g. cuDNN at large 2D filters) do not dominate
// the run time. One untimed call estimates the per-launch time.
template <typename Kernel>
unsigned int adaptive_runs(Kernel &&kernel, const unsigned int max_runs) {
    const float once = measure_execution_ms(kernel, 0, 1);
    const unsigned int wanted = once > 0.0f ? static_cast<unsigned int>(200.0f / once) + 1 : max_runs;
    return wanted < 3 ? 3 : (wanted > max_runs ? max_runs : wanted);
}
