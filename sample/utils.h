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