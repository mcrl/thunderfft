#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>

#include "helper.h"
#include "stat.h"

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n", #call, __FILE__, \
                    __LINE__, cudaGetErrorString(err));                        \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

#define CHECK_CUFFT(call)                                                      \
    do {                                                                       \
        cufftResult err = call;                                                \
        if (err != CUFFT_SUCCESS) {                                            \
            fprintf(stderr, "cuFFT error in %s (%s:%d): %d\n", #call,          \
                    __FILE__, __LINE__, err);                                  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

void baseline_fft(float2 *h_input, float2 *h_output, int N, int batch) {
    static constexpr unsigned int kernel_runs = 1;
    static constexpr unsigned int warm_up_runs = 1;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    float2 *d_input, *d_output;
    cudaMalloc(&d_input, sizeof(float2) * N * batch);
    cudaMalloc(&d_output, sizeof(float2) * N * batch);
    cudaMemcpy(d_input, h_input, sizeof(float2) * N * batch,
               cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftPlan1d(&plan, N, CUFFT_C2C, batch);

    auto kernel = [&](unsigned int inside_repeats) {
        (void)inside_repeats;
        cufftExecC2C(plan, (cufftComplex *)d_input, (cufftComplex *)d_output,
                     CUFFT_FORWARD);
        // assert("4096 half is not supported" && false);
    };

    double elapsedTime =
        measure_execution_ms(kernel, warm_up_runs, kernel_runs, 0);

    cudaMemcpy(h_output, d_output, sizeof(float2) * N * batch,
               cudaMemcpyDeviceToHost);

    stat::push(
        stat::RunStat{/*type*/ "cufft", // 자유롭게 "cufft" 등으로 바꿔도 됨
                      /*N*/ static_cast<unsigned>(N),
                      /*radix*/ 0, // baseline이라 없음
                      /*B*/ static_cast<unsigned>(batch),
                      /*max_err*/ 0.0,
                      /*compute_ms*/ 0.0,
                      /*e2e_ms*/ elapsedTime});

    cudaFree(d_input);
    cudaFree(d_output);
    cufftDestroy(plan);
}
