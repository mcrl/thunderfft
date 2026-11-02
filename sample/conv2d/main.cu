#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdio.h>

#include "../utils.h"

// Forward declarations implemented in thunder.cu / cufftdx.cu
template <int f>
void cufftdx_convolution(const float2* h_input, const float2* h_filter,
                         float2* h_output, int N);

template <int f>
void my_convolution(const float2* h_input, const float2* h_filter,
                    float2* h_output, int N, int warp_count);

// Kernel for element-wise complex multiplication (used by cuFFT reference)
__global__ void complex_multiply_kernel(cufftComplex* a, const cufftComplex* b, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float real = a[idx].x * b[idx].x - a[idx].y * b[idx].y;
        const float imag = a[idx].x * b[idx].y + a[idx].y * b[idx].x;
        a[idx].x = real;
        a[idx].y = imag;
    }
}

// Kernel for scaling (used by cuFFT reference)
__global__ void scale_kernel(cufftComplex* data, float scale, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx].x *= scale;
        data[idx].y *= scale;
    }
}

// ------------------------------
// Reference convolution using cuFFT
// ------------------------------
static void reference_convolution_cufft(const float2* h_input, const float2* h_filter,
                                        float2* h_output, int N, int f) {
    const int out_size = N - f + 1;
    const int fft_size = N;

    cufftComplex* d_input = nullptr;
    cufftComplex* d_filter = nullptr;
    cufftComplex* d_output = nullptr;
    cufftComplex* d_input_fft = nullptr;
    cufftComplex* d_filter_fft = nullptr;
    cufftComplex* d_result_fft = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_filter, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_input_fft, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_filter_fft, sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMalloc(&d_result_fft, sizeof(cufftComplex) * fft_size * fft_size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, sizeof(cufftComplex) * N * N,
                         cudaMemcpyHostToDevice));

    cufftComplex* h_filter_padded = static_cast<cufftComplex*>(
        std::calloc(fft_size * fft_size, sizeof(cufftComplex)));
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            const int idx_i = (fft_size - i) % fft_size;
            const int idx_j = (fft_size - j) % fft_size;
            h_filter_padded[idx_i * fft_size + idx_j].x = h_filter[i * f + j].x;
            h_filter_padded[idx_i * fft_size + idx_j].y = h_filter[i * f + j].y;
        }
    }
    CHECK_CUDA(cudaMemcpy(d_filter, h_filter_padded,
                          sizeof(cufftComplex) * fft_size * fft_size,
                          cudaMemcpyHostToDevice));
    std::free(h_filter_padded);

    cufftHandle plan_forward{};
    cufftHandle plan_inverse{};
    CHECK_CUFFT(cufftPlan2d(&plan_forward, fft_size, fft_size, CUFFT_C2C));
    CHECK_CUFFT(cufftPlan2d(&plan_inverse, fft_size, fft_size, CUFFT_C2C));

    CHECK_CUFFT(cufftExecC2C(plan_forward, d_input, d_input_fft, CUFFT_FORWARD));
    CHECK_CUFFT(cufftExecC2C(plan_forward, d_filter, d_filter_fft, CUFFT_FORWARD));

    const int num_threads = 256;
    const int num_blocks = (fft_size * fft_size + num_threads - 1) / num_threads;

    CHECK_CUDA(cudaMemcpy(d_result_fft, d_input_fft,
                          sizeof(cufftComplex) * fft_size * fft_size,
                          cudaMemcpyDeviceToDevice));
    complex_multiply_kernel<<<num_blocks, num_threads>>>(d_result_fft, d_filter_fft,
                                                         fft_size * fft_size);
    CHECK_CUDA(cudaGetLastError());

    CHECK_CUFFT(cufftExecC2C(plan_inverse, d_result_fft, d_output, CUFFT_INVERSE));

    const float scale = 1.0f / (fft_size * fft_size);
    scale_kernel<<<num_blocks, num_threads>>>(d_output, scale, fft_size * fft_size);
    CHECK_CUDA(cudaGetLastError());

    cufftComplex* h_result = static_cast<cufftComplex*>(
        std::malloc(sizeof(cufftComplex) * fft_size * fft_size));
    CHECK_CUDA(cudaMemcpy(h_result, d_output,
                          sizeof(cufftComplex) * fft_size * fft_size,
                          cudaMemcpyDeviceToHost));

    for (int i = 0; i < out_size; ++i) {
        for (int j = 0; j < out_size; ++j) {
            h_output[i * out_size + j].x = h_result[i * fft_size + j].x;
            h_output[i * out_size + j].y = h_result[i * fft_size + j].y;
        }
    }

    std::free(h_result);
    CHECK_CUFFT(cufftDestroy(plan_forward));
    CHECK_CUFFT(cufftDestroy(plan_inverse));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_filter));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_input_fft));
    CHECK_CUDA(cudaFree(d_filter_fft));
    CHECK_CUDA(cudaFree(d_result_fft));
}

// ------------------------------
// Helpers to build test cases
// ------------------------------
static void make_test_input_random(float2* h_input, int N) {
    srand(time(NULL));
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            h_input[i * N + j].x = rand() / (float)RAND_MAX;
            h_input[i * N + j].y = rand() / (float)RAND_MAX;
            // h_input[i * N + j].x = i+j;
            // h_input[i * N + j].y = 0;
        }
    }
}

static void make_test_filter_random(float2* h_filter, int f) {
    srand(time(NULL));
    for (int i = 0; i < f; ++i) {
        for (int j = 0; j < f; ++j) {
            h_filter[i * f + j].x = rand() / (float)RAND_MAX;
            h_filter[i * f + j].y = rand() / (float)RAND_MAX;
            // h_filter[i * f + j].x = i*3+j;
            // h_filter[i * f + j].y = 0;
        }
    }
}

static double l2_rel(const float2* a, const float2* b, size_t n) {
    long double num = 0.0L;
    long double den = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double diff_real = static_cast<long double>(a[i].x) - b[i].x;
        const long double diff_imag = static_cast<long double>(a[i].y) - b[i].y;
        const long double diff_mag = diff_real * diff_real + diff_imag * diff_imag;
        num += diff_mag;
        const long double ref_mag = static_cast<long double>(b[i].x) * b[i].x +
                                    static_cast<long double>(b[i].y) * b[i].y;
        den += ref_mag;
    }
    return den > 0.0L ? static_cast<double>(std::sqrt(num / den))
                      : static_cast<double>(std::sqrt(num));
}

static double linf_rel(const float2* a, const float2* b, size_t n) {
    long double max_diff = 0.0L;
    long double max_ref = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double diff_real = static_cast<long double>(a[i].x) - b[i].x;
        const long double diff_imag = static_cast<long double>(a[i].y) - b[i].y;
        const long double diff_mag = std::sqrt(diff_real * diff_real + diff_imag * diff_imag);
        const long double ref_mag = std::sqrt(static_cast<long double>(b[i].x) * b[i].x +
                                              static_cast<long double>(b[i].y) * b[i].y);
        if (diff_mag > max_diff) max_diff = diff_mag;
        if (ref_mag > max_ref) max_ref = ref_mag;
    }
    return max_ref > 0.0L ? static_cast<double>(max_diff / max_ref)
                          : static_cast<double>(max_diff);
}

int main(int argc, char** argv) {
    int N = 1024;
    int device = 0;
    int warp_count = 8;
    constexpr int F = 3;

    if (argc >= 2) {
        N = std::atoi(argv[1]);
    }
    if (argc >= 3) {
        const int requested_f = std::atoi(argv[2]);
        if (requested_f != F) {
            std::cerr << "This sample currently supports filter size f = " << F
                      << ". Requested f = " << requested_f << "\n";
            return EXIT_FAILURE;
        }
    }
    if (argc >= 4) {
        device = std::atoi(argv[3]);
    }
    if (argc >= 5) {
        warp_count = std::atoi(argv[4]);
    }

    if (N < F) {
        std::cerr << "Input size N must be >= " << F << "\n";
        return EXIT_FAILURE;
    }

    if (N != 1024) {
        std::cout << "[ThunderFFT] Forcing N=1024 for this path.\n";
        N = 1024;
    }

    const int out_size = N - F + 1;

    std::cout << "Convolution Benchmark: ThunderFFT vs cuFFTDx\n"
              << "  Input size : " << N << " x " << N << "\n"
              << "  Filter size: " << F << " x " << F << "\n"
              << "  Output size: " << out_size << " x " << out_size << "\n"
              << "  Device     : " << device << "\n"
              << "  Warp count : " << warp_count << "\n";

    CHECK_CUDA(cudaSetDevice(device));

    const size_t input_bytes = sizeof(float2) * static_cast<size_t>(N) * N;
    const size_t filter_bytes = sizeof(float2) * static_cast<size_t>(F) * F;
    const size_t output_bytes = sizeof(float2) * static_cast<size_t>(out_size) * out_size;

    float2* h_input = static_cast<float2*>(std::malloc(input_bytes));
    float2* h_filter = static_cast<float2*>(std::malloc(filter_bytes));
    float2* h_output_thunder = static_cast<float2*>(std::malloc(output_bytes));
    float2* h_output_cufftdx = static_cast<float2*>(std::malloc(output_bytes));
    float2* h_output_ref = static_cast<float2*>(std::malloc(output_bytes));

    if (!h_input || !h_filter || !h_output_thunder || !h_output_cufftdx || !h_output_ref) {
        std::fprintf(stderr, "Host allocation failed\n");
        std::free(h_input);
        std::free(h_filter);
        std::free(h_output_thunder);
        std::free(h_output_cufftdx);
        std::free(h_output_ref);
        return EXIT_FAILURE;
    }

    make_test_input_random(h_input, N);
    make_test_filter_random(h_filter, F);

    std::cout << "\n--- Running ThunderFFT convolution ---\n";
    my_convolution<F>(h_input, h_filter, h_output_thunder, N, warp_count);

    std::cout << "\n--- Running cuFFTDx convolution ---\n";
    // cufftdx_convolution<F>(h_input, h_filter, h_output_cufftdx, N);

    std::cout << "\n--- Running cuFFT reference convolution (validation) ---\n";
    reference_convolution_cufft(h_input, h_filter, h_output_ref, N, F);
    std::cout << "[cuFFT] Validation run complete." << std::endl;

    const size_t output_count = static_cast<size_t>(out_size) * out_size;
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n--- Validation vs cuFFT reference ---\n";
    const double thunder_l2 = l2_rel(h_output_thunder, h_output_ref, output_count);
    const double thunder_linf = linf_rel(h_output_thunder, h_output_ref, output_count);
    // const double cufftdx_l2 = l2_rel(h_output_cufftdx, h_output_ref, output_count);
    // const double cufftdx_linf = linf_rel(h_output_cufftdx, h_output_ref, output_count);

    std::cout << "L2 rel. error  (ThunderFFT vs cuFFT): " << thunder_l2 << "\n";
    std::cout << "Linf rel. error (ThunderFFT vs cuFFT): " << thunder_linf << "\n";
    // std::cout << "L2 rel. error  (cuFFTDx vs cuFFT)  : " << cufftdx_l2 << "\n";
    // std::cout << "Linf rel. error (cuFFTDx vs cuFFT) : " << cufftdx_linf << "\n";

    const int to_print = std::min(8, out_size);  // limit for readability
    std::cout << "\nFirst " << to_print << " x " << to_print << " samples:\n";
    std::cout << "  i  j     Thunder(real)  Thunder(imag)  cuFFTDx(real)  cuFFTDx(imag)"
                 "  cuFFT(real)   cuFFT(imag)\n";
    for (int i = 0; i < to_print; ++i) {
        for (int j = 0; j < to_print; ++j) {
            const int idx = i * out_size + j;
            std::cout << std::setw(3) << i << " " << std::setw(3) << j
                      << " " << std::setw(13) << h_output_thunder[idx].x
                      << " " << std::setw(13) << h_output_thunder[idx].y
                      << " " << std::setw(13) << h_output_cufftdx[idx].x
                      << " " << std::setw(13) << h_output_cufftdx[idx].y
                      << " " << std::setw(13) << h_output_ref[idx].x
                      << " " << std::setw(13) << h_output_ref[idx].y << "\n";
        }
    }

    std::free(h_input);
    std::free(h_filter);
    std::free(h_output_thunder);
    std::free(h_output_cufftdx);
    std::free(h_output_ref);

    std::cout << "\nDone." << std::endl;
    return 0;
}
