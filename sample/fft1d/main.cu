#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include <thunderfft/thunderfft.cuh>  // ThunderFFT<T,N>, ThunderFFTInitialize/Finalize
#include "../utils.h"

// ------------------------------
// Test input (bin-aligned tones)
// ------------------------------
static void make_test_input(float2* h, unsigned N, unsigned batch) {
    const float two_pi = 2.0f * float(M_PI);
    for (unsigned b = 0; b < batch; ++b) {
        const float f0  = float(1 + b);
        const float amp = 0.5f + 0.1f * float(b);
        for (unsigned i = 0; i < N; ++i) {
            const float t  = float(i) / float(N);
            const float re = amp * std::cos(two_pi * f0 * t)
                           + 0.1f * std::cos(two_pi * 7.0f * t);
            const float im = amp * std::sin(two_pi * f0 * t)
                           + 0.1f * std::sin(two_pi * 7.0f * t);
            h[b * N + i] = make_float2(re, im);
        }
    }
}

// ------------------------------
// Simple checksum (optional)
// ------------------------------
static double checksum(const float2* v, unsigned N, unsigned batch) {
    long double acc = 0.0L;
    const size_t total = size_t(N) * batch;
    for (size_t i = 0; i < total; ++i) {
        acc += (long double)v[i].x * 1.0001L
             + (long double)v[i].y * 0.9997L;
    }
    return (double)acc;
}

// ------------------------------
// Error metrics vs reference
// ------------------------------
static double l2_rel(const float2* a, const float2* b, size_t n) {
    long double num = 0.0L, den = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double dx = (long double)a[i].x - (long double)b[i].x;
        const long double dy = (long double)a[i].y - (long double)b[i].y;
        num += dx*dx + dy*dy;
        den += (long double)b[i].x*(long double)b[i].x
             + (long double)b[i].y*(long double)b[i].y;
    }
    return den > 0 ? (double)std::sqrt((double)(num/den)) : (double)std::sqrt((double)num);
}

static double linf_rel(const float2* a, const float2* b, size_t n) {
    long double max_diff = 0.0L, max_ref = 0.0L;
    for (size_t i = 0; i < n; ++i) {
        const long double dx = (long double)a[i].x - (long double)b[i].x;
        const long double dy = (long double)a[i].y - (long double)b[i].y;
        const long double diff = std::hypot(dx, dy);
        const long double refm = std::hypot((long double)b[i].x, (long double)b[i].y);
        if (diff > max_diff) max_diff = diff;
        if (refm > max_ref)  max_ref  = refm;
    }
    return max_ref > 0 ? (double)(max_diff / max_ref) : (double)max_diff;
}

int main(int /*argc*/, char** /*argv*/) {
    // Compile-time N for ThunderFFT template
    constexpr unsigned N = 64;

    unsigned batch = 65536;
    int device     = 0;

    CHECK_CUDA(cudaSetDevice(device));
    std::cout << "ThunderFFT vs cuFFT\n"
              << "  Type   : float/float2\n"
              << "  N      : " << N << " (compile-time)\n"
              << "  batch  : " << batch << "\n"
              << "  device : " << device << "\n";

    const size_t count = size_t(N) * batch;
    const size_t bytes = sizeof(float2) * count;

    // Host buffers
    float2* h_in  = static_cast<float2*>(std::malloc(bytes));
    float2* h_out = static_cast<float2*>(std::malloc(bytes)); // ThunderFFT
    float2* h_ref = static_cast<float2*>(std::malloc(bytes)); // cuFFT
    if (!h_in || !h_out || !h_ref) {
        std::fprintf(stderr, "Host allocation failed (%zu bytes)\n", bytes);
        return EXIT_FAILURE;
    }
    make_test_input(h_in, N, batch);

    // Device buffers
    float2* d_in  = nullptr;
    float2* d_out = nullptr; // ThunderFFT output
    float2* d_ref = nullptr; // cuFFT output
    CHECK_CUDA(cudaMalloc(&d_in,  bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_ref, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ThunderFFT twiddles (optional cache)
    thunderfft::ThunderFFTInitialize<float>(N);

    // --- ThunderFFT run (warm-up + timed) ---
    const unsigned int warm_up_runs = 1;
    const unsigned int runs = 5;
    const float ms_tf = measure_execution_ms(
        [&]() { thunderfft::ThunderFFT_global<float, N, true>(d_in, d_out, batch, /*stream=*/0); },
        warm_up_runs,
        runs);

    CHECK_CUDA(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- cuFFT reference (forward, unscaled) ---
    cufftHandle plan{};
    CHECK_CUFFT(cufftPlan1d(&plan, int(N), CUFFT_C2C, int(batch)));
    CHECK_CUFFT(cufftExecC2C(plan,
                             reinterpret_cast<cufftComplex*>(d_in),
                             reinterpret_cast<cufftComplex*>(d_ref),
                             CUFFT_FORWARD));
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_ref, d_ref, bytes, cudaMemcpyDeviceToHost));

    // --- Compare ThunderFFT vs cuFFT ---
    const double err_l2   = l2_rel(h_out, h_ref, count);
    const double err_linf = linf_rel(h_out, h_ref, count);

    // Print
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "ThunderFFT avg time : " << ms_tf << " ms\n";
    std::cout << "Checksum(ThunderFFT): " << checksum(h_out, N, batch) << "\n";
    std::cout << "Compare vs cuFFT    : L2_rel=" << std::setprecision(3) << err_l2
              << "  Linf_rel=" << err_linf << std::setprecision(6) << "\n";

    // Show first few bins for batch 0
    const unsigned to_print = std::min<unsigned>(8, N);
    std::cout << "First " << to_print << " bins (batch 0):\n";
    for (unsigned i = 0; i < to_print; ++i) {
        const float2 a = h_out[i];
        const float2 b = h_ref[i];
        std::cout << "  k=" << std::setw(3) << i
                  << "  TF=(" << a.x << ", " << a.y << ")"
                  << "  cuFFT=(" << b.x << ", " << b.y << ")\n";
    }

    // Cleanup
    thunderfft::ThunderFFTFinalize<float>();
    CHECK_CUFFT(cufftDestroy(plan));
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFree(d_ref));
    std::free(h_in);
    std::free(h_out);
    std::free(h_ref);

    std::cout << "Done.\n";
    return 0;
}