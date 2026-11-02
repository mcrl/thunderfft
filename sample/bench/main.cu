#include <math.h>
#include <stdio.h>

#include "thunderfft_bench.h"

#include "stat.h"

// #include "fft_tc_sm_bench.h"

void baseline_fft(float2 *h_input, float2 *h_output, int N, int batch);

#ifndef THUNDERFFT_PROFILE_BATCH
#define THUNDERFFT_PROFILE_BATCH 65536
#endif

#ifndef THUNDERFFT_PROFILE_N
#define THUNDERFFT_PROFILE_N 1024
#endif

#ifndef THUNDERFFT_PROFILE_FLOAT
#define THUNDERFFT_PROFILE_FLOAT 1
#endif

#ifndef THUNDERFFT_PROFILE_HALF
#define THUNDERFFT_PROFILE_HALF 0
#endif

template <long long N> int test() {
    constexpr long long batch = THUNDERFFT_PROFILE_BATCH;

    float2 *h_input = (float2 *)malloc(sizeof(float2) * N * batch);
    half2 *h_input_half = (half2 *)malloc(sizeof(half2) * N * batch);
    float2 *h_output = (float2 *)malloc(sizeof(float2) * N * batch);

    for (int i = 0; i < N * batch; ++i) {
        h_input[i].x = sinf(2 * M_PI * (i % N)/N)/sqrt(N);
        // h_input[i].x = (i % 2048);
        h_input[i].y = 0.0f;

        h_input_half[i] = make_half2(h_input[i].x, h_input[i].y);
    }

    baseline_fft(h_input, h_output, N, batch);
    
    #if THUNDERFFT_PROFILE_FLOAT
    thunderfft::ThunderFFTInitialize<float>(N);
    thunderfft::thunderfft_benchmark_reg<float, N>(h_input, h_output, batch);
    thunderfft::thunderfft_benchmark_smem<float, N>(h_input, h_output, batch);
    thunderfft::ThunderFFTFinalize<float>();
    #endif

    #if THUNDERFFT_PROFILE_HALF
    thunderfft::ThunderFFTInitialize<half>(N);
    thunderfft::thunderfft_benchmark_reg<half, N>(h_input_half, h_output, batch);
    thunderfft::thunderfft_benchmark_smem<half, N>(h_input_half, h_output, batch);
    thunderfft::ThunderFFTFinalize<half>();
    #endif

    // for(int i=0; i<2048; i++) {
    //     printf("%f %f, ", h_output[i].x, h_output[i].y);
    //     // printf("%f %f\n", h_input[i].x, h_input[i].y);
    //     if((i+1)%32==0) printf("\n");
    // }

    free(h_input);
    free(h_input_half);
    free(h_output);
    return 0;
}

int main() {
    // test<64>();
    test<128>();
    // test<256>();
    // test<512>();
    // test<1024>();
    // test<2048>();
    // test<4096>();

    stat::set_title("FFT benchmark results");
    stat::print_table();
    return 0;
}
