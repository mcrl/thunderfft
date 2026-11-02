#include <math.h>
#include <stdio.h>

#include "stat.h"
#include "helper.h"

#include "thunderfft_bench.h"
// #include "fft_tc_sm_bench.h"

void baseline_fft(float2 *h_input, float2 *h_output, int N, int batch);

template <long long N> int test() {
    constexpr long long batch = 65536;

    float2 *h_input = (float2 *)malloc(sizeof(float2) * N * batch);
    half2 *h_input_half = (half2 *)malloc(sizeof(half2) * N * batch);
    float2 *h_output = (float2 *)malloc(sizeof(float2) * N * batch);

    for (int i = 0; i < N * batch; ++i) {
        h_input[i].x = sinf(2 * M_PI * (i % N)/N)/sqrt(N);
        // h_input[i].x = i % N;
        h_input[i].y = 0.0f;

        h_input_half[i] = make_half2(h_input[i].x, h_input[i].y);
    }

    baseline_fft(h_input, h_output, N, batch);

    // fft_tc_sm_benchmark<N>(h_input, h_input_half, h_output, batch);
    
    thunderfft_benchmark<half, N>(h_input_half, h_output, batch);
    thunderfft_benchmark<float, N>(h_input, h_output, batch);
    // for(int i=0; i<4096; i++) {
    //     printf("%f %f\n", h_output[i].x, h_output[i].y);
    // }

    free(h_input);
    free(h_input_half);
    free(h_output);
    return 0;
}

int main() {
    test<64>();
    // test<256>();
    // test<1024>();
    // test<4096>();

    stat::set_title("FFT benchmark results");
    stat::print_table();
    return 0;
}