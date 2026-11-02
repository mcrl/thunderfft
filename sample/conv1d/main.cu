#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#include <thunderfft/thunderfft.cuh>

#include "../utils.h"

using complex_t = half2;

namespace {

constexpr int kFFTSize = 64;
constexpr int kBPB = thunderfft::batch_per_block<kFFTSize>;
constexpr int kThreads = thunderfft::threads_per_warp;
constexpr int kEPT = (kFFTSize * kBPB) / kThreads;
#ifndef THFF_CONV1D_TILES_PER_WARP
#define THFF_CONV1D_TILES_PER_WARP 1
#endif
constexpr int kTilesPerWarp = THFF_CONV1D_TILES_PER_WARP;
static_assert(kTilesPerWarp >= 1, "THFF_CONV1D_TILES_PER_WARP must be >= 1");

__global__ void f32_to_half2_kernel(const float2* __restrict__ in,
                                    half2* __restrict__ out,
                                    int n) {
    const int idx = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (idx < n) out[idx] = __floats2half2_rn(in[idx].x, in[idx].y);
}

template <int N = kFFTSize, int TilesPerWarp = kTilesPerWarp>
__global__ void conv1d_direct_fft_kernel(const complex_t* __restrict__ input,
                                         const complex_t* __restrict__ filter_fft,
                                         complex_t* __restrict__ output,
                                         int input_len,
                                         int output_len,
                                         int num_tiles,
                                         int valid_tile,
                                         int tile_base) {
    constexpr int BPB = kBPB;
    constexpr int ept = kEPT;

    extern __shared__ __align__(16) unsigned char _smem[];
    complex_t* s_in = reinterpret_cast<complex_t*>(_smem);
    complex_t reg[ept];

    using L_in = thunderfft::layout_t<N, BPB, 1, N, 64, 4, true>;
    using L_out = thunderfft::layout_t<N, BPB, 1, N, 16, 1, false>;

    const int tidx = threadIdx.x;
    const int laneid = threadIdx.x & 31;
    const int logical_block_base = tile_base + int(blockIdx.x) * BPB * TilesPerWarp;

    vec2_t<half> W_fwd[28];
    vec2_t<half> W_inv[28];
    thunderfft::unit_fp16::make_reg_b_precompute<64, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<64, false>(W_inv);

    complex_t filter_reg[ept / 2];
    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        const int fidx = i + (ept / 2) * (laneid & 3);
        filter_reg[i] = filter_fft[fidx];
    }

    #pragma unroll
    for (int it = 0; it < TilesPerWarp; ++it) {
        const int block_tile_base = logical_block_base + it * BPB;
        if (block_tile_base >= num_tiles) break;

        for (int i = 0; i < BPB; ++i) {
            const int tile = block_tile_base + i;
            for (int _j = 0; _j < N; _j += kThreads) {
                const int j = _j + tidx;
                if (j < N) {
                    int smem_idx = i * N + j;
                    smem_idx += (smem_idx / 64) * 4;
                    complex_t v = __float2half2_rn(0.0f);
                    if (tile < num_tiles) {
                        const int in_idx = tile * valid_tile + reverse_bit_groups<2, 6>(j);
                        if (in_idx < input_len) v = input[in_idx];
                    }
                    s_in[smem_idx] = v;
                }
            }
        }
        __syncthreads();

        thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
        __syncthreads();
        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, true>(reg, W_fwd, s_in);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ept / 2; ++i) {
            reg[i] = cmul(reg[i], filter_reg[i]);
            reg[i + ept / 2] = cmul(reg[i + ept / 2], filter_reg[i]);
        }

        swap_thread_data(reg);
        thunderfft::ThunderFFT_kernel_reg<half, N, BPB, false>(reg, W_inv, s_in);
        __syncthreads();
        thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
        __syncthreads();

        for (int i = 0; i < BPB; ++i) {
            const int tile = block_tile_base + i;
            for (int _j = 0; _j < valid_tile; _j += kThreads) {
                const int j = _j + tidx;
                if (j < valid_tile && tile < num_tiles) {
                    const int out_idx = tile * valid_tile + j;
                    if (out_idx < output_len) {
                        int smem_idx = i * N + j;
                        smem_idx += smem_idx / 16;
                        output[out_idx] = s_in[smem_idx];
                    }
                }
            }
        }
        __syncthreads();
    }
}

static void make_random_complex(std::vector<float2>& x) {
    for (auto& v : x) {
        v.x = rand() / float(RAND_MAX);
        v.y = rand() / float(RAND_MAX);
    }
}

static void to_half2(const std::vector<float2>& in, std::vector<half2>& out) {
    out.resize(in.size());
    for (size_t i = 0; i < in.size(); ++i) {
        out[i] = __floats2half2_rn(in[i].x, in[i].y);
    }
}

static std::vector<float2> make_padded_filter_fft64(const std::vector<float2>& filter) {
    std::vector<float2> filt_pad(kFFTSize, make_float2(0.0f, 0.0f));
    const int f = static_cast<int>(filter.size());
    for (int i = 0; i < f; ++i) {
        const int idx = (kFFTSize - i) % kFFTSize;
        filt_pad[idx].x = filter[i].x / float(kFFTSize);
        filt_pad[idx].y = filter[i].y / float(kFFTSize);
    }
    return filt_pad;
}

static void reference_conv1d_valid(const std::vector<float2>& input,
                                   const std::vector<float2>& filter,
                                   std::vector<float2>& output) {
    const int n = static_cast<int>(input.size());
    const int f = static_cast<int>(filter.size());
    const int out = n - f + 1;
    output.assign(out, make_float2(0.0f, 0.0f));
    for (int i = 0; i < out; ++i) {
        float re = 0.0f;
        float im = 0.0f;
        for (int k = 0; k < f; ++k) {
            const float2 a = input[i + k];
            const float2 b = filter[k];
            re += a.x * b.x - a.y * b.y;
            im += a.x * b.y + a.y * b.x;
        }
        output[i] = make_float2(re, im);
    }
}

static double l2_rel(const std::vector<float2>& a, const std::vector<float2>& b) {
    long double num = 0.0L;
    long double den = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        const long double dx = static_cast<long double>(a[i].x) - b[i].x;
        const long double dy = static_cast<long double>(a[i].y) - b[i].y;
        num += dx * dx + dy * dy;
        den += static_cast<long double>(b[i].x) * b[i].x +
               static_cast<long double>(b[i].y) * b[i].y;
    }
    return den > 0.0L ? static_cast<double>(std::sqrt(num / den))
                      : static_cast<double>(std::sqrt(num));
}

static double linf_rel(const std::vector<float2>& a, const std::vector<float2>& b) {
    long double max_diff = 0.0L;
    long double max_ref = 0.0L;
    for (size_t i = 0; i < a.size(); ++i) {
        const long double dx = static_cast<long double>(a[i].x) - b[i].x;
        const long double dy = static_cast<long double>(a[i].y) - b[i].y;
        const long double diff = std::sqrt(dx * dx + dy * dy);
        const long double ref = std::sqrt(static_cast<long double>(b[i].x) * b[i].x +
                                          static_cast<long double>(b[i].y) * b[i].y);
        if (diff > max_diff) max_diff = diff;
        if (ref > max_ref) max_ref = ref;
    }
    return max_ref > 0.0L ? static_cast<double>(max_diff / max_ref)
                          : static_cast<double>(max_diff);
}

}  // namespace

int main(int argc, char** argv) {
    int input_len = 1 << 16;
    int filter_len = 9;
    int device = 0;
    int validate = 1;
    int warmup = 5;
    int runs = 50;

    if (argc >= 2) input_len = std::atoi(argv[1]);
    if (argc >= 3) filter_len = std::atoi(argv[2]);
    if (argc >= 4) device = std::atoi(argv[3]);
    if (argc >= 5) validate = std::atoi(argv[4]);
    if (argc >= 6) warmup = std::atoi(argv[5]);
    if (argc >= 7) runs = std::atoi(argv[6]);

    if (filter_len < 1 || filter_len > kFFTSize) {
        std::cerr << "filter_len must be in [1, " << kFFTSize << "]\n";
        return EXIT_FAILURE;
    }
    if (input_len < filter_len) {
        std::cerr << "input_len must be >= filter_len\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    srand(42);

    const int valid_tile = kFFTSize - filter_len + 1;
    const int output_len = input_len - filter_len + 1;
    const int num_tiles = (output_len + valid_tile - 1) / valid_tile;
    const int total_blocks = (num_tiles + kBPB - 1) / kBPB;
    const int launch_blocks = (total_blocks + kTilesPerWarp - 1) / kTilesPerWarp;

    std::cout << "ThunderFFT tiled 1D conv (FFT size fixed to 64)\n"
              << "  input_len  : " << input_len << "\n"
              << "  filter_len : " << filter_len << "\n"
              << "  output_len : " << output_len << "\n"
              << "  valid_tile : " << valid_tile << "\n"
              << "  num_tiles  : " << num_tiles << "\n"
              << "  total_blocks: " << total_blocks << "\n"
              << "  launch_blocks: " << launch_blocks << "\n"
              << "  tiles_per_warp(static): " << kTilesPerWarp << "\n"
              << "  warmup/runs: " << warmup << "/" << runs << "\n";

    std::vector<float2> h_input_f32(input_len);
    std::vector<complex_t> h_input;
    std::vector<float2> h_filter(filter_len);
    std::vector<complex_t> h_output(output_len);
    std::vector<float2> h_ref;

    make_random_complex(h_input_f32);
    to_half2(h_input_f32, h_input);
    make_random_complex(h_filter);

    complex_t* d_input = nullptr;
    complex_t* d_output = nullptr;
    complex_t* d_filter_fft = nullptr;
    float2* d_filter_time = nullptr;
    float2* d_filter_fft_f32 = nullptr;

    CHECK_CUDA(cudaMalloc(&d_input, sizeof(complex_t) * input_len));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(complex_t) * output_len));
    CHECK_CUDA(cudaMalloc(&d_filter_fft, sizeof(complex_t) * kFFTSize));
    CHECK_CUDA(cudaMalloc(&d_filter_time, sizeof(float2) * kFFTSize));
    CHECK_CUDA(cudaMalloc(&d_filter_fft_f32, sizeof(float2) * kFFTSize));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), sizeof(complex_t) * input_len, cudaMemcpyHostToDevice));

    auto h_filter_padded = make_padded_filter_fft64(h_filter);
    CHECK_CUDA(cudaMemcpy(d_filter_time, h_filter_padded.data(),
                          sizeof(float2) * kFFTSize, cudaMemcpyHostToDevice));

    cufftHandle filter_plan;
    CHECK_CUFFT(cufftPlan1d(&filter_plan, kFFTSize, CUFFT_C2C, 1));
    CHECK_CUFFT(cufftExecC2C(filter_plan,
                             reinterpret_cast<cufftComplex*>(d_filter_time),
                             reinterpret_cast<cufftComplex*>(d_filter_fft_f32),
                             CUFFT_FORWARD));
    f32_to_half2_kernel<<<1, 64>>>(d_filter_fft_f32, d_filter_fft, kFFTSize);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    thunderfft::ThunderFFTInitialize<half>(kFFTSize);

    const size_t fft_shmem_bytes = sizeof(complex_t) * (kFFTSize + pad_h(kFFTSize)) * kBPB;

    auto run_once = [&]() {
        const dim3 grid(launch_blocks);
        conv1d_direct_fft_kernel<<<grid, 32, fft_shmem_bytes>>>(
            d_input, d_filter_fft, d_output, input_len, output_len, num_tiles, valid_tile, 0);
        CHECK_CUDA(cudaGetLastError());
    };

    const float ms = measure_execution_ms(run_once, static_cast<unsigned>(warmup),
                                          static_cast<unsigned>(runs));
    std::cout << std::fixed << std::setprecision(6)
              << "Kernel pipeline time: " << ms << " ms\n";

    CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, sizeof(complex_t) * output_len, cudaMemcpyDeviceToHost));

    if (validate != 0) {
        std::vector<float2> h_output_f32(output_len);
        for (int i = 0; i < output_len; ++i) {
            h_output_f32[i] = __half22float2(h_output[i]);
        }
        h_ref.resize(output_len);
        reference_conv1d_valid(h_input_f32, h_filter, h_ref);
        const double err_l2 = l2_rel(h_output_f32, h_ref);
        const double err_linf = linf_rel(h_output_f32, h_ref);
        std::cout << std::setprecision(4)
                  << "Validation vs direct conv: L2_rel=" << err_l2
                  << " Linf_rel=" << err_linf << "\n";

        const int print_n = std::min(8, output_len);
        for (int i = 0; i < print_n; ++i) {
            std::cout << "  y[" << i << "] th=(" << h_output_f32[i].x << ", " << h_output_f32[i].y
                      << ") ref=(" << h_ref[i].x << ", " << h_ref[i].y << ")\n";
        }
    }

    thunderfft::ThunderFFTFinalize<half>();
    CHECK_CUFFT(cufftDestroy(filter_plan));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaFree(d_filter_fft));
    CHECK_CUDA(cudaFree(d_filter_fft_f32));
    CHECK_CUDA(cudaFree(d_filter_time));
    return 0;
}
