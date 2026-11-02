#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include <cstdlib>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include <thunderfft/thunderfft.cuh>
#include <thunderfft/thunderfft_layout_tuned.h>

#include "../utils.h"

using cpx_h = half2;

namespace {

#ifndef THFF_LMID256_PAD_PERIOD
#define THFF_LMID256_PAD_PERIOD 64
#endif
#ifndef THFF_LMID256_PAD
#define THFF_LMID256_PAD 1
#endif
#ifndef THFF_LMID1024_PAD_PERIOD
#define THFF_LMID1024_PAD_PERIOD 16
#endif
#ifndef THFF_LMID1024_PAD
#define THFF_LMID1024_PAD 1
#endif
#ifndef THFF_SKIP_MID_ROUNDTRIP
#define THFF_SKIP_MID_ROUNDTRIP 0
#endif

template <int N, int BPB>
struct conv_lmid_layout {
    using type = typename thunderfft::bench_layout<half, N, BPB>::L_in;
};

template <int BPB>
struct conv_lmid_layout<256, BPB> {
    using type = thunderfft::layout_t<256, BPB, 1, 256, THFF_LMID256_PAD_PERIOD, THFF_LMID256_PAD, false>;
};

template <int BPB>
struct conv_lmid_layout<1024, BPB> {
    using type =
        thunderfft::layout_t<1024, BPB, 1, 1024, THFF_LMID1024_PAD_PERIOD, THFF_LMID1024_PAD, false>;
};

__device__ __forceinline__ half2 cmul_h2(half2 a, half2 b) {
    return __hcmadd(a, b, __float2half2_rn(0.f));
}

template <typename sL>
__device__ __forceinline__ void ThunderFFT_gmem2smem_realio(cpx_h* __restrict__ smem,
                                                            const half* __restrict__ gmem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride = sL::batch_stride;
    constexpr int pad_period = sL::pad_period;
    constexpr int pad = sL::pad;
    static_assert(!sL::reversed, "realio loader expects non-reversed layout");

    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    const half z = __float2half(0.0f);
    for (int i = 0; i < batch; ++i) {
        for (int _j = 0; _j < N; _j += thunderfft::threads_per_warp * thunderfft::warp_per_block<N>) {
            const int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx / pad_period * pad;
            smem[smem_idx] = __halves2half2(gmem[i * N + j], z);
        }
    }
}

template <typename sL>
__device__ __forceinline__ void ThunderFFT_smem2gmem_realio(half* __restrict__ gmem,
                                                            const cpx_h* __restrict__ smem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride = sL::batch_stride;
    constexpr int pad_period = sL::pad_period;
    constexpr int pad = sL::pad;
    static_assert(!sL::reversed, "realio storer expects non-reversed layout");

    const int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = 0; i < batch; ++i) {
        for (int _j = 0; _j < N; _j += thunderfft::threads_per_warp * thunderfft::warp_per_block<N>) {
            const int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx / pad_period * pad;
            gmem[i * N + j] = __low2half(smem[smem_idx]);
        }
    }
}

template <int N, bool forward>
__global__ void thunder_fftN_block_kernel(const cpx_h* __restrict__ d_input,
                                          cpx_h* __restrict__ d_output) {
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int ept = N * BPB / (thunderfft::threads_per_warp * WPB);

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, N, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, N, BPB>::L_out;
    using L_mid = typename conv_lmid_layout<N, BPB>::type;
    using L_mid = typename conv_lmid_layout<N, BPB>::type;
    using L_mid = typename conv_lmid_layout<N, BPB>::type;

    cpx_h W[36];
    thunderfft::unit_fp16::make_reg_b_precompute<N, forward>(W);

    thunderfft::ThunderFFT_gmem2smem<half, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, N, BPB, forward>(reg, W, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    thunderfft::ThunderFFT_smem2gmem<half, L_out>(d_output + blockIdx.x * BPB * N, s_in);
}

__device__ __forceinline__ void swap_h2(cpx_h& a, cpx_h& b) {
    cpx_h t = a;
    a = b;
    b = t;
}

__device__ __forceinline__ void reverse_data_4096(cpx_h* data) {
    swap_h2(data[1], data[4]);
    swap_h2(data[2], data[8]);
    swap_h2(data[3], data[12]);
    swap_h2(data[6], data[9]);
    swap_h2(data[7], data[13]);
    swap_h2(data[11], data[14]);

    swap_h2(data[17], data[20]);
    swap_h2(data[18], data[24]);
    swap_h2(data[19], data[28]);
    swap_h2(data[22], data[25]);
    swap_h2(data[23], data[29]);
    swap_h2(data[27], data[30]);
}

template <int N>
__global__ void permute_filter_layout4096(const cpx_h* __restrict__ d_H_in,
                                           cpx_h* __restrict__ d_H_perm,
                                           int D) {
    static_assert(N == 4096, "permute_filter_layout4096 is only for N=4096");
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;
    static_assert(BPB == 1, "permute_filter_layout4096 assumes BPB == 1");

    const int d = blockIdx.x;
    if (d >= D) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread_linear = ty * thunderfft::threads_per_warp + tx;
    const int h_base = d * N;
    const int out_base = d * N;
    const int col_base = (tx >> 2) + (ty << 4);
    const int row_base = (tx & 3) << 4;

    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        // Match the forward FFT register's natural frequency: row * 64 + col.
        const int idx0 = h_base + (row_base + i) * 64 + col_base;
        const int idx1 = h_base + (row_base + i) * 64 + col_base + 8;
        d_H_perm[out_base + i * block_threads + thread_linear] = d_H_in[idx0];
        d_H_perm[out_base + (i + 16) * block_threads + thread_linear] = d_H_in[idx1];
    }
}

template <int N>
__global__ void permute_filter_layout1024(const cpx_h* __restrict__ d_H_in,
                                          cpx_h* __restrict__ d_H_perm,
                                          int D) {
    static_assert(N == 1024, "permute_filter_layout1024 is only for N=1024");
    const int d = blockIdx.x;
    if (d >= D) return;
    const int laneid = threadIdx.x;
    const int out_base = d * N;
    const int in_base = d * N;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        const int row = (i % 4) + (laneid % 4) * 4;
        const int col0 = (laneid / 4) % 2 + (laneid / 8) * 8 + (i / 4) * 2;
        const int col1 = col0 + 32;
        const int n0 = row * 64 + col0;
        const int n1 = row * 64 + col1;
        d_H_perm[out_base + i * 32 + laneid] = d_H_in[in_base + n0];
        d_H_perm[out_base + (i + 16) * 32 + laneid] = d_H_in[in_base + n1];
    }
}

template <int N>
__global__ void permute_filter_layout256(const cpx_h* __restrict__ d_H_in,
                                         cpx_h* __restrict__ d_H_perm,
                                         int D) {
    static_assert(N == 256, "permute_filter_layout256 is only for N=256");
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int block_threads = thunderfft::threads_per_warp * WPB; // 128
    const int d = blockIdx.x;
    if (d >= D) return;
    const int laneid = threadIdx.x;
    const int warpid = threadIdx.y;
    const int thread_linear = warpid * thunderfft::threads_per_warp + laneid;
    const int out_base = d * (16 * block_threads); // d * 2048
    const int in_base = d * N;

    #pragma unroll
    for (int r = 0; r < 16; ++r) {
        const int n = (laneid % 4) + warpid * 4 + (r / 4) * 16 + (r % 4) * 64;
        d_H_perm[out_base + r * block_threads + thread_linear] = d_H_in[in_base + n];
    }
}

template <int N>
__device__ __forceinline__ void pointwise_mul_reg(cpx_h* __restrict__ reg,
                                                   const cpx_h* __restrict__ d_H,
                                                   const cpx_h* __restrict__ d_H_perm,
                                                   int D) {
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int block_threads = thunderfft::threads_per_warp * WPB;

    if constexpr (N == 4096) {
        static_assert(BPB == 1, "N=4096 expects BPB=1");
        const int d = blockIdx.x % D;
        const int h_base = d * N;
        const int thread_linear = threadIdx.y * thunderfft::threads_per_warp + threadIdx.x;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            const int idx = h_base + i * block_threads + thread_linear;
            reg[i] = cmul_h2(reg[i], d_H_perm[idx]);
        }
    } else if constexpr (N == 1024) {
        static_assert(BPB == 1, "N=1024 expects BPB=1");
        const int d = blockIdx.x % D;
        const int laneid = threadIdx.x;
        const int h_base = d * N;
        #pragma unroll
        for (int i = 0; i < 32; ++i) {
            reg[i] = cmul_h2(reg[i], d_H_perm[h_base + i * 32 + laneid]);
        }
    } else if constexpr (N == 256) {
        static_assert(BPB == 16, "N=256 expects BPB=16");
        const int laneid = threadIdx.x;
        const int warpid = threadIdx.y;
        const int thread_linear = warpid * thunderfft::threads_per_warp + laneid;
        constexpr int block_threads = thunderfft::threads_per_warp * WPB; // 128
        const int bd_base = blockIdx.x * BPB;
        const int batch0 = (laneid / 4);
        const int batch1 = batch0 + 8;
        const int d0 = (bd_base + batch0) % D;
        const int d1 = (bd_base + batch1) % D;
        const int d0_base = d0 * (16 * block_threads);
        const int d1_base = d1 * (16 * block_threads);

        #pragma unroll
        for (int r = 0; r < 16; ++r) {
            const int off = r * block_threads + thread_linear;
            reg[r] = cmul_h2(reg[r], d_H_perm[d0_base + off]);
            reg[r + 16] = cmul_h2(reg[r + 16], d_H_perm[d1_base + off]);
        }
    }
}

template <int N>
__global__ void thunder_convN_block_kernel_realin_realout(const half* __restrict__ d_input_real,
                                                           const cpx_h* __restrict__ d_H,
                                                           const cpx_h* __restrict__ d_H_perm,
                                                           half* __restrict__ d_output_real,
                                                           int D) {
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;
    constexpr int ept = N * BPB / (thunderfft::threads_per_warp * WPB);

    extern __shared__ __align__(16) unsigned char _smem[];
    cpx_h* s_in = reinterpret_cast<cpx_h*>(_smem);
    cpx_h reg[ept];

    using L_in = typename thunderfft::bench_layout<half, N, BPB>::L_in;
    using L_out = typename thunderfft::bench_layout<half, N, BPB>::L_out;
    using L_mid = typename conv_lmid_layout<N, BPB>::type;

    cpx_h W_fwd[36];
    cpx_h W_inv[36];
    thunderfft::unit_fp16::make_reg_b_precompute<N, true>(W_fwd);
    thunderfft::unit_fp16::make_reg_b_precompute<N, false>(W_inv);

    ThunderFFT_gmem2smem_realio<L_in>(s_in, d_input_real + blockIdx.x * BPB * N);
    __syncthreads();
    thunderfft::ThunderFFT_smem2reg<half, L_in>(reg, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_kernel_reg<half, N, BPB, true>(reg, W_fwd, s_in);
    __syncthreads();

    pointwise_mul_reg<N>(reg, d_H, d_H_perm, D);

    if constexpr (N == 4096) {
        reverse_data_4096(reg);
    } else {
#if THFF_SKIP_MID_ROUNDTRIP
        // Experimental: skip reorder roundtrip entirely for perf-only what-if.
#else
        // Mid-roundtrip layout can be tuned independently from benchmark L_in/L_out.
        thunderfft::ThunderFFT_reg2smem<half, L_mid>(s_in, reg);
        __syncthreads();
        thunderfft::ThunderFFT_smem2reg<half, L_mid>(reg, s_in);
        __syncthreads();
#endif
    }

    thunderfft::ThunderFFT_kernel_reg<half, N, BPB, false>(reg, W_inv, s_in);
    __syncthreads();
    thunderfft::ThunderFFT_reg2smem<half, L_out>(s_in, reg);
    __syncthreads();
    ThunderFFT_smem2gmem_realio<L_out>(d_output_real + blockIdx.x * BPB * N, s_in);
}

__global__ void f32_to_h2(const float2* __restrict__ in,
                          half2* __restrict__ out,
                          int n) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n) out[i] = __floats2half2_rn(in[i].x, in[i].y);
}

__global__ void real_to_complex_kernel(const half* __restrict__ in,
                                       cpx_h* __restrict__ out,
                                       int n) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n) out[i] = __halves2half2(in[i], __float2half(0.0f));
}

__global__ void complex_to_real_kernel(const cpx_h* __restrict__ in,
                                       half* __restrict__ out,
                                       int n) {
    int i = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
    if (i < n) out[i] = __low2half(in[i]);
}

template <int N>
float run_case(int K,
               int B,
               int D,
               int warmup,
               int runs,
               int include_filter_fft,
               int validate,
               int real_input,
               int real_filter) {
    constexpr int BPB = thunderfft::batch_per_block<N>;
    constexpr int WPB = thunderfft::warp_per_block<N>;

    if ((B * D) % BPB != 0) {
        std::cerr << "B*D must be divisible by BPB=" << BPB << " for N=" << N << "\n";
        return -1.0f;
    }
    if (!real_input) {
        std::cerr << "This benchmark path currently supports real_input=1 only.\n";
        return -1.0f;
    }

    const int batch_count = B * D;
    const int sig_elems = batch_count * N;
    const int D_fft = ((D + BPB - 1) / BPB) * BPB;
    const int filt_elems = D * N;
    const int filt_elems_fft = D_fft * N;

    std::vector<cpx_h> h_x(sig_elems);
    std::vector<half> h_x_real(sig_elems, __float2half(0.0f));
    std::vector<float2> h_h_time(filt_elems, make_float2(0.f, 0.f));
    std::vector<cpx_h> h_h_half_padded;

    for (int i = 0; i < sig_elems; ++i) {
        float re = std::rand() / float(RAND_MAX);
        float im = 0.0f;
        h_x[i] = __floats2half2_rn(re, im);
        h_x_real[i] = __float2half(re);
    }
    for (int d = 0; d < D; ++d) {
        for (int k = 0; k < K; ++k) {
            float re = (std::rand() / float(RAND_MAX)) / float(N);
            float im = real_filter ? 0.0f : ((std::rand() / float(RAND_MAX)) / float(N));
            h_h_time[d * N + k] = make_float2(re, im);
        }
    }
    if constexpr (N != 4096) {
        h_h_half_padded.assign(filt_elems_fft, __float2half2_rn(0.0f));
        for (int d = 0; d < D; ++d) {
            for (int n = 0; n < N; ++n) {
                const float2 v = h_h_time[d * N + n];
                h_h_half_padded[d * N + n] = __floats2half2_rn(v.x, v.y);
            }
        }
    }

    cpx_h *d_H = nullptr, *d_H_perm = nullptr, *d_h_time_half = nullptr;
    half *d_x_real = nullptr, *d_out_real = nullptr;
    float2 *d_h_time = nullptr, *d_h_freq = nullptr;

    CHECK_CUDA(cudaMalloc(&d_H, sizeof(cpx_h) * filt_elems_fft));
    if constexpr (N == 4096) {
        CHECK_CUDA(cudaMalloc(&d_H_perm, sizeof(cpx_h) * filt_elems));
    } else if constexpr (N == 1024) {
        CHECK_CUDA(cudaMalloc(&d_H_perm, sizeof(cpx_h) * filt_elems));
    } else {
        CHECK_CUDA(cudaMalloc(&d_H_perm, sizeof(cpx_h) * D * 16 * (thunderfft::threads_per_warp * WPB)));
    }
    CHECK_CUDA(cudaMalloc(&d_x_real, sizeof(half) * sig_elems));
    CHECK_CUDA(cudaMalloc(&d_out_real, sizeof(half) * sig_elems));
    CHECK_CUDA(cudaMalloc(&d_h_time, sizeof(float2) * filt_elems));
    CHECK_CUDA(cudaMalloc(&d_h_freq, sizeof(float2) * filt_elems));
    if constexpr (N != 4096) {
        CHECK_CUDA(cudaMalloc(&d_h_time_half, sizeof(cpx_h) * filt_elems_fft));
    }

    CHECK_CUDA(cudaMemcpy(d_x_real, h_x_real.data(), sizeof(half) * sig_elems, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_h_time, h_h_time.data(), sizeof(float2) * filt_elems, cudaMemcpyHostToDevice));
    if constexpr (N != 4096) {
        CHECK_CUDA(cudaMemcpy(d_h_time_half, h_h_half_padded.data(),
                              sizeof(cpx_h) * filt_elems_fft, cudaMemcpyHostToDevice));
    }

    cufftHandle plan_h;
    CHECK_CUFFT(cufftPlan1d(&plan_h, N, CUFFT_C2C, D));
    thunderfft::ThunderFFTInitialize<half>(N);

    const dim3 grid_signal(batch_count / BPB);
    const dim3 block(thunderfft::threads_per_warp, WPB);
    const size_t shmem_bytes = sizeof(cpx_h) * (N + pad_h(N)) * BPB;

    CHECK_CUDA(cudaFuncSetAttribute(
        thunder_convN_block_kernel_realin_realout<N>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)));

    auto filter_fft_once = [&]() {
        if constexpr (N == 4096) {
            CHECK_CUFFT(cufftExecC2C(plan_h,
                                     reinterpret_cast<cufftComplex*>(d_h_time),
                                     reinterpret_cast<cufftComplex*>(d_h_freq),
                                     CUFFT_FORWARD));
            f32_to_h2<<<(filt_elems + 255) / 256, 256>>>(d_h_freq, d_H, filt_elems);
            CHECK_CUDA(cudaGetLastError());
            permute_filter_layout4096<N><<<D, block>>>(d_H, d_H_perm, D);
            CHECK_CUDA(cudaGetLastError());
        } else if constexpr (N == 1024) {
            const dim3 grid_filter(D_fft / BPB);
            thunder_fftN_block_kernel<N, true><<<grid_filter, block, shmem_bytes>>>(d_h_time_half, d_H);
            CHECK_CUDA(cudaGetLastError());
            permute_filter_layout1024<N><<<D, block>>>(d_H, d_H_perm, D);
            CHECK_CUDA(cudaGetLastError());
        } else {
            const dim3 grid_filter(D_fft / BPB);
            thunder_fftN_block_kernel<N, true><<<grid_filter, block, shmem_bytes>>>(d_h_time_half, d_H);
            CHECK_CUDA(cudaGetLastError());
            permute_filter_layout256<N><<<D, block>>>(d_H, d_H_perm, D);
            CHECK_CUDA(cudaGetLastError());
        }
    };

    if (!include_filter_fft) {
        filter_fft_once();
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    auto run_once = [&]() {
        if (include_filter_fft) filter_fft_once();
        thunder_convN_block_kernel_realin_realout<N><<<grid_signal, block, shmem_bytes>>>(
            d_x_real, d_H, d_H_perm, d_out_real, D);
        CHECK_CUDA(cudaGetLastError());
    };

    float ms = measure_execution_ms(run_once, unsigned(warmup), unsigned(runs));

    bool validation_passed = true;
    if (validate) {
        std::vector<half> h_out_real(sig_elems);
        CHECK_CUDA(cudaMemcpy(h_out_real.data(), d_out_real, sizeof(half) * sig_elems, cudaMemcpyDeviceToHost));

        double max_abs_raw = 0.0;
        double max_abs_scaled = 0.0;
        double num_raw = 0.0, num_scaled = 0.0, den = 0.0;
        double max_abs_scaled_corr = 0.0;
        double num_scaled_corr = 0.0, den_corr = 0.0;
        for (int bd = 0; bd < batch_count; ++bd) {
            const int d = bd % D;
            for (int n = 0; n < N; ++n) {
                float ref_re = 0.0f, ref_im = 0.0f;
                float refc_re = 0.0f, refc_im = 0.0f;
                for (int kk = 0; kk < K; ++kk) {
                    const int idx = n - kk;
                    const int xi = (idx >= 0) ? idx : (idx + N);
                    float2 xv = __half22float2(h_x[bd * N + xi]);
                    float2 hv = h_h_time[d * N + kk];
                    ref_re += xv.x * hv.x - xv.y * hv.y;
                    ref_im += xv.x * hv.y + xv.y * hv.x;

                    const int idxc = n + kk;
                    const int xic = (idxc < N) ? idxc : (idxc - N);
                    float2 xvc = __half22float2(h_x[bd * N + xic]);
                    refc_re += xvc.x * hv.x - xvc.y * hv.y;
                    refc_im += xvc.x * hv.y + xvc.y * hv.x;
                }
                const float ovx = __half2float(h_out_real[bd * N + n]);
                const float ovy = 0.0f;
                const float dx_raw = ovx - ref_re;
                const float dy_raw = ovy - ref_im;
                const float dx_scaled = ovx / float(N) - ref_re;
                const float dy_scaled = ovy / float(N) - ref_im;
                const double abs_raw = std::sqrt(double(dx_raw) * dx_raw + double(dy_raw) * dy_raw);
                const double abs_scaled = std::sqrt(double(dx_scaled) * dx_scaled + double(dy_scaled) * dy_scaled);
                max_abs_raw = std::max(max_abs_raw, abs_raw);
                max_abs_scaled = std::max(max_abs_scaled, abs_scaled);
                num_raw += double(dx_raw) * dx_raw + double(dy_raw) * dy_raw;
                num_scaled += double(dx_scaled) * dx_scaled + double(dy_scaled) * dy_scaled;
                den += double(ref_re) * ref_re + double(ref_im) * ref_im;

                const float dx_corr = ovx / float(N) - refc_re;
                const float dy_corr = ovy / float(N) - refc_im;
                const double abs_corr = std::sqrt(double(dx_corr) * dx_corr + double(dy_corr) * dy_corr);
                max_abs_scaled_corr = std::max(max_abs_scaled_corr, abs_corr);
                num_scaled_corr += double(dx_corr) * dx_corr + double(dy_corr) * dy_corr;
                den_corr += double(refc_re) * refc_re + double(refc_im) * refc_im;
            }
        }
        const double l2_raw = den > 0.0 ? std::sqrt(num_raw / den) : std::sqrt(num_raw);
        const double l2_scaled = den > 0.0 ? std::sqrt(num_scaled / den) : std::sqrt(num_scaled);
        const double l2_scaled_corr =
            den_corr > 0.0 ? std::sqrt(num_scaled_corr / den_corr) : std::sqrt(num_scaled_corr);
        constexpr double relative_l2_limit = 5e-3;
        validation_passed = std::isfinite(l2_scaled) && l2_scaled <= relative_l2_limit;
        std::cout << std::setprecision(6)
                  << "Validation N=" << N << " vs direct circular conv (raw): L2_rel=" << l2_raw
                  << " max_abs=" << max_abs_raw << "\n"
                  << "Validation N=" << N << " vs direct circular conv (raw/N): L2_rel=" << l2_scaled
                  << " max_abs=" << max_abs_scaled << "\n"
                  << "Validation N=" << N << " vs direct circular corr (raw/N): L2_rel=" << l2_scaled_corr
                  << " max_abs=" << max_abs_scaled_corr << "\n"
                  << "Circular convolution validation: " << (validation_passed ? "PASS" : "FAIL")
                  << " (raw/N relative L2 limit=" << relative_l2_limit << ")\n";
    }

    thunderfft::ThunderFFTFinalize<half>();
    CHECK_CUFFT(cufftDestroy(plan_h));
    CHECK_CUDA(cudaFree(d_H));
    CHECK_CUDA(cudaFree(d_H_perm));
    CHECK_CUDA(cudaFree(d_x_real));
    CHECK_CUDA(cudaFree(d_out_real));
    CHECK_CUDA(cudaFree(d_h_time));
    CHECK_CUDA(cudaFree(d_h_freq));
    if constexpr (N != 4096) {
        CHECK_CUDA(cudaFree(d_h_time_half));
    }

    return validation_passed ? ms : -1.0f;
}

}  // namespace

int main(int argc, char** argv) {
    int N = 4096;
    int K = 9;
    int device = 0;
    int B = 1;
    int D = 1;
    int warmup = 5;
    int runs = 30;
    int include_filter_fft = 1;
    int validate = 1;
    int real_input = 1;
    int real_filter = 1;

    if (argc >= 2) N = std::atoi(argv[1]);
    if (argc >= 3) K = std::atoi(argv[2]);
    if (argc >= 4) device = std::atoi(argv[3]);
    if (argc >= 5) B = std::atoi(argv[4]);
    if (argc >= 6) D = std::atoi(argv[5]);
    if (argc >= 7) warmup = std::atoi(argv[6]);
    if (argc >= 8) runs = std::atoi(argv[7]);
    if (argc >= 9) include_filter_fft = std::atoi(argv[8]);
    if (argc >= 10) validate = std::atoi(argv[9]);
    if (argc >= 11) real_input = std::atoi(argv[10]);
    if (argc >= 12) real_filter = std::atoi(argv[11]);

    if (!(N == 256 || N == 1024 || N == 4096)) {
        std::cerr << "N must be one of {256, 1024, 4096}\n";
        return EXIT_FAILURE;
    }
    if (K < 1 || K > N) {
        std::cerr << "K must be in [1, " << N << "]\n";
        return EXIT_FAILURE;
    }
    if (argc > 12 || device < 0 || B <= 0 || D <= 0 || warmup < 0 || runs <= 0 ||
        (include_filter_fft != 0 && include_filter_fft != 1) ||
        (validate != 0 && validate != 1) || real_input != 1 || real_filter != 1) {
        std::cerr << "Require a valid device, positive B/D/runs, nonnegative warmup, "
                     "boolean flags, and real_input=real_filter=1.\n";
        return EXIT_FAILURE;
    }

    CHECK_CUDA(cudaSetDevice(device));
    std::srand(42);

    float ms = -1.0f;
    if (N == 256) {
        ms = run_case<256>(K, B, D, warmup, runs, include_filter_fft, validate, real_input, real_filter);
    } else if (N == 1024) {
        ms = run_case<1024>(K, B, D, warmup, runs, include_filter_fft, validate, real_input, real_filter);
    } else {
        ms = run_case<4096>(K, B, D, warmup, runs, include_filter_fft, validate, real_input, real_filter);
    }
    if (ms < 0.0f) return EXIT_FAILURE;

    std::cout << std::fixed << std::setprecision(6)
              << "ThunderFFT full FFT-Conv (N=" << N
              << ", include_filter_fft=" << include_filter_fft
              << ") time: " << ms << " ms\n";
    return 0;
}
