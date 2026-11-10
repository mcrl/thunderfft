#pragma once

#include <cuComplex.h>
#include <cuda_runtime.h>
#include "historical_branch_51e346c.cuh"

// Historical compute-only prototype from commit e4aa1b0.
// Timing-only use: the original B-fragment setup is commented out, leaving
// indeterminate MMA operands. Compiler choices affect this workload.
// Original executable statements and omissions are preserved; this source
// provides historical instruction timing, without numerical FFT validation.
// One warp per block; batch must be a positive multiple of 16. W_64 is unused.
namespace thff_table7_e4aa1b0 {
using thff_table7_51e346c::fill_reg_b;

static constexpr int TC_M_DEVICE_CONST = 16;
static constexpr int TC_N_DEVICE_CONST = 8;
static constexpr int TC_K_DEVICE_CONST = 8;
static constexpr int RADIX_DEVICE_CONST = TC_K_DEVICE_CONST / 2;
static constexpr int ITER_DEVICE_CONST = 3;
static constexpr int N_DEVICE_CONST = 64;
static constexpr int WARP_SIZE_DEVICE_CONST = 32;

// Source: fft/my_fft.h:22-30.
template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

// Source: fft/my_fft.cu:148-165.
static __device__ void mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4],
                                                   const float b[2],
                                                   const float c[4]) {
    auto a0 = __float_as_uint(a[0]);
    auto a1 = __float_as_uint(a[1]);
    auto a2 = __float_as_uint(a[2]);
    auto a3 = __float_as_uint(a[3]);
    auto b0 = __float_as_uint(b[0]);
    auto b1 = __float_as_uint(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "      // D (also C)
                 "{%4, %5, %6, %7}, "      // A (tf32 in .b32 regs)
                 "{%8, %9}, "              // B (tf32 in .b32 regs)
                 "{%10, %11, %12, %13};\n" // C
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Source: fft/my_fft.cu:167-175.
template <typename T>
__device__ void permute_radix4_local(T &a, T &b, T &c, T &d, int pattern) {
    // version 1
    T tmp[4] = {a, b, c, d};
    a = tmp[pattern];
    b = tmp[(pattern - 1) & 3];
    c = tmp[(pattern - 2) & 3];
    d = tmp[(pattern - 3) & 3];
}

// Source: fft/my_fft.cu:176-197.
template <typename T>
__device__ void permute_radix4_branch(T &a, T &b, T &c, T &d, int pattern) {
    if (pattern == 1 || pattern == 3) {
        T tmp = a;
        a = b;
        b = tmp;

        tmp = c;
        c = d;
        d = tmp;
    }
    if (pattern == 0 || pattern == 3) {
        T tmp = b;
        b = d;
        d = tmp;
    }
    if (pattern == 2 || pattern == 3) {
        T tmp = a;
        a = c;
        c = tmp;
    }
}

// Source: fft/my_fft.cu:359-477.
template <int N>
__device__ void fft_kernel_r64_b16(cuFloatComplex *reg,
                                   const cuFloatComplex *W_4096) {
    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
#pragma unroll
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];

            reg_frag_a[0] = reg[j].x;
            reg_frag_a[1] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x;
            reg_frag_a[2] = reg[j].y;
            reg_frag_a[3] = reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y;

            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                 W_4096);
            // fill_reg_b<N>(reg_frag_b, stride, i_perm, j_perm, k, W_4096);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, reg_frag_b[0], reg_frag_b[1]);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);

            reg[j].x = reg_frag_d[0];
            reg[j].y = reg_frag_d[1];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].x = reg_frag_d[2];
            reg[j + N_DEVICE_CONST / RADIX_DEVICE_CONST].y = reg_frag_d[3];
        }

        if (i == 0) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9	->  10 14 2  6
                // 13 14 15 12		7  11 15 3
                permute_radix4_local(reg[k + j], reg[k + j + stride],
                                     reg[k + j + stride * 2],
                                     reg[k + j + stride * 3], laneid & 3);
            }
        }
        if (i == 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9	->  10 14 2  6
                // 13 14 15 12		7  11 15 3
                permute_radix4_branch(reg[k + j], reg[k + j + stride],
                                      reg[k + j + stride * 2],
                                      reg[k + j + stride * 3], laneid & 3);
            }
        }
        /*if (i < ITER_DEVICE_CONST - 1) {
            // #pragma unroll
            // for (int j = 0; j < 32; j += 4 * stride) {
            //     #pragma unroll
            //     for (int k = 0; k < stride; k++) {
            //         // int
        perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
            //         // t0 t1 t2 t3
            //         // 0  1  2  3       0  4  8  12
            //         // 7  4  5  6       13 1  5  9
            //         // 10 11 8  9	->  10 14 2  6
            //         // 13 14 15 12		7  11 15 3
            //         permute_radix4(reg[k + j], reg[k + j + stride],
            //                        reg[k + j + stride * 2],
            //                        reg[k + j + stride * 3], laneid & 3);
            //     }
            // }

            for (int jk = 0; jk < 8; jk ++) {
                int j= (jk / stride) * (4*stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9	->  10 14 2  6
                // 13 14 15 12		7  11 15 3
                permute_radix4(reg[k + j], reg[k + j + stride],
                                reg[k + j + stride * 2],
                                reg[k + j + stride * 3], laneid & 3);
            }
        }*/
    }
}

// Source: fft/my_fft.cu:615-691.
__global__ void
fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64,
                           unsigned int repeat) {
    // Tensor core shape
    constexpr int m = 16;
    constexpr int n = 8;
    constexpr int k = 8;

    constexpr int radix = k / 2; // = 4
    constexpr int iter = 3;
    constexpr int N = 64; // radix^iter
    constexpr int batch = m;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch / warp_size; // element_per_thread

    // Registers for data
    cuFloatComplex reg[ept];

    __shared__ cuFloatComplex s_data[ept * (warp_size + 1)];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    for (int i = 0; i < ept; i++) {
        s_data[i * (warp_size + 1) + laneid] =
            d_data[block_id * N * batch + i * warp_size + laneid];
    }

    __syncwarp();
    // for (int i = 0; i < ept / 2; i++) {
    //     reg[i] = s_data[(laneid / 2) * (warp_size + 1) +
    //                     reverse_2bit_groups<4>(i) + (ept / 2) * (laneid %
    //                     2)];
    //     reg[i + ept / 2] =
    //         s_data[(ept / 2) * (warp_size + 1) +
    //                (laneid / 2) * (warp_size + 1) + reverse_2bit_groups<4>(i)
    //                + (ept / 2) * (laneid % 2)];
    // }

    for (unsigned int i = 0; i < repeat; i++) {
        for (int i = 0; i < ept / 2; i++) {
            reg[i] =
                s_data[(laneid / 2) * (warp_size + 1) +
                       reverse_2bit_groups<4>(i) + (ept / 2) * (laneid % 2)];
            reg[i + ept / 2] =
                s_data[(ept / 2) * (warp_size + 1) +
                       (laneid / 2) * (warp_size + 1) +
                       reverse_2bit_groups<4>(i) + (ept / 2) * (laneid % 2)];
        }

        fft_kernel_r64_b16<64>(reg, W_64);

        for (int i = 0; i < ept / 2; i++) {
            s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] =
                reg[i];
            s_data[(ept / 2) * (warp_size + 1) +
                   (warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] =
                reg[i + ept / 2];
        }
        __syncwarp();
    }

    // write to smem
    // for (int i = 0; i < ept / 2; i++) {
    //     s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] =
    //     reg[i]; s_data[(ept / 2) * (warp_size + 1) + (warp_size + 1) *
    //     (laneid / 2) +
    //            16 * (laneid % 2) + i] = reg[i + ept / 2];
    // }
    // __syncwarp();

    // write to gmem
    for (int i = 0; i < ept; i++)
        d_data[block_id * N * batch + laneid + i * warp_size] =
            s_data[i * (warp_size + 1) + laneid];
}

} // namespace thff_table7_e4aa1b0
