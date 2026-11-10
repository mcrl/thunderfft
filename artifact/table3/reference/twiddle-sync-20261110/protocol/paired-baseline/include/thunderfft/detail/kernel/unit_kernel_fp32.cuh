namespace thunderfft::unit {
__device__ __forceinline__ float W_cos(int index, int N) {
    return __cosf(-2 * PI * index / N);
}

__device__ __forceinline__ int f(int x) {
    return (x % 2)*4+(x/2);
}

template <bool forward>
__device__ __forceinline__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k, int stage) {
    int laneid = threadIdx.x % threads_per_warp;
    int i0 = laneid / 4;       // col
    int i1 = laneid / 4;       // col
    int j0 = (laneid % 4) * 2; // row (2j, 2j+1)
    int j1 = (laneid % 4) * 2 + 1;


    if (stage == 0) {
        j0=f(j0);
        j1=f(j1);
    }
    if (stage == 2) {
        i0=f(i0);
        i1=f(i1);
    }

    i0 ^= i_perm;
    i1 ^= i_perm;
    j0 ^= j_perm;
    j1 ^= j_perm;

    // int i0 = 2*i + (threadIdx.x/4 &1);
    // int i1 = 2*i + (threadIdx.x/4 &1);
    // int j0 = 2*j;
    // int j1 = 2*j + 1;

    i0 = (i0 % 4) * 2 + i0 / 4;
    i1 = (i1 % 4) * 2 + i1 / 4;

    j0 = (j0 % 4) * 2 + j0 / 4;
    j1 = (j1 % 4) * 2 + j1 / 4;

    int index1 = (j0 / 2) * (k + stride * (i0 / 2)) + stride * (i0 & 1) -
                    stride * (j0 & 1);
    int index2 = (j1 / 2) * (k + stride * (i1 / 2)) + stride * (i1 & 1) -
                    stride * (j1 & 1);
    // auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    if constexpr (forward) {
        b[0] = W_cos(index1,4*stride);
        b[1] = W_cos(index2,4*stride);
        // b[0] = 3.1f;
        // b[1] = 3.2f;
    } else {
        b[0] = ((i0+j0)&1)? -W_cos(index1,4*stride) : W_cos(index1,4*stride);
        b[1] = ((i1+j1)&1)? -W_cos(index2,4*stride) : W_cos(index2,4*stride);
        // b[0] = W_cos(index1,4*stride) * (1-2*((i0+j0)&1));
        // b[1] = W_cos(index2,4*stride) * (1-2*((i1+j1)&1));
        // b[0] = 3.3f;
        // b[1] = 3.4f;
    }

    // b[0] = W_ptr[(index1 & (4 * stride - 1)) * (16 / stride)].x;
    // b[1] = W_ptr[(index2 & (4 * stride - 1)) * (16 / stride)].x;
}

template <bool forward>
__device__ __forceinline__ void make_reg_b(float2 *W) {
    float2* W_ptr = ThunderFFT_get_unit_twiddle_float2();
    if(!forward) W_ptr += 32 * 28;

    for(int i=0; i<28; i++) {
        W[i] = W_ptr[32*i + threadIdx.x % 32];
    }
    // constexpr int n = 64;
    // constexpr int radix = 4;
    // for (int i = 0; i < 3; ++i) {
    //     const int stride = 1 << (i << 1); // 4^i

    //     #pragma unroll
    //     for (int _j = 0; _j < n / radix; ++_j) {
    //         int j = (_j%4 ) * 4 + _j/4;
    //         const int j_perm = (stride >= radix)
    //                              ? ((j / (stride / radix)) / 2 * 2) % radix
    //                              : 0;
    //         const int i_perm = ((j / stride) / 2 * 2) % radix;
    //         const int k      = j % stride;

    //         if( _j % ( 1<< (2-i)) == 0) {
    //             fill_reg_b<forward>((float*)W, i * 2, stride, i_perm, j_perm, k, i);
    //             W++;
    //         }

    //     }

    // }
}
template <typename T>
__device__ void permute_radix4_tmp(T &a, T &b, T &c, T &d, T &e, T &f, T &g,
                                   T &h, int pattern) {
    if (pattern == 1 || pattern == 3) {
        swap_inline(a, e);
        swap_inline(b, f);
        swap_inline(c, g);
        swap_inline(d, h);
    }
    swap_inline(b, c);
    swap_inline(f, g);
}

static __device__ __forceinline__ void mma_m16n8k8_tf32_f32_rowcol(float d[4], const float a[4],
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
    // if (threadIdx.y==0 && blockIdx.x==0) {
    //     printf("%d %f %f %f %f %f %f %f %f %f %f\n", threadIdx.x, a[0], a[1],
    //     a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3]);
    // }

}

template <bool forward>
__device__ void fft_kernel_r64_b16(float* reg)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 3;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / threads_per_warp; // element-per-thread (if needed)

    float reg_frag_zero[tc_m * tc_n / threads_per_warp];
    #pragma unroll
    for (int i = 0; i < tc_m * tc_n / threads_per_warp; ++i)
        reg_frag_zero[i] = 0.0f;

    const int laneid = threadIdx.x % threads_per_warp;


    // const float* W = thunderfft::ThunderFFT_get_twiddle<float>();

    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        float reg_frag_b[tc_k * tc_n / threads_per_warp];
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            // int j = _j;
            float reg_frag_a[tc_m * tc_k / threads_per_warp];
            float reg_frag_d[tc_m * tc_n / threads_per_warp];

            reg_frag_a[0] = reg[j * 2];
            reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            reg_frag_a[2] = reg[j * 2 + 1];
            reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];

            // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            if( _j % ( 1<< (2-i)) == 0) {
                fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i);
            }

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

            reg[j * 2]                                     = reg_frag_d[0];
            reg[j * 2 + 1]                                 = reg_frag_d[1];
            reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            for (int jk = 0; jk < tc_k; ++jk) {
                const int j = (jk / stride) * (radix * stride); // 4*stride
                const int k = jk % stride;
                permute_radix4_tmp(
                    reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
                    reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
                    reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
                    reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
                    laneid & 3);
            }
        }
    }
}


template <bool forward>
__device__ void fft_kernel_r64_b16_precompute(float* reg, float2* W)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 3;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / threads_per_warp; // element-per-thread (if needed)

    float reg_frag_zero[tc_m * tc_n / threads_per_warp];
    #pragma unroll
    for (int i = 0; i < tc_m * tc_n / threads_per_warp; ++i)
        reg_frag_zero[i] = 0.0f;

    const int laneid = threadIdx.x % threads_per_warp;
    W--;

    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        float reg_frag_b[tc_k * tc_n / threads_per_warp];
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            // int j = _j;
            float reg_frag_a[tc_m * tc_k / threads_per_warp];
            float reg_frag_d[tc_m * tc_n / threads_per_warp];

            reg_frag_a[0] = reg[j * 2];
            reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            reg_frag_a[2] = reg[j * 2 + 1];
            reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];

            // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            if( _j % ( 1<< (2-i)) == 0) {
                W++;
                // fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i);
            }

            auto reg_frag_b = (float*)W;
            
            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

            reg[j * 2]                                     = reg_frag_d[0];
            reg[j * 2 + 1]                                 = reg_frag_d[1];
            reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            for (int jk = 0; jk < tc_k; ++jk) {
                const int j = (jk / stride) * (radix * stride); // 4*stride
                const int k = jk % stride;
                permute_radix4_tmp(
                    reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
                    reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
                    reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
                    reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
                    laneid & 3);
            }
        }
    }
}


template <bool forward>
__device__ void fft_kernel_r16_b64(float* reg)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 2;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / threads_per_warp; // element-per-thread (if needed)

    float reg_frag_zero[tc_m * tc_n / threads_per_warp];
    #pragma unroll
    for (int i = 0; i < tc_m * tc_n / threads_per_warp; ++i)
        reg_frag_zero[i] = 0.0f;

    const int laneid = threadIdx.x % threads_per_warp;


    // const float* W = thunderfft::ThunderFFT_get_twiddle<float>();

    #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        float reg_frag_b[tc_k * tc_n / threads_per_warp];
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            // int j = _j;
            float reg_frag_a[tc_m * tc_k / threads_per_warp];
            float reg_frag_d[tc_m * tc_n / threads_per_warp];

            reg_frag_a[0] = reg[j * 2];
            reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            reg_frag_a[2] = reg[j * 2 + 1];
            reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];

            // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = i==0? ((j / stride) / 2 * 2) % radix : 0;
            const int k      = j % stride;

            if( _j % ( 1<< (2-i)) == 0)
                fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i*2);

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

            reg[j * 2]                                     = reg_frag_d[0];
            reg[j * 2 + 1]                                 = reg_frag_d[1];
            reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            for (int jk = 0; jk < tc_k; ++jk) {
                const int j = (jk / stride) * (radix * stride); // 4*stride
                const int k = jk % stride;
                permute_radix4_tmp(
                    reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
                    reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
                    reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
                    reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
                    laneid & 3);
            }
        }
    }
}
// {
//     // compile-time constants (function-local)
//     constexpr int tc_m      = 16;
//     constexpr int tc_n      = 8;
//     constexpr int tc_k      = 8;
//     constexpr int radix     = tc_k / 2;    // 4
//     constexpr int iter      = 2;
//     constexpr int n         = 64;          // 4^3
//     constexpr int ept       = (n * tc_m) / threads_per_warp; // element-per-thread (if needed)

//     float reg_frag_zero[tc_m * tc_n / threads_per_warp];
//     #pragma unroll
//     for (int i = 0; i < tc_m * tc_n / threads_per_warp; ++i)
//         reg_frag_zero[i] = 0.0f;

//     const int laneid = threadIdx.x % threads_per_warp;

//     #pragma unroll
//     for (int i = 0; i < iter; ++i) {
//         const int stride = 1 << (i << 1); // 4^i

//         float reg_frag_b[tc_k * tc_n / threads_per_warp];
//         #pragma unroll
//         for (int _j = 0; _j < n / radix; ++_j) {
//             int j = (_j%4 ) * 4 + _j/4;
//             float reg_frag_a[tc_m * tc_k / threads_per_warp];
//             float reg_frag_d[tc_m * tc_n / threads_per_warp];

//             reg_frag_a[0] = reg[j * 2];
//             reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
//             reg_frag_a[2] = reg[j * 2 + 1];
//             reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];

//             // twiddle/permutation indices
//             const int j_perm = (stride >= radix)
//                                  ? ((j / (stride / radix)) / 2 * 2) % radix
//                                  : 0;
//             const int i_perm = ((j / stride) / 2 * 2) % radix;
//             const int k      = j % stride;

//             if( _j % ( 1<< (2-i)) == 0)
//                 fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i*2);

//             mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b, reg_frag_zero);

//             reg[j * 2]                                     = reg_frag_d[0];
//             reg[j * 2 + 1]                                 = reg_frag_d[1];
//             reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
//             reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
//         }

//         if (i < iter - 1) {
//             // tc_k == 8 iterations
//             for (int jk = 0; jk < tc_k; ++jk) {
//                 const int j = (jk / stride) * (radix * stride); // 4*stride
//                 const int k = jk % stride;
//                 permute_radix4_tmp(
//                     reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
//                     reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
//                     reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
//                     reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
//                     laneid & 3);
//             }
//         }
//     }
// }

__device__ __forceinline__
void smem2reg(vec2_t<float>* reg,
              const vec2_t<float>* __restrict__ s_0,
              const vec2_t<float>* __restrict__ s_1,
              int stride = 1)
{
    int laneid = threadIdx.x % threads_per_warp;
    // int block_id = blockIdx.x;
    int ept = 32; // N * batch / threads_per_warp

    int b =  ((laneid>>1) & 1);
    for (int i = 0; i < ept / 2; i++) { 
        // reg[i] = f_0[laneid + 32*i];
        // reg[i+ept/2] = f_1[laneid + 32*i];

        reg[i] =
            s_0[(i * 4 + (laneid % 4) ) * stride];
        reg[i + ept / 2] =
            s_1[(i * 4 + (laneid % 4) ) * stride];
        // reg[2 * i] =
        //     f_0[(i * 4 + (laneid % 2) * 2    ) * stride * 2 + b];
        // reg[2 * i + 1] =
        //     f_0[(i * 4 + (laneid % 2) * 2 + 1) * stride * 2 + b];
        // reg[2 * i + ept] =
        //     f_1[(i * 4 + (laneid % 2) * 2    ) * stride * 2 + b];
        // reg[2 * i + ept + 1] =
        //     f_1[(i * 4 + (laneid % 2) * 2 + 1) * stride * 2 + b];
    }
//     if(blockIdx.x==0 && threadIdx.x==0){
//         for(int i=0; i<ept; i++){
//             printf("reg[%d]=%f\n", i, reg[i]);
//         }
//     }
}

__device__ __forceinline__
void reg2smem(vec2_t<float>* reg,
              vec2_t<float>* __restrict__ s_0,
              vec2_t<float>* __restrict__ s_1,
              int stride = 1)
{
    int laneid = threadIdx.x % threads_per_warp;
    int block_id = blockIdx.x;
    int ept = 32; // N * batch / threads_per_warp
    int b =  ((laneid>>1) & 1);


    for (int i = 0; i < ept / 2; i++) {
        // f_0[laneid + 32*i] = reg[i];
        // f_1[laneid + 32*i] = reg[i+ept/2];
        // if ((laneid % 4) < 2) {
        //     s_0[(i / 2 + (i & 1) * 16 + (laneid % 2) * 32)*stride]
        //         .x = reg[i];
        //     s_1[(i / 2 + (i & 1) * 16 +
        //            (laneid % 2) * 32)*stride]
        //         .x = reg[i + ept];
        // } else {
        //     s_0[(i / 2 + (i & 1) * 16 + (laneid % 2) * 32)*stride]
        //         .y = reg[i];
        //     s_1[(i / 2 + (i & 1) * 16 + (laneid % 2) * 32)*stride]
        //         .y = reg[i + ept];
        // }
        s_0[(i + (laneid % 4) * (17))*stride] = reg[i];
        s_1[(i + (laneid % 4) * (17))*stride] = reg[i + ept/2];
    }
}

} // namespace thunderfft::detail::unit