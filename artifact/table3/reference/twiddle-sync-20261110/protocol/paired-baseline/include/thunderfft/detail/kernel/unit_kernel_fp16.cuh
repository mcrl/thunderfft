namespace thunderfft::unit_fp16 {

static constexpr int warp_size = 32;

__device__ __forceinline__ float W_cos(int index, int N) {
    return __cosf(-2 * PI * index / N);
}

__device__ __forceinline__ int f(int x) {
    return (x % 2)*4+(x/2);
}


template <bool forward>
__device__ __forceinline__ void fill_reg_b(half2 b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k, int stage) {
    int laneid = threadIdx.x % warp_size;
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
        // b[0] = W_cos(index1,4*stride);
        // b[1] = W_cos(index2,4*stride);
        b[0] = make_half2(W_cos(index1,4*stride), W_cos(index2,4*stride));
    } else {
        // b[0] = W_cos(index1,4*stride) * (1-2*((i0+j0)&1));
        // b[1] = W_cos(index2,4*stride) * (1-2*((i1+j1)&1));
        // b[0] = ((i0+j0)&1)? W_cos(index1,4*stride) : - W_cos(index1,4*stride);
        // b[1] = ((i1+j1)&1)? W_cos(index2,4*stride) : - W_cos(index2,4*stride);
        b[0] = make_half2(((i0+j0)&1)? -W_cos(index1,4*stride) : W_cos(index1,4*stride),
                          ((i1+j1)&1)? -W_cos(index2,4*stride) : W_cos(index2,4*stride));
    }

    // b[0] = W_ptr[(index1 & (4 * stride - 1)) * (16 / stride)].x;
    // b[1] = W_ptr[(index2 & (4 * stride - 1)) * (16 / stride)].x;
}

template <bool forward>
__device__ __forceinline__ void make_reg_b(vec2_t<half> *W) {
    constexpr int n = 64;
    constexpr int radix = 4;
    for (int i = 0; i < 3; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            if( _j % ( 1<< (2-i)) == 0) {
                fill_reg_b<forward>((half2*)W, i * 2, stride, i_perm, j_perm, k, i);
                W++;
            }

        }

    }
}

template <int N, bool forward>
__device__ __forceinline__ void make_reg_b_precompute(vec2_t<half> *W) {
    auto twiddle = ThunderFFT_get_unit_twiddle_half2();

    if(!forward) twiddle += 32*36;

    if constexpr (N==64 || N==128) {
        for(int i=0; i<28; i++) W[i] = twiddle[(threadIdx.x%32) + 32*i];
    } else if constexpr (N==256 || N==1024 || N==2048 || N==4096) {
        for(int i=0; i<36; i++) W[i] = twiddle[(threadIdx.x%32) + 32*i];
    }
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

static __device__ void mma_m16n8k8_fp16_fp16_rowcol(unsigned int d[2],
                                                    const unsigned int a[2],
                                                    const unsigned int b[1],
                                                    const unsigned int c[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3}, "
                 "{%4}, "
                 "{%5, %6};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
}

template <bool forward>
__device__ void fft_kernel_r64_b16(vec2_t<half>* reg, vec2_t<half>* W)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 3;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / warp_size; // element-per-thread (if needed)

    half2 reg_frag_zero[tc_m * tc_n / warp_size / 2];
    for (int i = 0; i < tc_m * tc_n / warp_size / 2; i++)
        reg_frag_zero[i] = make_half2(0, 0);

    const int laneid = threadIdx.x % warp_size;

    W--;
    // #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        half2 reg_frag_b[tc_k * tc_n / warp_size / 2];
        // unsigned int *reg_frag_b = W-1;
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            half2 reg_frag_a[tc_m * tc_k / warp_size/2];
            half2 reg_frag_d[tc_m * tc_n / warp_size/2];

            // reg_frag_a[0] = reg[j * 2];
            // reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            // reg_frag_a[2] = reg[j * 2 + 1];
            // reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];
            // reg_frag_a[0].x = reg[j].x;
            // reg_frag_a[0].y = reg[j + n / radix].x;
            // reg_frag_a[1].x = reg[j].y;
            // reg_frag_a[1].y = reg[j + n / radix].y;
            reg_frag_a[0] = reg[j];
            reg_frag_a[1] = reg[j + n / radix];
            

    //         // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            // if( _j % ( 1<< (2-i)) == 0)
            //     reg_frag_b++;
            if( _j % ( 1<< (2-i)) == 0) {
                // fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i);
                W++;
            }

            auto reg_frag_b = (unsigned int *)W;

            mma_m16n8k8_fp16_fp16_rowcol(
                (unsigned int *)reg_frag_d, (unsigned int *)reg_frag_a,
                (unsigned int *)reg_frag_b, (unsigned int *)reg_frag_zero);

            // reg[j * 2]                                     = reg_frag_d[0];
            // reg[j * 2 + 1]                                 = reg_frag_d[1];
            // reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            // reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
            reg[j] = reg_frag_d[0];
            reg[j + n / radix] = reg_frag_d[1];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            // for (int jk = 0; jk < tc_k; ++jk) {
            //     const int j = (jk / stride) * (radix * stride); // 4*stride
            //     const int k = jk % stride;
            //     permute_radix4_tmp(
            //         reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
            //         reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
            //         reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
            //         reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
            //         laneid & 3);
            // }
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;

                // int pattern = laneid & 3;
                // if(pattern==1 || pattern==3) {
                //     swap_inline(reg[k+j], reg[k+j+stride*2]);
                //     swap_inline(reg[k+j+stride], reg[k+j+stride*3]);
                // }
                // swap_inline(reg[k + j].y, reg[k + j + stride].x);
                // swap_inline(reg[k + j + stride * 2].y, reg[k + j + stride * 3].x);
                permute_radix4_tmp(
                    reg[k + j].x, reg[k + j].y, reg[k + j + stride].x,
                    reg[k + j + stride].y, reg[k + j + stride * 2].x,
                    reg[k + j + stride * 2].y, reg[k + j + stride * 3].x,
                    reg[k + j + stride * 3].y, laneid & 3);
            }
        }
    }
}

template <bool forward>
__device__ void fft_kernel_r64_b16_fuse(vec2_t<half>* reg, unsigned int* W)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 3;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / warp_size; // element-per-thread (if needed)

    swap_thread_data(reg);
    half2 reg_frag_zero[tc_m * tc_n / warp_size / 2];
    for (int i = 0; i < tc_m * tc_n / warp_size / 2; i++)
        reg_frag_zero[i] = make_half2(0, 0);

    const int laneid = threadIdx.x % warp_size;

    W--;

    // #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        half2 reg_frag_b[tc_k * tc_n / warp_size / 2];
        // unsigned int *reg_frag_b = W-1;
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            half2 reg_frag_a[tc_m * tc_k / warp_size/2];
            half2 reg_frag_d[tc_m * tc_n / warp_size/2];

            // reg_frag_a[0] = reg[j * 2];
            // reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            // reg_frag_a[2] = reg[j * 2 + 1];
            // reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];
            // reg_frag_a[0].x = reg[j].x;
            // reg_frag_a[0].y = reg[j + n / radix].x;
            // reg_frag_a[1].x = reg[j].y;
            // reg_frag_a[1].y = reg[j + n / radix].y;
            reg_frag_a[0] = reg[j];
            reg_frag_a[1] = reg[j + n / radix];
            

    //         // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = ((j / stride) / 2 * 2) % radix;
            const int k      = j % stride;

            // if( _j % ( 1<< (2-i)) == 0)
            //     reg_frag_b++;
            if( _j % ( 1<< (2-i)) == 0) {
                // fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i);
                W++;
            }

            auto reg_frag_b = W;

            mma_m16n8k8_fp16_fp16_rowcol(
                (unsigned int *)reg_frag_d, (unsigned int *)reg_frag_a,
                (unsigned int *)reg_frag_b, (unsigned int *)reg_frag_zero);

            // reg[j * 2]                                     = reg_frag_d[0];
            // reg[j * 2 + 1]                                 = reg_frag_d[1];
            // reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            // reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
            reg[j] = reg_frag_d[0];
            reg[j + n / radix] = reg_frag_d[1];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            // for (int jk = 0; jk < tc_k; ++jk) {
            //     const int j = (jk / stride) * (radix * stride); // 4*stride
            //     const int k = jk % stride;
            //     permute_radix4_tmp(
            //         reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
            //         reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
            //         reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
            //         reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
            //         laneid & 3);
            // }
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;

                // int pattern = laneid & 3;
                // if(pattern==1 || pattern==3) {
                //     swap_inline(reg[k+j], reg[k+j+stride*2]);
                //     swap_inline(reg[k+j+stride], reg[k+j+stride*3]);
                // }
                // swap_inline(reg[k + j].y, reg[k + j + stride].x);
                // swap_inline(reg[k + j + stride * 2].y, reg[k + j + stride * 3].x);
                permute_radix4_tmp(
                    reg[k + j].x, reg[k + j].y, reg[k + j + stride].x,
                    reg[k + j + stride].y, reg[k + j + stride * 2].x,
                    reg[k + j + stride * 2].y, reg[k + j + stride * 3].x,
                    reg[k + j + stride * 3].y, laneid & 3);
            }
        }
    }
}
template <bool forward>
__device__ void fft_kernel_r16_b64(vec2_t<half>* reg, vec2_t<half>* W)
{
    // compile-time constants (function-local)
    constexpr int tc_m      = 16;
    constexpr int tc_n      = 8;
    constexpr int tc_k      = 8;
    constexpr int radix     = tc_k / 2;    // 4
    constexpr int iter      = 2;
    constexpr int n         = 64;          // 4^3
    constexpr int ept       = (n * tc_m) / warp_size; // element-per-thread (if needed)

    half2 reg_frag_zero[tc_m * tc_n / warp_size / 2];
    for (int i = 0; i < tc_m * tc_n / warp_size / 2; i++)
        reg_frag_zero[i] = make_half2(0, 0);

    const int laneid = threadIdx.x % warp_size;

    W--;
    // #pragma unroll
    for (int i = 0; i < iter; ++i) {
        const int stride = 1 << (i << 1); // 4^i

        half2 reg_frag_b[tc_k * tc_n / warp_size / 2];
        // unsigned int *reg_frag_b = W-1;
        #pragma unroll
        for (int _j = 0; _j < n / radix; ++_j) {
            int j = (_j%4 ) * 4 + _j/4;
            half2 reg_frag_a[tc_m * tc_k / warp_size/2];
            half2 reg_frag_d[tc_m * tc_n / warp_size/2];

            // reg_frag_a[0] = reg[j * 2];
            // reg_frag_a[1] = reg[j * 2 + 2 * n / radix];
            // reg_frag_a[2] = reg[j * 2 + 1];
            // reg_frag_a[3] = reg[j * 2 + 1 + 2 * n / radix];
            // reg_frag_a[0].x = reg[j].x;
            // reg_frag_a[0].y = reg[j + n / radix].x;
            // reg_frag_a[1].x = reg[j].y;
            // reg_frag_a[1].y = reg[j + n / radix].y;
            reg_frag_a[0] = reg[j];
            reg_frag_a[1] = reg[j + n / radix];
            

    //         // twiddle/permutation indices
            const int j_perm = (stride >= radix)
                                 ? ((j / (stride / radix)) / 2 * 2) % radix
                                 : 0;
            const int i_perm = i==0? ((j / stride) / 2 * 2) % radix : 0;
            const int k      = j % stride;

            // if( _j % ( 1<< (2-i)) == 0)
            //     reg_frag_b++;
            if( _j % ( 1<< (2-i)) == 0) {
                // fill_reg_b<forward>(reg_frag_b, i * 2, stride, i_perm, j_perm, k, i*2);
                W++;

                // if(threadIdx.x<32 && blockIdx.x==0) {
                //     float a1 = __half2float(reg_frag_b[0].x);
                //     float a2 = __half2float(reg_frag_b[0].y);
                //     float b1 = __half2float(W[0].x);
                //     float b2 = __half2float(W[0].y);

                //     if(abs(a1-b1)>1e-5 || abs(a2-b2) >1e-5)
                //         printf("%d %d %d %f %f %f %f\n", threadIdx.x, i, _j, a1, a2, b1, b2);
                // }
            }

            auto reg_frag_b = (unsigned int *)W;

            mma_m16n8k8_fp16_fp16_rowcol(
                (unsigned int *)reg_frag_d, (unsigned int *)reg_frag_a,
                (unsigned int *)reg_frag_b, (unsigned int *)reg_frag_zero);

            // reg[j * 2]                                     = reg_frag_d[0];
            // reg[j * 2 + 1]                                 = reg_frag_d[1];
            // reg[j * 2 + 2 * n / radix]                     = reg_frag_d[2];
            // reg[j * 2 + 1 + 2 * n / radix]                 = reg_frag_d[3];
            reg[j] = reg_frag_d[0];
            reg[j + n / radix] = reg_frag_d[1];
        }

        if (i < iter - 1) {
            // tc_k == 8 iterations
            // for (int jk = 0; jk < tc_k; ++jk) {
            //     const int j = (jk / stride) * (radix * stride); // 4*stride
            //     const int k = jk % stride;
            //     permute_radix4_tmp(
            //         reg[2 * (k + j)],                  reg[2 * (k + j) + 1],
            //         reg[2 * (k + j + stride)],         reg[2 * (k + j + stride) + 1],
            //         reg[2 * (k + j + stride * 2)],     reg[2 * (k + j + stride * 2) + 1],
            //         reg[2 * (k + j + stride * 3)],     reg[2 * (k + j + stride * 3) + 1],
            //         laneid & 3);
            // }
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;

                // int pattern = laneid & 3;
                // if(pattern==1 || pattern==3) {
                //     swap_inline(reg[k+j], reg[k+j+stride*2]);
                //     swap_inline(reg[k+j+stride], reg[k+j+stride*3]);
                // }
                // swap_inline(reg[k + j].y, reg[k + j + stride].x);
                // swap_inline(reg[k + j + stride * 2].y, reg[k + j + stride * 3].x);
                permute_radix4_tmp(
                    reg[k + j].x, reg[k + j].y, reg[k + j + stride].x,
                    reg[k + j + stride].y, reg[k + j + stride * 2].x,
                    reg[k + j + stride * 2].y, reg[k + j + stride * 3].x,
                    reg[k + j + stride * 3].y, laneid & 3);
            }
        }
        W+=24;
    }
}

__device__ __forceinline__
void smem2reg(vec2_t<half>* reg,
              const vec2_t<half>* __restrict__ s_0,
              const vec2_t<half>* __restrict__ s_1,
              int stride = 1)
{
    int laneid = threadIdx.x % warp_size;
    int block_id = blockIdx.x;
    int ept = 32; // N * batch / warp_size

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
void reg2smem(vec2_t<half>* reg,
              vec2_t<half>* __restrict__ s_0,
              vec2_t<half>* __restrict__ s_1,
              int stride = 1)
{
    int laneid = threadIdx.x % warp_size;
    int block_id = blockIdx.x;
    int ept = 32; // N * batch / warp_size
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
