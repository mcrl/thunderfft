#pragma once

#include <cmath>
#include <cuComplex.h>
#include <cuda_runtime.h>

// Historical TF32 candidates recovered from the original project, not certified
// identities of the paper's ablation variants. See provenance.json.
// One warp per block; batch must be a positive multiple of 16. Both kernels are
// in-place, forward, unnormalized N=64 transforms. W_64 is retained for the
// historical interface; these TF32 helpers compute their twiddles directly.
// Compile this header in one CUDA translation unit per executable.
namespace thff_table7_legacy {

static constexpr int TC_M_DEVICE_CONST = 16;
static constexpr int TC_N_DEVICE_CONST = 8;
static constexpr int TC_K_DEVICE_CONST = 8;
static constexpr int RADIX_DEVICE_CONST = TC_K_DEVICE_CONST / 2;
static constexpr int ITER_DEVICE_CONST = 3;
static constexpr int N_DEVICE_CONST = 64;
static constexpr int WARP_SIZE_DEVICE_CONST = 32;
static constexpr double kLegacyPi = 3.14159265358979323846;

// Source: fft/utils.h:66-69; PI macro replaced by local same-value constant.
__device__ __forceinline__ float2 W(int index, int N) {
    return make_float2(__cosf(-2 * kLegacyPi * index / N),
                       __sinf(-2 * kLegacyPi * index / N));
}

// Source: fft/my_fft.h:23-31.
template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

// Source: fft/my_fft.cu:73-106.
template <unsigned int N>
__device__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_ptr) {
    int i0 = threadIdx.x / 4; //col
    int i1 = threadIdx.x / 4; //col
    int j0 = (threadIdx.x % 4) * 2; // row (2j, 2j+1)
    int j1 = (threadIdx.x % 4) * 2 + 1;

    i0^=i_perm;
    i1^=i_perm;
    j0^=j_perm;
    j1^=j_perm;

    // int i0 = 2*i + (threadIdx.x/4 &1);
    // int i1 = 2*i + (threadIdx.x/4 &1);
    // int j0 = 2*j;
    // int j1 = 2*j + 1;

    i0 = (i0 % 4) * 2 + i0/4;
    i1 = (i1 % 4) * 2 + i1/4;

    j0 = (j0 % 4) * 2 + j0/4;
    j1 = (j1 % 4) * 2 + j1/4;



    int index1 = (j0/2)*(k+stride*(i0/2)) + stride * (i0 & 1) - stride * (j0 & 1);
    int index2 = (j1/2)*(k+stride*(i1/2)) + stride * (i1 & 1) - stride * (j1 & 1);
    // auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    b[0] = W(index1,4*stride).x;
    b[1] = W(index2,4*stride).x;
}

// Source: fft/my_fft.cu:108-126.
template <unsigned int N>
__device__ void fill_reg_b_branch(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_ptr) {
    int i = (threadIdx.x / 8 - i_perm) & 3;
    int j = (threadIdx.x % 4 - j_perm) & 3;

    int i0 = 2*i + (threadIdx.x/4 &1);
    int i1 = 2*i + (threadIdx.x/4 &1);
    int j0 = 2*j;
    int j1 = 2*j + 1;

    int index1 = (j0/2)*(k+stride*(i0/2)) + stride * (i0 & 1) - stride * (j0 & 1);
    int index2 = (j1/2)*(k+stride*(i1/2)) + stride * (i1 & 1) - stride * (j1 & 1);
    auto w = W(j*(k+stride*i) + stride * ((threadIdx.x / 4) & 1),4*stride);

    b[0] = W(index1,4*stride).x;
    b[1] = W(index2,4*stride).x;
}

// Source: fft/my_fft.cu:141-161.
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
    // if (threadIdx.y==0 && blockIdx.x==0) {
    //     printf("%d %f %f %f %f %f %f %f %f %f %f\n", threadIdx.x, a[0], a[1], a[2], a[3], b[0], b[1], d[0], d[1], d[2], d[3]);
    // }
}

// Source: fft/my_fft.cu:263-268.
template <typename T>
__device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}

// Source: fft/my_fft.cu:269-279.
template <typename T>
__device__ void permute_radix4_tmp(T &a, T &b, T &c, T &d, T &e, T &f, T &g, T &h, int pattern) {
    if (pattern == 1 || pattern == 3) {
        swap_inline(a,e);
        swap_inline(b,f);
        swap_inline(c,g);
        swap_inline(d,h);
    }
    swap_inline(b,c);
    swap_inline(f,g);
}

// Source: fft/my_fft.cu:281-417.
template <typename T>
__device__ void permute_radix4(T &a, T &b, T &c, T &d, int pattern) {
    // version 0
    // T t0 = a, t1 = b, t2 = c, t3 = d;
    // switch (pattern & 3) {
    // // {0,3,2,1}
    // case 0:
    //     a = t0;
    //     b = t3;
    //     c = t2;
    //     d = t1;
    //     break;
    // // {1,0,3,2}
    // case 1:
    //     a = t1;
    //     b = t0;
    //     c = t3;
    //     d = t2;
    //     break;
    // // {2,1,0,3}
    // case 2:
    //     a = t2;
    //     b = t1;
    //     c = t0;
    //     d = t3;
    //     break;
    // // {3,2,1,0}
    // default:
    //     a = t3;
    //     b = t2;
    //     c = t1;
    //     d = t0;
    //     break;
    // }
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

    // version 1
    // T tmp[4] = {a,b,c,d};
    // a=tmp[pattern];
    // b=tmp[(pattern-1)&3];
    // c=tmp[(pattern-2)&3];
    // d=tmp[(pattern-3)&3];

    // version 2 (x 2)
    // unsigned int buf_x[2] = {__byte_perm(__float_as_uint(a.x),
    // __float_as_uint(b.x), 0x5410),
    //                       __byte_perm(__float_as_uint(c.x),
    //                       __float_as_uint(d.x), 0x5410)};
    // unsigned int buf_y[2] = {__byte_perm(__float_as_uint(a.x),
    // __float_as_uint(b.x), 0x7632),
    //                       __byte_perm(__float_as_uint(c.x),
    //                       __float_as_uint(d.x), 0x7632)};

    // auto tmp_x = (long long*)(buf_x);
    // *tmp_x = ((*tmp_x) >> (pattern * 16)) | ((*tmp_x) << ((4 - pattern) *
    // 16)); auto tmp_y = (long long*)(buf_y); *tmp_y = ((*tmp_y) >> (pattern *
    // 16)) | ((*tmp_y) << ((4 - pattern) * 16));

    // a.x = __byte_perm(buf_x[0], buf_y[0], 0x5410);
    // b.x = __byte_perm(buf_x[0], buf_y[0], 0x7632);
    // c.x = __byte_perm(buf_x[1], buf_y[1], 0x5410);
    // d.x = __byte_perm(buf_x[1], buf_y[1], 0x7632);

    // version 3
    // half2 buf[2] = {{(((half2*)(&a))[0]).x, (*(half2*)(&(b.x))).x},
    // {(*(half2*)(&(c.x))).x, (*(half2*)(&(d.x))).x}}; half2 buf[2] =
    // {{(*(half2*)(&(a.x))).y, (*(half2*)(&(b.x))).y}, {(*(half2*)(&(c.x))).y,
    // (*(half2*)(&(d.x))).y}}; auto tmp = reinterpret_cast<long long*>(buf);
    // *tmp = ((*tmp) >> (pattern * 16)) | ((*tmp) << ((4 - pattern) * 16));

    // (*(half2*)(&(a.x))).y = buf[0].x;
    // (*(half2*)(&(b.x))).y = buf[0].y;
    // (*(half2*)(&(c.x))).y = buf[1].x;
    // (*(half2*)(&(d.x))).y = buf[1].y;

    // printf("%.3f %.3f %.3f %.3f -> (%.3f %.3f %.3f %.3f) %.3f %.3f %.3f
    // %.3f\n", t0.x, t1.x, t2.x, t3.x, __half2float(buf[0].x),
    // __half2float(buf[0].y), __half2float(buf[1].x),
    // __half2float(buf[1].y),a.x, b.x, c.x, d.x);

    // version 4

    // float2 x[2] = {{a.x,b.x},{c.x,d.x}};
    // unsigned long long *lx = reinterpret_cast<unsigned long long*>(x);

    // lx[0] = (lx[0] >> (pattern * 16)) | (lx[0] << (64 - pattern * 16));
    // lx[1] = (lx[1] >> (pattern * 16)) | (lx[1] << (64 - pattern * 16));
    // a.x = x[0].x, b.x=x[0].y, c.x=x[1].x, d.x=x[1].y;

    // float2 y[2] = {{a.y,b.y},{c.y,d.y}};
    // unsigned long long *ly = reinterpret_cast<unsigned long long*>(y);

    // ly[0] = (ly[0] >> (pattern * 16)) | (ly[0] << (64 - pattern * 16));
    // ly[1] = (ly[1] >> (pattern * 16)) | (ly[1] << (64 - pattern * 16));
    // a.y = y[0].x, b.y=y[0].y, c.y=y[1].x, d.y=y[1].y;

    // y[0] = {a.y, b.y};
    // y[1] = {c.y, d.y};
    // ly[0] = (ly[0] >> (pattern * 16)) | (ly[0] << (64 - pattern * 16));
    // ly[1] = (ly[1] >> (pattern * 16)) | (ly[1] << (64 - pattern * 16));
    // a.y = y[0].x, b.y=y[0].y, c.y=y[1].x, d.y=y[1].y;
    // version 5

    // float tmp[4] = {a.x,b.x,c.x,d.x};
    // a.x = tmp[0]*(pattern==0) + tmp[1]*(pattern==1) + tmp[2]*(pattern==2) +
    // tmp[3]*(pattern==3); b.x = tmp[3]*(pattern==0) + tmp[0]*(pattern==1) +
    // tmp[1]*(pattern==2) + tmp[2]*(pattern==3); c.x = tmp[2]*(pattern==0) +
    // tmp[3]*(pattern==1) + tmp[0]*(pattern==2) + tmp[1]*(pattern==3); d.x =
    // tmp[1]*(pattern==0) + tmp[2]*(pattern==1) + tmp[3]*(pattern==2) +
    // tmp[0]*(pattern==3);

    // float tmp2[4] = {a.y,b.y,c.y,d.y};
    // a.y = tmp2[0]*(pattern==0) + tmp2[1]*(pattern==1) + tmp2[2]*(pattern==2)
    // + tmp2[3]*(pattern==3); b.y = tmp2[3]*(pattern==0) + tmp2[0]*(pattern==1)
    // + tmp2[1]*(pattern==2) + tmp2[2]*(pattern==3); c.y = tmp2[2]*(pattern==0)
    // + tmp2[3]*(pattern==1) + tmp2[0]*(pattern==2) + tmp2[1]*(pattern==3); d.y
    // = tmp2[1]*(pattern==0) + tmp2[2]*(pattern==1) + tmp2[3]*(pattern==2) +
    // tmp2[0]*(pattern==3);
}

// Source: fft/my_fft.cu:420-534.
template <int N>
__device__ void fft_kernel_r64_b16(float *reg,
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
            reg_frag_a[0] = reg[j*2];
            reg_frag_a[1] = reg[j*2 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST];
            reg_frag_a[2] = reg[j*2+1];
            reg_frag_a[3] = reg[j*2+1 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST];
            // w = w_4stride
            // b = [ w^ i ( k + Nj) ] ^ T
            int j_perm;
            if (stride >= 4)
                j_perm = ((j / (stride / 4)) / 2 * 2) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;
            int i_perm = ((j / stride) / 2 * 2) % RADIX_DEVICE_CONST;
            int k = j % stride;
            fill_reg_b<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                W_4096);
            // fill_reg_b<N>(reg_frag_b, stride, i_perm, j_perm, k, W_4096);
            // printf("%d %d %d %d %d : %f %f\n", threadIdx.x, stride, i_perm,
            // j_perm, k, reg_frag_b[0], reg_frag_b[1]);
            // if(threadIdx.x==0 && blockIdx.x==0 ) { printf("\nidx %d %d %d %d\n", i, j, i_perm, j_perm);}
            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);
            reg[j*2] = reg_frag_d[0];
            reg[j*2+1] = reg_frag_d[1];
            reg[j*2 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[2];
            reg[j*2+1 + 2*N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[3];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
                // t0 t1 t2 t3
                // 0  1  2  3       0  4  8  12
                // 7  4  5  6       13 1  5  9
                // 10 11 8  9	->  10 14 2  6
                // 13 14 15 12		7  11 15 3
                permute_radix4_tmp(reg[2*(k + j)], reg[2*(k+j)+1],
                                     reg[2*(k + j + stride)], reg[2*(k + j + stride)+1],
                                     reg[2*(k + j + stride * 2)], reg[2*(k + j + stride*2)+1],
                                     reg[2*(k + j + stride * 3)],reg[2*(k + j + stride*3)+1],
                                        laneid & 3);
            }
        }
        // if (i < ITER_DEVICE_CONST - 1) {
        //     for (int jk = 0; jk < 8; jk++) {
        //         int j = (jk / stride) * (4 * stride);
        //         int k = jk % stride;
        //         // int perm[4][4]={0,3,2,1},{1,0,3,2},{2,1,0,3},{3,2,1,0};
        //         // t0 t1 t2 t3
        //         // 0  1  2  3       0  4  8  12
        //         // 7  4  5  6       13 1  5  9
        //         // 10 11 8  9   ->  10 14 2  6
        //         // 13 14 15 12      7  11 15 3
        //         permute_radix4_tmp(reg[2*(k + j)], reg[2*(k+j)+1],
        //                              reg[2*(k + j + stride)], reg[2*(k + j + stride)+1],
        //                              reg[2*(k + j + stride * 2)], reg[2*(k + j + stride*2)+1],
        //                              reg[2*(k + j + stride * 3)],reg[2*(k + j + stride*3)+1],
        //                                 laneid & 3);
        //     }
        // }
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
            //         // 10 11 8  9    ->  10 14 2  6
            //         // 13 14 15 12       7  11 15 3
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
                // 10 11 8  9   ->  10 14 2  6
                // 13 14 15 12      7  11 15 3
                permute_radix4(reg[k + j], reg[k + j + stride],
                                reg[k + j + stride * 2],
                                reg[k + j + stride * 3], laneid & 3);
            }
        }*/
    }
}

// Source: fft/my_fft.cu:672-767.
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
    // cuFloatComplex reg[ept];

    extern __shared__ __align__(sizeof(float4)) cuFloatComplex s_data[];

    int laneid = threadIdx.x;
    int block_id = blockIdx.x;

    // for (int i = 0; i < ept; i++) {
    //     s_data[i * (warp_size + 2) + laneid/(warp_size/2) * (warp_size/2 + 1) + laneid%(warp_size/2)] =
    //         d_data[block_id * N * batch + i * warp_size + laneid];
    // }

    // __syncwarp();

    // for (int i = 0; i < ept / 2; i++) {
    //     reg[i] = s_data[laneid * (warp_size/2 + 1) +
    //                     reverse_2bit_groups<4>(i)];
    //     reg[i + ept / 2] =
    //         s_data[ept * (warp_size/2 + 1) +
    //             laneid * (warp_size/2 + 1) + reverse_2bit_groups<4>(i)];
    // }
    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            s_data[i + N * j] =
                d_data[reverse_2bit_groups<6>(i) + N * j +
                       (threadIdx.y + blockIdx.x * blockDim.y) * N * batch];
        }
    }
    __syncwarp();

    float reg[ept*2];
    for (int i = 0; i < ept/2; i++) {
        if((laneid % 4) < 2) {
            reg[2*i] = s_data[laneid / 4 * N  + i*4 + (laneid % 2)*2].x;
            reg[2*i+1] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].x;
            reg[2*i + ept] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2].x;
            reg[2*i + ept+1] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].x;
        } else {
            reg[2*i] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2].y;
            reg[2*i+1] = s_data[laneid / 4 * N + i*4 + (laneid % 2)*2 + 1].y;
            reg[2*i + ept] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 ].y;
            reg[2*i + ept+1] = s_data[(laneid / 4 + 8) * N + i*4 + (laneid % 2)*2 + 1].y;
        }
    }

    #pragma unroll 1
    for (unsigned int i = 0; i < repeat; i++) {
        // for (int i = 0; i < ept / 2; i++) {
        //     reg[i] = s_data[laneid * (warp_size/2 + 1) +
        //                     reverse_2bit_groups<4>(i)];
        //     reg[i + ept / 2] =
        //         s_data[ept * (warp_size/2 + 1) +
        //             laneid * (warp_size/2 + 1) + reverse_2bit_groups<4>(i)];
        // }
        // __syncwarp();
        fft_kernel_r64_b16<64>(reg, W_64);
        // write to smem
        // for (int i = 0; i < ept / 2; i++) {
        //     s_data[(warp_size/2 + 1) * laneid + i] = reg[i];
        //     s_data[ept* (warp_size/2 + 1) + laneid * (warp_size/2 + 1) + i] = reg[i + ept / 2];
        // }
    }

    for (int i = 0; i < ept; i++) {
        if((laneid % 4) < 2) {
            s_data[laneid / 4 * N  + i / 2 + (i & 1) * 16 + (laneid %2) * 32].x = reg[i];
            // Adaptation: also store the second group of eight transforms.
            s_data[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 + (laneid %2) * 32].x = reg[i + ept];
        } else {
            s_data[laneid / 4 * N  + i / 2 + (i & 1) * 16 + (laneid %2) * 32].y = reg[i];
            // Adaptation: also store the second group of eight transforms.
            s_data[(laneid / 4 + 8) * N + i / 2 + (i & 1) * 16 + (laneid %2) * 32].y = reg[i + ept];
        }
    }

    __syncwarp();

    for (int j = 0; j < batch; j++) {
        for (int i = threadIdx.x; i < N; i += warp_size) {
            d_data[i + N * j + (threadIdx.y + blockIdx.x * blockDim.y) * N * batch] = s_data[i + N * j];
        }
    }
}

// Source: fft/my_fft.cu:769-840.
template <int N>
__device__ void fft_kernel_r64_b16_branch(cuFloatComplex *reg,
                                   const cuFloatComplex *W_4096) {
    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;

    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); // 4^iter;
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            // if(threadIdx.x==0 && blockIdx.x==0) printf("\nidx %d %d\n", i, j);
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

            fill_reg_b_branch<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
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

        if (i < ITER_DEVICE_CONST - 1) {
            for (int j = 0; j < 32; j += 4 * stride) {
                for (int k = 0; k < stride; k++) {
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
            }
        }
    }
}

// Source: fft/my_fft.cu:844-897.
__global__ void
fft_kernel_radix64_batch16_branch(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64, unsigned int repeat) {
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
    for (int i = 0; i < ept / 2; i++) {
        reg[i] = s_data[(laneid / 2) * (warp_size + 1) +
                        reverse_2bit_groups<4>(i) + (ept / 2) * (laneid % 2)];
        reg[i + ept / 2] =
            s_data[(ept / 2) * (warp_size + 1) +
                   (laneid / 2) * (warp_size + 1) + reverse_2bit_groups<4>(i) +
                   (ept / 2) * (laneid % 2)];
    }

    for(unsigned int i=0; i<repeat; i++) {
        fft_kernel_r64_b16_branch<64>(reg, W_64);
    }

    // write to smem
    for (int i = 0; i < ept / 2; i++) {
        s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] = reg[i];
        s_data[(ept / 2) * (warp_size + 1) + (warp_size + 1) * (laneid / 2) +
               16 * (laneid % 2) + i] = reg[i + ept / 2];
    }

    // Adaptation: publish shared output before other lanes read it.
    __syncwarp();

    // write to gmem
    for (int i = 0; i < ept; i++)
        d_data[block_id * N * batch + laneid + i * warp_size] =
            s_data[i * (warp_size + 1) + laneid];
}

} // namespace thff_table7_legacy
