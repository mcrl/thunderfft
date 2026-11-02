#include <mma.h>

namespace thunderfft {

#ifndef THUNDERFFT_RADIX8_IMPL
#define THUNDERFFT_RADIX8_IMPL 1
#endif

static __device__ __forceinline__ void radix8_mma_m16n8k8_tf32_f32_rowcol(
    float d[4], const float a[4], const float b[2], const float c[4]) {
    auto a0 = __float_as_uint(a[0]);
    auto a1 = __float_as_uint(a[1]);
    auto a2 = __float_as_uint(a[2]);
    auto a3 = __float_as_uint(a[3]);
    auto b0 = __float_as_uint(b[0]);
    auto b1 = __float_as_uint(b[1]);
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0, %1, %2, %3}, "
                 "{%4, %5, %6, %7}, "
                 "{%8, %9}, "
                 "{%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void radix8_mma_m16n8k8_fp16_fp16_rowcol(
    unsigned int d[2], const unsigned int a[2], const unsigned int b[1],
    const unsigned int c[2]) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, "
                 "{%2, %3}, "
                 "{%4}, "
                 "{%5, %6};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(c[0]), "r"(c[1]));
}

static __device__ __forceinline__ void fill_reg_b_n512(float b[2], int laneid) {
    const int row = laneid & 3;
    const int col = laneid / 8;

    int pattern = (row*col+((laneid/4) % 2)) % 4;

    float sign = pattern/2? -1.0f : 1.0f;
    if(pattern%2) {
        b[0] = 0;
        b[1] = sign;
    } else {
        b[0] = sign;
        b[1] = 0;
    }
}

static __device__ __forceinline__ void fill_reg_b_n512(half2 b[1], int laneid) {
    float bf[2];
    fill_reg_b_n512(bf, laneid);
    b[0] = __floats2half2_rn(bf[0], bf[1]);
}

__device__ __forceinline__ void radix8_dft_forward_m16n16k16(float2* v) {
    namespace wmma = ::nvcuda::wmma;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    const half ax = __float2half(v[0].x);
    const half ay = __float2half(v[0].y);
    const half bx = __float2half(v[1].x);
    const half by = __float2half(v[1].y);

    #pragma unroll
    for (int i = 0; i < a_frag.num_elements; ++i) {
        a_frag.x[i] = (i & 3) == 0 ? ax : ((i & 3) == 1 ? ay : ((i & 3) == 2 ? bx : by));
    }
    #pragma unroll
    for (int i = 0; i < b_frag.num_elements; ++i) {
        b_frag.x[i] = (i & 1) ? __float2half(0.5f) : __float2half(-1.0f);
    }

    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

    v[0].x = c_frag.x[0];
    v[0].y = c_frag.x[1];
    v[1].x = c_frag.x[2];
    v[1].y = c_frag.x[3];
}

__device__ __forceinline__ void radix8_dft_forward(float2* v, float* b) {
    const int laneid = threadIdx.x & 31;
    float a[4] = {v[0].x, v[1].x, v[0].y, v[1].y};
    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float2 d2[2]  = {{0.0f, 0.0f}, {0.0f, 0.0f}};
    radix8_mma_m16n8k8_tf32_f32_rowcol(reinterpret_cast<float*>(d2), a, b, c);

    rotate(d2[1], laneid%4, 8);
    v[0] = d2[0]+d2[1];
    v[1] = d2[0]-d2[1];
}

__device__ __forceinline__ void radix8_dft_forward(float2* v) {
    float b[2];
    fill_reg_b_n512(b, threadIdx.x & 31);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        radix8_dft_forward(v + i * 2, b);
    }
}

__device__ __forceinline__ void radix8_dft_forward(half2* v, float* b) {
    float2 tmp[2];
    tmp[0] = __half22float2(v[0]);
    tmp[1] = __half22float2(v[1]);
    radix8_dft_forward(tmp, b);
    v[0] = __floats2half2_rn(tmp[0].x, tmp[0].y);
    v[1] = __floats2half2_rn(tmp[1].x, tmp[1].y);
}

__device__ __forceinline__ void radix8_dft_forward(half2* v, half2* b) {
    const int laneid = threadIdx.x & 31;
    half2 a2[2] = {v[0], v[1]};
    half2 d2[2] = {make_half2(0.0f, 0.0f), make_half2(0.0f, 0.0f)};
    half2 c2[2] = {make_half2(0.0f, 0.0f), make_half2(0.0f, 0.0f)};

    radix8_mma_m16n8k8_fp16_fp16_rowcol(reinterpret_cast<unsigned int*>(d2),
                                        reinterpret_cast<unsigned int*>(a2),
                                        reinterpret_cast<unsigned int*>(b),
                                        reinterpret_cast<unsigned int*>(c2));

    rotate(d2[1], laneid % 4, 8);
    v[0] = d2[0] + d2[1];
    v[1] = d2[0] - d2[1];
}

__device__ __forceinline__ void radix8_dft_forward(half2* v) {
    float b[2];
    fill_reg_b_n512(b, threadIdx.x & 31);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        radix8_dft_forward(v + i * 2, b);
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 512, 2, true>(vec2_t<float>* __restrict__ reg,
                                           vec2_t<float>* W_precompute,
                                           void* workspace) {
    (void)W_precompute;
    int laneid = threadIdx.x;
    int ept = 32;

    vec2_t<float>* smem = (vec2_t<float>*)workspace;

    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    using L_in = layout_t<64, 16, 1, 64, 16, 1, false>;
    {
        int row = laneid/4;
        row = row/2 + (row%2)*4;
        auto *s_0 = smem + row *
            (L_in::batch_stride + L_in::batch_stride / L_in::pad_period * L_in::pad);
        auto *s_1 = smem + (row+ 8) *
            (L_in::batch_stride + L_in::batch_stride / L_in::pad_period * L_in::pad);

        #pragma unroll
        for (int i = 0; i < ept / 2; ++i) {
            int idx = (i + (laneid % 4) * 16) * L_in::elem_stride;
            idx += idx / L_in::pad_period * L_in::pad;
            s_0[idx] = reg[i];
            s_1[idx] = reg[i + ept / 2];
        }
    }
    __syncthreads();

    for (int i = 0; i < ept; ++i) {
        int row_swap = (laneid % 4) + (i % 2) * 4;
        int row = (laneid % 4)*2 + (i % 2);
        int col = (laneid / 4) * 8 + (i / 2) % 8 ;
        // int col = (i/2) % 4 + (laneid / 4) * 4 + ((i/8) % 2) * 32;

        int idx = (row_swap + (i/16)*8) * 64 + col;
        idx = idx * L_in::elem_stride;
        idx = idx + idx / L_in::pad_period * L_in::pad;

        reg[i] = smem[idx];
        rotate(reg[i], row * col, 512);
    }

    
    float b[2];
    fill_reg_b_n512(b, laneid);

    for(int i=0; i<16; i++) {
        radix8_dft_forward(reg+i*2, b);
    }


}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 512, 2, true>(vec2_t<half>* __restrict__ reg,
                                          vec2_t<half>* W_precompute,
                                          void* workspace) {
    (void)W_precompute;
    int laneid = threadIdx.x;
    int ept = 32;

    vec2_t<half>* smem = (vec2_t<half>*)workspace;
    vec2_t<half> W_stage0[28];
    thunderfft::unit_fp16::make_reg_b_precompute<64, true>(W_stage0);

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_stage0);

    using L_in = layout_t<64, 16, 1, 64, 32, 1, false>;
    {
        int row = laneid/4;
        row = row/2 + (row%2)*4;
        auto *s_0 = smem + row *
            (L_in::batch_stride + L_in::batch_stride / L_in::pad_period * L_in::pad);
        auto *s_1 = smem + (row + 8) *
            (L_in::batch_stride + L_in::batch_stride / L_in::pad_period * L_in::pad);

        #pragma unroll
        for (int i = 0; i < ept / 2; ++i) {
            int idx = (i + (laneid % 4) * 16) * L_in::elem_stride;
            idx += idx / L_in::pad_period * L_in::pad;
            s_0[idx] = reg[i];
            s_1[idx] = reg[i + ept / 2];
        }
    }
    __syncthreads();

    for (int i = 0; i < ept; ++i) {
        int row_swap = (laneid % 4) + (i % 2) * 4;
        int row = (laneid % 4)*2 + (i % 2);
        int col = (laneid / 4) * 8 + (i / 2) % 8 ;

        int idx = (row_swap + (i/16)*8) * 64 + col;
        idx = idx * L_in::elem_stride;
        idx = idx + idx / L_in::pad_period * L_in::pad;

        reg[i] = smem[idx];
        rotate(reg[i], row * col, 512);
    }

    half2 b[1];
    fill_reg_b_n512(b, laneid);

    for(int i=0; i<16; i++) {
        radix8_dft_forward(reg+i*2, b);
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N512(vec2_t<T>* __restrict__ reg,
                         const vec2_t<T>* __restrict__ smem) {
    int laneid = threadIdx.x;
    int ept = 32;


    if constexpr (sL::reversed) {
        auto *s_0 = (smem+(laneid/4)*(64+64/sL::pad_period*sL::pad)*sL::elem_stride);
        auto *s_1 = (smem+(laneid/4+8)*(64+ 64/sL::pad_period*sL::pad)*sL::elem_stride);

        for (int i = 0; i < ept / 2; i++) { 
            reg[i] =
                s_0[(i * 4 + (laneid % 4) ) * sL::elem_stride];
            reg[i + ept / 2] =
                s_1[(i * 4 + (laneid % 4) ) * sL::elem_stride];
        }
    } else {
        for (int i = 0; i < ept; i++) {
            int idx = (laneid/4) + ((i/4)%4+4*(i%4)+16*(laneid%4))*8 + (i/16)*512;
            idx *= sL::elem_stride;
            idx += idx/sL::pad_period*sL::pad;
            reg[i] = smem[idx];
        }
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N512(vec2_t<T>* __restrict__ smem,
                         const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = (sL::N * sL::batch) / (threads_per_warp * warp_per_block<sL::N>);
    const int laneid = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < ept; ++i) {        
        int row = (laneid % 4) + (i % 2)*4 + (i/16)*8;
        int col = (laneid / 4) * 8 + (i / 2) % 8 ;

        int logical_idx = 64*row+col;
        int idx = logical_idx * sL::elem_stride;
        idx += idx / sL::pad_period * sL::pad;
        smem[idx] = reg[i];
    }

    //      __syncthreads();
    // if(blockIdx.x==0) {
    //     for(int i=0;i<32;i++) {
    //         if(laneid==i) {
    //             printf("laneid %d : ", laneid);
    //             for(int j=0;j<ept;j++) {
    //                 printf("(%f, %f) ", reg[j].x, reg[j].y);
    //             }
    //             printf("\n");
    //         }
    //         __syncthreads();
    //     }
    //     if(threadIdx.x==0) {
    //         for(int i=0;i<16;i++) {
    //             for(int j=0;j<64;j++) {
    //                 int idx = i*64+j;
    //                 idx+=idx/16;
    //                 printf("(%f, %f) ", smem[idx].x, smem[idx].y);
    //             }
    //             printf("\n");
    //         }
    //         printf("---------------------------\n");
    //     }
    // }
    // __syncthreads();
}

} // namespace thunderfft
