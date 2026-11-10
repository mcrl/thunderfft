#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
// Scratch-only restoration of bbeffb2, not an unmodified historical measurement.
// See restoration.json. Namespace/variant ID reuse is only for the local harness.
namespace thff_table7_652808b {
static constexpr int TC_M_DEVICE_CONST = 16;
static constexpr int TC_N_DEVICE_CONST = 8;
static constexpr int TC_K_DEVICE_CONST = 8;
static constexpr int RADIX_DEVICE_CONST = TC_K_DEVICE_CONST / 2;
static constexpr int ITER_DEVICE_CONST = 3;
static constexpr int N_DEVICE_CONST = 64;
static constexpr int WARP_SIZE_DEVICE_CONST = 32;
template <int N> __device__ int reverse_2bit_groups(int x) {
    int num_groups = N / 2;
    int result = 0;
    for (int i = 0; i < num_groups; ++i) {
        int group = (x >> (2 * i)) & 0b11;
        result |= group << (2 * (num_groups - 1 - i));
    }
    return result;
}

template <unsigned int N>
__device__ void fill_reg_b(float b[], int stride_log2, int stride, int i_perm,
                           int j_perm, int k,
                           const cuFloatComplex *__restrict__ W_ptr) {
    

    
    
    
    
    
    int i = (threadIdx.x / 8 - i_perm) & 3;
    int j = (threadIdx.x % 4 - j_perm) & 3;

    
    
    
    
    
    
    
    
    
    
    
    
    

    
    constexpr unsigned int N_log2 = (N == 64) ? 4 : 10;
    int index = (1 << (N_log2 - stride_log2)) *
                ((j * (k + stride * i)) & (4 * stride - 1));

    
    
    
    
    

    cuFloatComplex w = W_ptr[index];
    

    
    if ((threadIdx.x / 4) & 1) {
        b[0] = w.y;
        b[1] = w.x;
    } else {
        b[0] = w.x;
        b[1] = -w.y;
    }
    
    
    
    
    
    
    
    
    
}

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
                 "{%0, %1, %2, %3}, "      
                 "{%4, %5, %6, %7}, "      
                 "{%8, %9}, "              
                 "{%10, %11, %12, %13};\n" 
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

template <typename T>
__device__ __forceinline__ void swap_inline(T &x, T &y) {
    T tmp = x;
    x = y;
    y = tmp;
}

// Restored four-pattern complex permutation; scalar real/imag pairs stay together.
template <typename T>
__device__ void permute_radix4_tmp(T &a, T &b, T &c, T &d, T &e, T &f, T &g, T &h, int pattern) {
    if (pattern == 1 || pattern == 3) {
        swap_inline(a,c); swap_inline(b,d);
        swap_inline(e,g); swap_inline(f,h);
    }
    if (pattern == 0 || pattern == 3) {
        swap_inline(c,g); swap_inline(d,h);
    }
    if (pattern == 2 || pattern == 3) {
        swap_inline(a,e); swap_inline(b,f);
    }
}

template <int N>
__device__ void fft_kernel_r64_b16(cuFloatComplex *_reg,
                                   const cuFloatComplex *W_4096) {
    float reg_frag_zero[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                        WARP_SIZE_DEVICE_CONST];
    float *reg = reinterpret_cast<float*>(_reg);

    for (int i = 0;
         i < TC_M_DEVICE_CONST * TC_N_DEVICE_CONST / WARP_SIZE_DEVICE_CONST;
         i++)
        reg_frag_zero[i] = 0.0f;

    int laneid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < ITER_DEVICE_CONST; i++) {
        const int stride = 1 << (i << 1); 
#pragma unroll
        for (int j = 0; j < N_DEVICE_CONST / RADIX_DEVICE_CONST; j++) {
            float reg_frag_a[TC_M_DEVICE_CONST * TC_K_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_b[TC_K_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];
            float reg_frag_d[TC_M_DEVICE_CONST * TC_N_DEVICE_CONST /
                             WARP_SIZE_DEVICE_CONST];

            reg_frag_a[0] = reg[j*2];
            reg_frag_a[1] = reg[j*2 + 2 * N_DEVICE_CONST / RADIX_DEVICE_CONST];
            reg_frag_a[2] = reg[j*2+1];
            reg_frag_a[3] = reg[j*2+1 + 2 * N_DEVICE_CONST / RADIX_DEVICE_CONST];

            
            
            int j_perm;
            if (stride >= 4)
                j_perm = (j / (stride / 4)) % RADIX_DEVICE_CONST;
            else
                j_perm = 0;

            int i_perm = (j / stride) % RADIX_DEVICE_CONST;
            int k = j % stride;

            fill_reg_b<N>(reg_frag_b, i * 2, stride, i_perm, j_perm, k,
                                 W_4096);
            
            
            

            mma_m16n8k8_tf32_f32_rowcol(reg_frag_d, reg_frag_a, reg_frag_b,
                                        reg_frag_zero);

            reg[j*2] = reg_frag_d[0];
            reg[j*2+1] = reg_frag_d[1];
            reg[j*2 + 2 * N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[2];
            reg[j*2+1 + 2 * N_DEVICE_CONST / RADIX_DEVICE_CONST] = reg_frag_d[3];
        }

        if (i < ITER_DEVICE_CONST - 1) {
            for (int jk = 0; jk < 8; jk++) {
                int j = (jk / stride) * (4 * stride);
                int k = jk % stride;
                
                
                
                
                
                
                permute_radix4_tmp(reg[2*(k + j)], reg[2*(k+j)+1],
                                     reg[2*(k + j + stride)], reg[2*(k + j + stride)+1],
                                     reg[2*(k + j + stride * 2)], reg[2*(k + j + stride*2)+1],
                                     reg[2*(k + j + stride * 3)],reg[2*(k + j + stride*3)+1],
                                        laneid & 3);
            }
        }
        
    }
}

__global__ void
fft_kernel_radix64_batch16(cuFloatComplex *d_data,
                           const cuFloatComplex *__restrict__ W_64,
                           unsigned int repeat) {
    
    constexpr int m = 16;
    constexpr int n = 8;
    constexpr int k = 8;

    constexpr int radix = k / 2; 
    constexpr int iter = 3;
    constexpr int N = 64; 
    constexpr int batch = m;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch / warp_size; 

    
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
                        reverse_2bit_groups<4>(i) + (ept / 2) * (laneid %
                        2)];
        reg[i + ept / 2] =
            s_data[(ept / 2) * (warp_size + 1) +
                   (laneid / 2) * (warp_size + 1) + reverse_2bit_groups<4>(i)
                   + (ept / 2) * (laneid % 2)];
    }

    for (unsigned int i = 0; i < repeat; i++) {
        
        
        
        
        
        
        
        
        

        fft_kernel_r64_b16<64>(reg, W_64);

        
        
        
        
        
        
        
        
    }

    
    for (int i = 0; i < ept / 2; i++) {
        s_data[(warp_size + 1) * (laneid / 2) + 16 * (laneid % 2) + i] =
        reg[i]; s_data[(ept / 2) * (warp_size + 1) + (warp_size + 1) *
        (laneid / 2) +
               16 * (laneid % 2) + i] = reg[i + ept / 2];
    }
    __syncwarp();

    
    for (int i = 0; i < ept; i++)
        d_data[block_id * N * batch + laneid + i * warp_size] =
            s_data[i * (warp_size + 1) + laneid];
}

} // namespace thff_table7_652808b
