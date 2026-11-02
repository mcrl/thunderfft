#pragma once
#include <cuda_runtime.h>
#include <cuComplex.h>

#include <thunderfft/thunderfft.cuh>           // vec2_t<T>, preprocess_W<T>(N)
#include <thunderfft/detail/utils.h>


#include "thunderfft_bench_half.h"
// #include "../util/helper.h"   // CHECK_CUDA
// #include "../util/stat.h"

template <typename T, unsigned N, unsigned batch_per_block>
__global__ void ThunderFFT_kernel_ir(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats);


template <>
__global__ void ThunderFFT_kernel_ir<float,64,16>(
    vec2_t<float>*       d_input,
    vec2_t<float>*       d_output,
    const float*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 64;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size; // element_per_thread
    constexpr int total_ept = N * batch_per_block;

    // int threadid = threadIdx.x * blockDim.y + threadIdx.y;
    int laneid =  threadIdx.x; //% warp_size;
    int blockid = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<float>* s_in = reinterpret_cast<vec2_t<float>*>(_smem);
    auto s_out = s_in + (N + pad(N)) * batch_per_block; 

    // gmem -> smem (input)
    /* Original load */
    // for (unsigned i = 0; i < batch_per_block; i++) {
    //     for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
    //         s_in[i * (N+4)+ j ] = d_input[blockid * N * batch_per_block + i * N + reverse_bit_groups<2,6>(j)];
    //         // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
    //     }
    // }
    /* Vectorized (float2) load */
    // for (int pair = tid; pair < (total_ept / 2); pair += block_size) {
    //     const int elem0    = pair * 2;
    //     const int batch_id = elem0 / N;
    //     const int j0       = elem0 % N;
    //     const int j1       = j0 + 1;

    //     const int rev_j0   = reverse_bit_groups<2,6>(j0);
    //     const int rev_j1   = reverse_bit_groups<2,6>(j1);

    //     const int base_g   = blockid * (N * batch_per_block) + batch_id * N;
    //     const float2 a     = reinterpret_cast<const float2*>(d_input)[ base_g + rev_j0 ];
    //     const float2 b     = reinterpret_cast<const float2*>(d_input)[ base_g + rev_j1 ];

    //     const int srow     = batch_id * (N + pad(N));
    //     reinterpret_cast<float2*>(s_in)[srow + j0] = a;
    //     reinterpret_cast<float2*>(s_in)[srow + j1] = b;
    // }
    /* Vectorized (float4) load */
    for (int pair = tid; pair < (total_ept / 2); pair += block_size) {
        const int elem0    = pair * 2;  
        const int batch_id = elem0 / N;
        const int j0       = elem0 % N;         
        const int j1       = j0 + 1;
        
        const int rev_j0   = reverse_bit_groups<2,6>(j0);
        const int rev_j1   = reverse_bit_groups<2,6>(j1);

        const int base_g   = blockid * (N * batch_per_block) + batch_id * N;
        const float2 a     = reinterpret_cast<const float2*>(d_input)[ base_g + rev_j0 ];
        const float2 b     = reinterpret_cast<const float2*>(d_input)[ base_g + rev_j1 ];

        const float4 v     = make_float4(a.x, a.y, b.x, b.y);
        const int srow     = batch_id * (N + pad(N));
        reinterpret_cast<float4*>(s_in)[srow + (j0 / 2)] = v;
    }
    __syncthreads();

    // repeat in shared memory
    // for (unsigned r = 0; r < inside_repeats; ++r) {
        float reg[ept * 2];
        vec2_t<float>* reg2 = (vec2_t<float>*)reg;

        int b =  ((laneid>>1) & 1);
        int pad_r = ((laneid/4)&1);

        auto *i_0 = (s_in+(laneid/4)*(N+4) );
        auto *i_1 = (s_in+(laneid/4+8)*(N+4));

        thunderfft::detail::unit::smem2reg(reg2, i_0, i_1, 1);

        // for (unsigned r = 0; r < inside_repeats; ++r) {
            thunderfft::detail::unit::fft_kernel_r64_b16<true>(reg, dW);
        // }

        auto *o_0 = (s_out+(laneid/4)*(N+4));
        auto *o_1 = (s_out+(laneid/4+8)*(N+4));

        thunderfft::detail::unit::reg2smem(reg2, o_0, o_1, 1);
        __syncthreads();
    // }
    
    // smem -> gmem (output)
    /* Original store */
    // for (unsigned i = 0; i < batch_per_block; i++) {
    //     for(unsigned j=threadIdx.x; j<N; j+= blockDim.x) {
    //         d_output[blockid * N * batch_per_block + i * N + j] = s_out[i * (N+4) + j +j/16];
    //     }
    // }
    /* Vectorized (float2) store */
    // for (int pair = tid; pair < (total_ept / 2); pair += block_size) {
    //     const int elem0    = pair * 2;        
    //     const int batch_id = elem0 / N;
    //     const int j0       = elem0 % N;         
    //     const int j1       = j0 + 1;

    //     const int sbase = batch_id * (N+4);
    //     const float2 a = reinterpret_cast<const float2*>(s_out)[sbase + j0 + j0/16];
    //     const float2 b = reinterpret_cast<const float2*>(s_out)[sbase + j1 + j1/16];

    //     const int dst_base = blockid * total_ept + batch_id * N;
    //     reinterpret_cast<float2*>(d_output)[dst_base + j0] = a;
    //     reinterpret_cast<float2*>(d_output)[dst_base + j1] = b;
    // }
    /* Vectorized (float4) store */
    for (int pair = tid; pair < (total_ept / 2); pair += block_size) {
        const int elem0    = pair * 2;        
        const int batch_id = elem0 / N;
        const int j0       = elem0 % N;         
        const int j1       = j0 + 1;

        const int sbase = batch_id * (N+4);
        const float2 a = reinterpret_cast<const float2*>(s_out)[sbase + j0 + j0/16];
        const float2 b = reinterpret_cast<const float2*>(s_out)[sbase + j1 + j1/16];

        const float4 v = make_float4(a.x, a.y, b.x, b.y);

        const int dst_v = blockid * (total_ept / 2) + batch_id * (N / 2) + (j0 / 2);
        reinterpret_cast<float4*>(d_output)[dst_v] = v;
    }
}

template <>
__global__ void ThunderFFT_kernel_ir<float,256,16>(
    vec2_t<float>*       d_input,
    vec2_t<float>*       d_output,
    const float*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    typedef float T;
    constexpr unsigned N = 256;
    constexpr unsigned batch_per_block = 16;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size; // element_per_thread

    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * warp_size + laneid;
    int blockid = blockIdx.x;


    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<float>* s_in = reinterpret_cast<vec2_t<float>*>(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;

    // gmem -> smem (input)
   for (int i = 0; i < batch_per_block; i++) {
        for(int j=laneid; j<64; j+=32) {
            s_in[ (i * 4+ warpid) * (64+2) + j - (warpid&1) ] = d_input[blockid * N * batch_per_block + i * N + reverse_bit_groups<2,LOG2P_builtin<N>>(64*warpid+j)];
        }
    }

    float reg[ept * 2];
    float2* reg2 = (reinterpret_cast<float2*>(reg));

    int b =  ((laneid>>1) & 1);
    int pad_r = ((laneid/4)&1);

    float *i_0 = (float*)(s_in+(laneid/4 + warpid * 16) * (64+2) - pad_r);
    float *i_1 = (float*)(s_in+(laneid/4+8 +  warpid * 16) * (64+2) - pad_r);

    for (int i = 0; i < ept / 2; i++) {
        reg[2 * i] =
            i_0[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
        reg[2 * i + 1] =
            i_0[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
        reg[2 * i + ept] =
            i_1[(i * 4 + (laneid % 2) * 2    ) * 2 + b];
        reg[2 * i + ept + 1] =
            i_1[(i * 4 + (laneid % 2) * 2 + 1) * 2 + b];
    }

    __syncthreads();
    for (int r = 0; r < inside_repeats; ++r) {
    // repeat in shared memory

        if(true) {


        // for (unsigned r = 0; r < inside_repeats; ++r) {
            thunderfft::detail::unit::fft_kernel_r64_b16<true>(reg, dW);
        // }
            float *o_0 = (float*)(s_out+(laneid/4 + warpid * 16)*(64+2));
            float *o_1 = (float*)(s_out+(laneid/4+8 + warpid * 16)*(64+2));
            for (int i = 0; i < ept; i++) {
                o_0[(i / 2 + (i & 1) * 16 + (laneid % 2) * (32+1)) * 2 + b] = reg[i];
                o_1[(i / 2 + (i & 1) * 16 + (laneid % 2) * (32+1)) * 2 + b] = reg[i + ept];
            }
        }
        __syncthreads();
        if(true) {
            for (int i = 0; i < 4; i++) {
                const int batch = warpid * 4 + i;

                for (int j = 0; j < 2; j++) {
                    const int index = laneid + j * 32;   // 0..127 (for j=0..3) when 32-lane warp

                    // keep the original stride pattern: +0, +64, +128, +192 (N must be >= 256)

                    const int stride = (N + pad(N));
                    const int base   = batch * stride;

                    reg2[0 + 4 * (i*2+j)] = s_out[base + (index +   0)];
                    reg2[1 + 4 * (i*2+j)] = s_out[base + (index +  64)];
                    reg2[2 + 4 * (i*2+j)] = s_out[base + (index + 128)];
                    reg2[3 + 4 * (i*2+j)] = s_out[base + (index + 192)];

                    // very rough twiddle usage (same k for all four, just to make it run)
                    const int k = (index % 64);  // your original code used (index % 64)
                    const float2 w = W(k, 256);

                    reg2[0 + 4 * (i*2+j)] = cmul(reg2[0 + 4 * (i*2+j)], w);
                    reg2[1+ 4 * (i*2+j)] = cmul(reg2[1], w);
                    reg2[2+ 4 * (i*2+j)] = cmul(reg2[2], w);
                    reg2[3+ 4 * (i*2+j)] = cmul(reg2[3], w);

                    // radix-4 DIF with middle (-i)
                    float2 a = reg2[0+ 4 * (i*2+j)] + reg2[2+ 4 * (i*2+j)]; // x0 + x2
                    float2 b = reg2[0+ 4 * (i*2+j)] - reg2[2+ 4 * (i*2+j)]; // x0 - x2
                    float2 c = reg2[1+ 4 * (i*2+j)] + reg2[3+ 4 * (i*2+j)]; // x1 + x3
                    float2 d = reg2[1+ 4 * (i*2+j)] - reg2[3+ 4 * (i*2+j)]; // x1 - x3

                    d = make_float2(-d.y, d.x);        // center-stage rotation by -i

                    reg2[0+ 4 * (i*2+j)] = a + c;             // y0
                    reg2[1+ 4 * (i*2+j)] = b + d;             // y1
                    reg2[2+ 4 * (i*2+j)] = a - c;             // y2
                    reg2[3+ 4 * (i*2+j)] = b - d;             // y3

                }
            }
        }
    }

    for (int i = 0; i < 4; i++) {
        const int batch = warpid * 4 + i;

        for (int j = 0; j < 2; j++) {
            const int index = laneid + j * 32;   // 0..127 (for j=0..3) when 32-lane warp

            // keep the original stride pattern: +0, +64, +128, +192 (N must be >= 256)

            const int stride = (N + pad(N));
            const int base   = batch * stride;
            s_out[base + (index +   0)] = reg2[0+ 4 * (i*2+j)];
            s_out[base + (index +  64)] = reg2[1+ 4 * (i*2+j)];
            s_out[base + (index + 128)] = reg2[2+ 4 * (i*2+j)];
            s_out[base + (index + 192)] = reg2[3+ 4 * (i*2+j)];
        }
    }

    for (int i = 0; i < batch_per_block; i++) {
        for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
            d_output[blockid * N * batch_per_block + i * N + j] = s_out[i * (N+pad(N)) + j /*+ (j/32)*/];
        }
    }
}

template <>
__global__ void ThunderFFT_kernel_ir<float,4096,1>(
    vec2_t<float>*       d_input,
    vec2_t<float>*       d_output,
    const float*         __restrict__ dW,
    unsigned                  inside_repeats)  {
        typedef float T;
    constexpr unsigned N = 4096;
    constexpr unsigned batch_per_block = 1;
    constexpr int warp_size = 32;
    constexpr int ept = N * batch_per_block / warp_size / 4; // element_per_thread

    int laneid =  threadIdx.x;
    int warpid = threadIdx.y;
    int threadid = warpid * warp_size + laneid;
    int blockid = blockIdx.x;

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<float>* s_in = reinterpret_cast<vec2_t<float>*>(_smem);
    auto s_out=s_in + (N+pad(N)) * batch_per_block;

    // gmem -> smem (input)
    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        // s_in[j] = d_input[blockid * N * batch_per_block + j];
        s_in[j/64 * (64+4) + j % 64] = d_input[blockid * N * batch_per_block + reverse_bit_groups<2,LOG2P_builtin<N>>(j)];
    }
    

    float reg[ept*2];
    vec2_t<float>* reg2 = (vec2_t<float>*)reg;

    __syncthreads();

    #pragma unroll 1
    for (int r = 0; r < inside_repeats; ++r) {
    thunderfft::detail::unit::smem2reg(reg2,  s_in+((laneid/4) + warpid * 16)*(64+4), s_in+((laneid/4+8) + warpid*16)*(64+4), 1);
    

    
    // #pragma unroll 1
    // for (int r = 0; r < inside_repeats; ++r) {
        
        thunderfft::detail::unit::fft_kernel_r64_b16<true>(reg, dW);

        for(int i=0; i<ept/2; i++) {
            int col = i + (laneid%4)*16;
            // int row1 = (laneid/4) + warpid * 16;
            // int row2 = (laneid/4+8) + warpid * 16;
            int rev_row1 = warpid + laneid/16 * 4 + ((laneid/4) % 4) * 16;
            int rev_row2 = rev_row1+8;
            
            // reg2[i] = cmul(reg2[i], W(col * reverse_bit_groups<2,6>(row1), 4096));
            // reg2[i+ept/2] = cmul(reg2[i+ept/2], W(col * reverse_bit_groups<2,6>(row2), 4096));
            reg2[i] = cmul(reg2[i], W(col * rev_row1, 4096));
            reg2[i+ept/2] = cmul(reg2[i+ept/2], W(col * rev_row2, 4096));
        }

        thunderfft::detail::unit::reg2smem(reg2, s_out+((laneid/4) + warpid * 16)*(64+4), s_out + ((laneid/4+8) + warpid*16)*(64+4), 1);

        __syncthreads();

        // for(int i = 0 ; i < 16 ; i++) {
        //     s_out[(16*warpid+i)*(64+4) + laneid + (laneid/16)] = cmul(s_out[(16*warpid+i)*(64+4) + laneid + (laneid/16)] , W((laneid) * reverse_bit_groups<2,6>(16*warpid+i), 4096));
        //     s_out[(16*warpid+i)*(64+4) + laneid+34 + (laneid/16)] = cmul(s_out[(16*warpid+i)*(64+4) + laneid+34 + (laneid/16)] , W((laneid+32) * reverse_bit_groups<2,6>(16*warpid+i), 4096));
        // }

        // __syncthreads();

        auto s_0 = s_out + (laneid/4) + warpid * 17;
        auto s_1 = s_out + (laneid/4+8) + warpid * 17;
        thunderfft::detail::unit::smem2reg(reg2,  s_0, s_1, 64+4);
        thunderfft::detail::unit::fft_kernel_r64_b16<true>(reg, dW);


        __syncthreads();
    // }

    for (int i = 0; i < ept / 2; i++) {
        int col0 = (laneid/4) + warpid * 16;
        int col1 = (laneid/4+8) + warpid * 16;
        int row=(i + (laneid % 4) * (16));
        s_out[col0+row*64 + row/4]  = reg2[i];
        s_out[col1+row*64 + row/4]  = reg2[i + ept/2];
    }
    }

    // if(threadid==0 && blockIdx.x==0) {
    //     for(int i=0; i<N; i++) {
    //             printf("%f + %fi\n", s_out[i].x, s_out[i].y);
    //     }
    // }
    
    __syncthreads();

    for(int j=threadid; j<N; j+= blockDim.x * blockDim.y) {
        d_output[blockid * N * batch_per_block + j] = s_out[(j/64) * 64 + (j%64) + (j/256)];
    }
}


template <typename T, unsigned int N>
void thunderfft_benchmark(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr unsigned batch_per_block  = (N <= 512u ? 16u : 1u);
    constexpr unsigned threads_per_warp = 32;
    constexpr unsigned warp_per_block =
        (N == 64  || N == 1024) ? 1 :
        (N == 256 || N == 4096) ? 4 :
        -1;

    const dim3 grid ( batch  / batch_per_block );
    const dim3 block( threads_per_warp, warp_per_block );

    const size_t shmem_bytes = 2 * sizeof(T2) * (N+pad_h(N)) * batch_per_block;
    // const size_t shmem_bytes = 2 * sizeof(float2) * (N+pad_h(N)) * batch_per_block;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_kernel_ir<T, N, batch_per_block>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_kernel_ir<T, N, batch_per_block>
            <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, nullptr, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };

    // auto kernel_half = [grid, block, shmem_bytes, dW_half, stream]
    //             (half2* d_data, unsigned int inside_repeats) {
    //     ThunderFFT_kernel_ir<half, N, batch_per_block>
    //         <<<grid, block, shmem_bytes, stream>>>(d_data, d_data, dW_half, inside_repeats);
    //     CHECK_CUDA(cudaGetLastError());
    // };

    benchmark_run<T, N, 4>(kernel, h_input, baseline, batch, "th");
    CHECK_CUDA(cudaStreamDestroy(stream));
}
