#include "../../../include/thunderfft/thunderfft.cuh"
#include "../../../include/thunderfft/thunderfft_layout_tuned.h"

#include "helper.h"

namespace thunderfft {
// These benchmark launches contain complete warps and no inactive lanes.
// N=64/128/512 use a single warp, so warp synchronization orders all of their
// shared-memory accesses. Keep block barriers for TF32 N=128 and all N=1024:
// their cached-twiddle paths generate faster kernels with block synchronization.
// Multi-warp transforms always retain their block-wide synchronization.
template<typename T, int N>
__device__ __forceinline__ void benchmark_memory_sync() {
    if constexpr (N == 64 || (N == 128 && std::is_same_v<T, half>) || N == 512)
        __syncwarp(0xffffffffu);
    else
        __syncthreads();
}

template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_reg(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB); 

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];

    using L_fwd_in = typename bench_layout<T, N, BPB>::L_in;
    using L_fwd_out = typename bench_layout<T, N, BPB>::L_out;
    using L_in = std::conditional_t<forward, L_fwd_in, L_fwd_out>;
    using L_out = std::conditional_t<forward, L_fwd_out, L_fwd_in>;

    
    vec2_t<T> W[36];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<N, forward>(W);
    }


    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    benchmark_memory_sync<T, N>();


    // ThunderFFT_gmem2reg<T, N, BPB>(reg, d_input+blockIdx.x * BPB * N);

    #pragma unroll 1
    for(int i=0; i<inside_repeats; i++) {
        ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, (vec2_t<T>*)W, s_in);
        benchmark_memory_sync<T, N>();
    }

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2gmem<T, L_out>(d_output + blockIdx.x * BPB * N, s_in);
    benchmark_memory_sync<T, N>();

    // ThunderFFT_reg2gmem<T, N, BPB>(d_output+blockIdx.x * BPB * N, reg);

}

template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_reg_e2e(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB);

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];

    using L_fwd_in = typename bench_layout<T, N, BPB>::L_in;
    using L_fwd_out = typename bench_layout<T, N, BPB>::L_out;
    using L_in = std::conditional_t<forward, L_fwd_in, L_fwd_out>;
    using L_out = std::conditional_t<forward, L_fwd_out, L_fwd_in>;

    vec2_t<T> W[36];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<N, forward>(W);
    }

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    benchmark_memory_sync<T, N>();


    ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, (vec2_t<T>*)W, s_in);
    benchmark_memory_sync<T, N>();

    
    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2gmem<T, L_out>(d_output + blockIdx.x * BPB * N, s_in);
    benchmark_memory_sync<T, N>();

}

template <typename T, unsigned int N>
void thunderfft_benchmark_reg(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int BPB   = batch_per_block<N>;
    constexpr int WPB   = warp_per_block<N>;

    const dim3 grid ( batch  / BPB );
    const dim3 block( threads_per_warp, WPB );

    const size_t shmem_bytes = sizeof(T2) * (N+pad_h(N)) * BPB;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_benchmark_reg<T, N, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_benchmark_reg<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };
    auto kernel_e2e = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data) {
        ThunderFFT_benchmark_reg_e2e<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr);
        CHECK_CUDA(cudaGetLastError());
    };

    benchmark_run<T, N, 4>(kernel, kernel_e2e, h_input, baseline, batch, "th_r");
    CHECK_CUDA(cudaStreamDestroy(stream));
}

template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_smem(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW,
    unsigned                  inside_repeats)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB); 

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];

    vec2_t<T> W[36];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<N, forward>(W);
    }

    using L_fwd_in = typename bench_layout<T, N, BPB>::L_in;
    using L_fwd_out = typename bench_layout<T, N, BPB>::L_out;
    using L_in = std::conditional_t<forward, L_fwd_in, L_fwd_out>;
    using L_out = std::conditional_t<forward, L_fwd_out, L_fwd_in>;

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    benchmark_memory_sync<T, N>();



    #pragma unroll 1
    for(int i=0; i<inside_repeats; i++) {
    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    benchmark_memory_sync<T, N>();

    ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, W, s_in);
    benchmark_memory_sync<T, N>();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    benchmark_memory_sync<T, N>();
    }

    ThunderFFT_smem2gmem<T, L_out>(d_output + blockIdx.x * BPB * N, s_in);
    benchmark_memory_sync<T, N>();

}

template <typename T, int N, bool forward>
__global__ void ThunderFFT_benchmark_smem_e2e(
    vec2_t<T>*       d_input,
    vec2_t<T>*       d_output,
    const T*         __restrict__ dW)  {
    constexpr int BPB = batch_per_block<N>;
    constexpr int WPB = warp_per_block<N>;
    constexpr int ept = N * BPB / (threads_per_warp * WPB);

    extern __shared__ __align__(16) unsigned char _smem[];
    vec2_t<T>* s_in = reinterpret_cast<vec2_t<T>*>(_smem);

    vec2_t<T> reg[ept];

    vec2_t<T> W[36];
    if constexpr (std::is_same_v<T, half>) {
        unit_fp16::make_reg_b_precompute<N, forward>(W);
    }

    using L_fwd_in = typename bench_layout<T, N, BPB>::L_in;
    using L_fwd_out = typename bench_layout<T, N, BPB>::L_out;
    using L_in = std::conditional_t<forward, L_fwd_in, L_fwd_out>;
    using L_out = std::conditional_t<forward, L_fwd_out, L_fwd_in>;

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    benchmark_memory_sync<T, N>();

    ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, W, s_in);
    benchmark_memory_sync<T, N>();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    benchmark_memory_sync<T, N>();

    ThunderFFT_smem2gmem<T, L_out>(d_output + blockIdx.x * BPB * N, s_in);
    benchmark_memory_sync<T, N>();
}

template <typename T, unsigned int N>
void thunderfft_benchmark_smem(vec2_t<T>* h_input, float2* baseline,
                          int batch)
{
    using T2=vec2_t<T>;

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    constexpr int BPB   = batch_per_block<N>;
    constexpr int WPB   = warp_per_block<N>;

    const dim3 grid ( batch  / BPB );
    const dim3 block( threads_per_warp, WPB );

    const size_t shmem_bytes = sizeof(T2) * (N+pad_h(N)) * BPB;

    T* dW;
    CHECK_CUDA(cudaFuncSetAttribute(
        ThunderFFT_benchmark_smem<T, N, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    auto kernel = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data, unsigned int inside_repeats) {
        ThunderFFT_benchmark_smem<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr, inside_repeats);
        CHECK_CUDA(cudaGetLastError());
    };
    auto kernel_e2e = [grid, block, shmem_bytes, dW, stream]
                (T2* d_data) {
        ThunderFFT_benchmark_smem_e2e<T, N, true>
            <<<grid, block, shmem_bytes, 0>>>(d_data, d_data, nullptr);
        CHECK_CUDA(cudaGetLastError());
    };

    benchmark_run<T, N, 4>(kernel, kernel_e2e, h_input, baseline, batch, "th_s");
    CHECK_CUDA(cudaStreamDestroy(stream));
}
}
