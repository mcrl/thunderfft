#include "kernel/global_kernel.cuh"

namespace thunderfft {

    
template <typename T, int N, bool forward>
inline void ThunderFFT_global(vec2_t<T>* d_input,
                            vec2_t<T>* d_output,
                            int batch,
                            cudaStream_t stream) {
  using T2=vec2_t<T>;

  constexpr int BPB   = batch_per_block<N>;
  constexpr int WPB   = warp_per_block<N>;

    dim3 grid((batch + BPB - 1) / BPB);
    dim3 block(threads_per_warp, WPB);
    constexpr size_t shmem_bytes = (sizeof(T) * 2) * (N + pad_h(N)) * BPB;

    CHECK_CUDA(cudaFuncSetAttribute(
        detail::ThunderFFT_global_kernel<T, N, forward>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ));
    
    detail::ThunderFFT_global_kernel<T, N, forward>
            <<<grid, block, shmem_bytes, stream>>>(d_input, d_output, batch);

    cudaStreamSynchronize(stream);
}
} // namespace thunderfft

#include "kernel/unit_kernel_fp32.cuh"
#include "kernel/unit_kernel_fp16.cuh"

#include "kernel/shared_kernel_n64_b16.cuh"
#include "kernel/shared_kernel_n128_b8.cuh"
#include "kernel/shared_kernel_n256_b16.cuh"
#include "kernel/shared_kernel_n512_b2.cuh"
#include "kernel/shared_kernel_n1024_b1.cuh"
#include "kernel/shared_kernel_n2048_b1.cuh"
#include "kernel/shared_kernel_n4096_b1.cuh"
