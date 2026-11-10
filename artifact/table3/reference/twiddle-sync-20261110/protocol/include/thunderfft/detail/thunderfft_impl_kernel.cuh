#include "kernel/global_kernel.cuh"

namespace thunderfft {

    
template <typename T, int N, bool forward>
inline void ThunderFFT_global(vec2_t<T>* d_input,
                            vec2_t<T>* d_output,
                            int batch,
                            cudaStream_t stream) {
  using T2=vec2_t<T>;

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                "ThunderFFT_global supports float and half only");
  static_assert(N == 64 || N == 128 || N == 256 || N == 512 ||
                N == 1024 || N == 2048 || N == 4096,
                "Unsupported ThunderFFT length");
  if (batch < 0) throw std::invalid_argument("ThunderFFT batch must be nonnegative");
  if (batch == 0) return;
  if (!d_input || !d_output || d_input == d_output)
    throw std::invalid_argument("ThunderFFT requires separate non-null input/output buffers");

  constexpr int BPB   = batch_per_block<N>;
  constexpr int WPB   = warp_per_block<N>;

    dim3 grid((batch - 1) / BPB + 1);
    dim3 block(threads_per_warp, WPB);
    constexpr size_t shmem_bytes = (sizeof(T) * 2) * (N + pad_h(N)) * BPB;

    tf_check_cuda(cudaFuncSetAttribute(
        detail::ThunderFFT_global_kernel<T, N, forward>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(shmem_bytes)
    ), "ThunderFFT_global shared-memory attribute");
    
    detail::ThunderFFT_global_kernel<T, N, forward>
            <<<grid, block, shmem_bytes, stream>>>(d_input, d_output, batch);

    tf_check_cuda(cudaGetLastError(), "ThunderFFT_global launch");
    tf_check_cuda(cudaStreamSynchronize(stream), "ThunderFFT_global synchronize");
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
