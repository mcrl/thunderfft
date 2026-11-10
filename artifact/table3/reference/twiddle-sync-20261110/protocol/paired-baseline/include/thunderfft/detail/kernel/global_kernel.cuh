namespace thunderfft::unit_fp16 {
template <int N, bool forward>
__device__ __forceinline__ void make_reg_b_precompute(vec2_t<half>* W);
}

namespace thunderfft::detail {
// NOTE:
// - This kernel is a thin wrapper that ties together:
//   (1) gmem -> registers
//   (2) register-resident FFT
//   (3) registers -> gmem
//
// Assumptions:
// - ThunderFFT_gmem2reg / ThunderFFT_reg2gmem expect block-local base pointers
//   and map only threadIdx to the logical FFT fragment.
// - ThunderFFT_kernel_reg performs the in-register FFT on the per-thread reg buffer.
//
// If your implementation expects separate input/output pointers inside gmem2reg/reg2gmem,
// adjust the calls accordingly.

template <typename T, int N, bool forward>
__global__ void ThunderFFT_global_kernel(const vec2_t<T>* __restrict__ d_input,
                                          vec2_t<T>* __restrict__ d_output,
                                          int batch) {

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

    // The radix-2 block kernels for 512/2048/4096 consume natural-order input.
    using L_in = std::conditional_t<
        (N == 512 || N == 2048 || N == 4096),
        layout_t<N, BPB, 1, N, 16, 1, false>,
        layout_t<N, BPB, 1, N, 64, 4, true>>;
    using L_out = layout_t<N, BPB, 1, N, 16, 1, false>;

    const size_t first_batch = size_t(blockIdx.x) * BPB;
    const auto* input = d_input + first_batch * N;
    auto* output = d_output + first_batch * N;
    const int valid = min(BPB, batch - int(first_batch));

    if (valid == BPB) {
        ThunderFFT_gmem2smem<T, L_in>(s_in, input);
    } else {
        // Every lane still participates in the FFT and its block barriers.
        // Pad inactive transforms with zero, without reading past the caller's input.
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        for (int b = 0; b < BPB; ++b) {
            for (int j = tid; j < N; j += threads_per_warp * WPB) {
                int index = b * L_in::batch_stride + j;
                index += index / L_in::pad_period * L_in::pad;
                vec2_t<T> value{};
                if (b < valid) {
                    int n = j;
                    if constexpr (L_in::reversed)
                        n = reverse_bit_groups<2, LOG2P_builtin<N>>(j);
                    value = input[b * N + n];
                }
                s_in[index] = value;
            }
        }
    }
    __syncthreads();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    __syncthreads();

    if constexpr (std::is_same_v<T, half>) {
        ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, W, s_in);
    } else {
        ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, nullptr, s_in);
    }
    __syncthreads();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    __syncthreads();

    if (valid == BPB) {
        ThunderFFT_smem2gmem<T, L_out>(output, s_in);
    } else {
        const int tid = threadIdx.x + threadIdx.y * blockDim.x;
        for (int b = 0; b < valid; ++b) {
            for (int j = tid; j < N; j += threads_per_warp * WPB) {
                int index = b * L_out::batch_stride + j;
                index += index / L_out::pad_period * L_out::pad;
                output[b * N + j] = s_in[index];
            }
        }
    }

}

} // namespace thunderfft::detail
