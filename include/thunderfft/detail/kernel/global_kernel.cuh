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

    // The radix-2 block kernels for 512/2048/4096 consume natural-order input.
    using L_in = std::conditional_t<
        (N == 512 || N == 2048 || N == 4096),
        layout_t<N, BPB, 1, N, 16, 1, false>,
        layout_t<N, BPB, 1, N, 64, 4, true>>;
    using L_out = layout_t<N, BPB, 1, N, 16, 1, false>;

    ThunderFFT_gmem2smem<T, L_in>(s_in, d_input + blockIdx.x * BPB * N);
    __syncthreads();

    ThunderFFT_smem2reg<T, L_in>(reg, s_in);
    __syncthreads();

    ThunderFFT_kernel_reg<T, N, BPB, forward>(reg, nullptr, s_in);
    __syncthreads();

    ThunderFFT_reg2smem<T, L_out>(s_in, reg);
    __syncthreads();

    ThunderFFT_smem2gmem<T, L_out>(d_output + blockIdx.x * BPB * N, s_in);

}

} // namespace thunderfft::detail
