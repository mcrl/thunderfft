namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 64, 16, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    // float2 W[28];
    // unit::make_reg_b<true>(W);

    // unit::fft_kernel_r64_b16_precompute<true>((float*)reg, W);
    unit::fft_kernel_r64_b16<true>((float*)reg);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 64, 16, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    unit::fft_kernel_r64_b16<false>((float*)reg);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 64, 16, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    unit_fp16::fft_kernel_r64_b16<true>(reg, W);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 64, 16, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    unit_fp16::fft_kernel_r64_b16<false>(reg, W);
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N64(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    constexpr int N = sL::N; // 64
    constexpr int ept = (N * sL::batch) / threads_per_warp; // 32
    int laneid = threadIdx.x % threads_per_warp;
    


    int b0 = (laneid/4)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad);
    int b1= (laneid/4+8)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad);



    for (int i = 0; i < ept / 2; i++) { 
        int idx = i * 4 + (laneid % 4);

        if constexpr( !sL::reversed ) {
            idx = reverse_bit_groups<2,6>(idx);
        }

        idx = idx * sL::elem_stride;
        idx = idx + idx/sL::pad_period * sL::pad;
        
        reg[i] = smem[b0 + idx];
        reg[i + ept / 2] = smem[b1+ idx];
    }
}


// --------------------------------------------
// Register file -> shared memory (layout-aware)
// --------------------------------------------
// Stores per-thread registers into shared memory using layout sL.
// - smem : destination shared memory buffer
// - reg  : source register buffer (read-only)
// - sL   : compile-time layout describing (N, batch, block, pad) and indexing
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N64(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    constexpr int N = sL::N; // 64
    constexpr int ept = (N * sL::batch) / threads_per_warp; // 32
    int laneid = threadIdx.x % threads_per_warp;
    //assert sL::reversed ==false; output must not be reversed

    auto *s_0 = (smem+(laneid/4)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));
    auto *s_1 = (smem+(laneid/4+8)*(sL::batch_stride+sL::batch_stride/sL::pad_period * sL::pad));

    for (int i = 0; i < ept / 2; i++) { 
        int idx = (i + (laneid % 4)*16 ) * sL::elem_stride;
        idx = idx + idx/sL::pad_period * sL::pad;
        s_0[idx] = reg[i];
        s_1[idx] = reg[i + ept / 2];
    }
}
}