namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 128, 8, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* _W, void *workspace) {
    // float2 W[28];
    // unit::make_reg_b<true>(W);

    // unit::fft_kernel_r64_b16_precompute<true>((float*)reg, W);
    unit::fft_kernel_r64_b16<true>((float*)reg);

    int laneid = threadIdx.x % threads_per_warp;
    int ept = 32;

    for (int i = 0; i < ept / 2; i++) {
        int idx = i + (laneid % 4) * 16;
        rotate(reg[i + ept / 2], idx, 128);

        auto t1 = reg[i] + reg[i + ept / 2];
        auto t2 = reg[i] - reg[i + ept / 2];

        reg[i] = t1;
        reg[i + ept / 2] = t2;
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 128, 8, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* _W, void *workspace) {
    unit::fft_kernel_r64_b16<false>((float*)reg);

    int laneid = threadIdx.x % threads_per_warp;
    int ept = 32;

    for (int i = 0; i < ept / 2; i++) {
        int idx = i + (laneid % 4) * 16;
        rotate(reg[i + ept / 2], -idx, 128);

        auto t1 = reg[i] + reg[i + ept / 2];
        auto t2 = reg[i] - reg[i + ept / 2];

        reg[i] = t1;
        reg[i + ept / 2] = t2;
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 128, 8, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    int laneid = threadIdx.x % threads_per_warp;

    unit_fp16::fft_kernel_r64_b16<true>(reg, W);
    
    int ept = 32;

    for (int i = 0; i < ept / 2; i++) { 
        int idx = i + (laneid % 4)*16 ;
        rotate(reg[i + ept / 2], idx, 128);


        auto t1 = reg[i] + reg[i + ept / 2];
        auto t2 = reg[i] - reg[i + ept / 2];

        reg[i] = t1;
        reg[i + ept / 2] = t2;
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 128, 8, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    int laneid = threadIdx.x % threads_per_warp;

    unit_fp16::fft_kernel_r64_b16<false>(reg, W);
    
    int ept = 32;

    for (int i = 0; i < ept / 2; i++) { 
        int idx = i + (laneid % 4)*16 ;
        rotate(reg[i + ept / 2], -idx, 128);


        auto t1 = reg[i] + reg[i + ept / 2];
        auto t2 = reg[i] - reg[i + ept / 2];

        reg[i] = t1;
        reg[i + ept / 2] = t2;
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N128(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    constexpr int ept = 32;
    int laneid = threadIdx.x % threads_per_warp;
    


    int batch = (laneid/4*sL::batch_stride); batch+=batch/sL::pad_period * sL::pad;
    // int b1= (laneid/4*sL::batch_stride+64*sL::elem_stride); b1+=b1/sL::pad_period * sL::pad;


    for (int i = 0; i < ept / 2; i++) { 
        int idx_0 = i * 4 + (laneid % 4);
        int idx_1 = idx_0 + 64;

        if constexpr( !sL::reversed ) {
            idx_0 = (i/4)*2 + (i%4)*8 + (laneid % 4) * 32;
            idx_1 = idx_0+1;
        }

        idx_0 = idx_0 * sL::elem_stride;
        idx_0 = idx_0 + idx_0/sL::pad_period * sL::pad;
        idx_1 = idx_1 * sL::elem_stride;
        idx_1 = idx_1 + idx_1/sL::pad_period * sL::pad;
        
        reg[i] = smem[batch + idx_0];
        reg[i + ept / 2] = smem[batch + idx_1];
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
ThunderFFT_reg2smem_N128(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    constexpr int ept = 32;
    int laneid = threadIdx.x % threads_per_warp;
    


    int batch = (laneid/4*sL::batch_stride); batch+=batch/sL::pad_period * sL::pad;
    // int b1= (laneid/4*sL::batch_stride+64*sL::elem_stride); b1+=b1/sL::pad_period * sL::pad;



    for (int i = 0; i < ept / 2; i++) { 
        int idx_0 = i + (laneid % 4)*16 ;
        int idx_1 = idx_0 + 64;

        idx_0 = idx_0 * sL::elem_stride;
        idx_0 = idx_0 + idx_0/sL::pad_period * sL::pad;
        idx_1 = idx_1 * sL::elem_stride;
        idx_1 = idx_1 + idx_1/sL::pad_period * sL::pad;
        
        smem[batch + idx_0] = reg[i];
        smem[batch + idx_1] = reg[i + ept / 2];
    }
}
}
