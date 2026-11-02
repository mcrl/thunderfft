namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 2048, 1, true>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;



    
    for (int i = 0; i < ept / 2; ++i) {
        int row = i%4 + (laneid%4) * 4;
        auto tmp0 = reg[i];
        auto tmp1 = reg[i + ept / 2];

        reg[i] = tmp0 + tmp1;
        reg[i+ept/2] = tmp0 - tmp1;

        rotate(reg[i+ept/2], row, 32);
    }

    // __syncthreads();
    // for(int i=0; i<32; i++) {
    //     if(blockIdx.x==0 && threadIdx.x==i && threadIdx.y ==0){
    //         printf("%d : ", i);
    //         for(int j=0; j<32; j++) {
    //             printf("%f, %f; ", reg[j].x, reg[j].y);
    //         }
    //         printf("\n");
    //     }
    //     __syncthreads();
    // }
    // __syncthreads();

    thunderfft::unit::fft_kernel_r16_b64<true>((float*)reg);

    __syncthreads();
    for(int i=0; i<ept/2; i++) {
        int col0 = (i%4)*2 + (laneid%4) * 8;
        int col1 = col0 + 1;
        int row = (laneid / 4) + (i/4) * 8 + warpid * 32;

        int idx0 = row * 32 + col0;
        int idx1 = row * 32 + col1;

        idx0 = idx0 + idx0 / 8 + (row / 16) *4;
        idx1 = idx1 + idx1 / 8 + (row / 16) *4;

        smem[idx0] = reg[i];
        smem[idx1] = reg[i+ept/2];
    }

    
    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.x ==0 && threadIdx.y ==0){
    //     for(int i=0; i<64; i++) {
    //         for(int j=0; j<32; j++) {
    //             int idx = (i*32+j); idx += idx/ 16; 
    //             printf("(%f, %f) ", smem[idx].x, smem[idx].y);
    //         }
    //         printf("\n");
    //     }
    // }
    __syncthreads();


    for(int i=0; i<ept/2; i++) {
        int row = i/4 + (i%4)*4 + (laneid %4 ) * 16;
        int col0 = (laneid / 4) + warpid * 16;
        int col1 = col0+8;

        int idx0 = row * 32 + col0;
        int idx1 = row * 32 + col1;

        idx0 = idx0 + idx0 / 8 + (row / 16) * 4;
        idx1 = idx1 + idx1 / 8 + (row / 16) * 4;

        reg[i] = smem[idx0];
        reg[i+ept/2] = smem[idx1];
        
        rotate(reg[i], col0 * row, 2048);
        rotate(reg[i+ept/2], col1 * row, 2048);
    }


    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    // __syncthreads();
    // for(int i=0; i<32; i++) {
    //     if(blockIdx.x==0 && threadIdx.x==i && threadIdx.y ==0){
    //         printf("%d : ", i);
    //         for(int j=0; j<32; j++) {
    //             printf("%f, %f; ", reg[j].x, reg[j].y);
    //         }
    //         printf("\n");
    //     }
    //     __syncthreads();
    // }
    // __syncthreads();

    // __syncthreads();
    // for(int i=0; i<32; i++) {
    //     if(blockIdx.x==0 && threadIdx.x==i && threadIdx.y ==0){
    //         printf("%d : ", i);
    //         for(int j=0; j<32; j++) {
    //             printf("%f, %f; ", reg[j].x, reg[j].y);
    //         }
    //         printf("\n");
    //     }
    //     __syncthreads();
    // }
    // __syncthreads();

}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 2048, 1, false>(vec2_t<float>* __restrict__ reg,
                                             vec2_t<float>* W_precompute,
                                             void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 2) {
        return;
    }

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit::fft_kernel_r64_b16<false>((float*)reg);

    // TODO(2048-b1): implement the inverse transpose / twiddle stage pair.
    // Keep the control flow parallel to the forward kernel above and mirror the
    // 4096 inverse structure once the 2048 row/col mapping is fixed.

    (void)laneid;
    (void)smem;
    (void)ept;
    (void)W_precompute;
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 2048, 1, true>(vec2_t<half>* __restrict__ reg,
                                            vec2_t<half>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    for (int i = 0; i < ept / 2; ++i) {
        int row = i % 4 + (laneid % 4) * 4;
        auto tmp0 = reg[i];
        auto tmp1 = reg[i + ept / 2];

        reg[i] = tmp0 + tmp1;
        reg[i + ept / 2] = tmp0 - tmp1;

        rotate(reg[i + ept / 2], row, 32);
    }

    thunderfft::unit_fp16::fft_kernel_r16_b64<true>(reg, W_precompute);

    __syncthreads();
    for (int i = 0; i < ept / 2; ++i) {
        int col0 = (i % 4) * 2 + (laneid % 4) * 8;
        int col1 = col0 + 1;
        int row = (laneid / 4) + (i / 4) * 8 + warpid * 32;

        int idx0 = row * 32 + col0;
        int idx1 = row * 32 + col1;

        idx0 = idx0 + idx0 / 8 + (row / 16) * 8;
        idx1 = idx1 + idx1 / 8 + (row / 16) * 8;

        smem[idx0] = reg[i];
        smem[idx1] = reg[i + ept / 2];
    }

    __syncthreads();

    for (int i = 0; i < ept / 2; ++i) {
        int row = i / 4 + (i % 4) * 4 + (laneid % 4) * 16;
        int col0 = (laneid / 4) + warpid * 16;
        int col1 = col0 + 8;

        int idx0 = row * 32 + col0;
        int idx1 = row * 32 + col1;

        idx0 = idx0 + idx0 / 8 + (row / 16) * 8;
        idx1 = idx1 + idx1 / 8 + (row / 16) * 8;

        reg[i] = smem[idx0];
        reg[i + ept / 2] = smem[idx1];

        rotate(reg[i], col0 * row, 2048);
        rotate(reg[i + ept / 2], col1 * row, 2048);
    }

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_precompute);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 2048, 1, false>(vec2_t<half>* __restrict__ reg,
                                             vec2_t<half>* W_precompute,
                                             void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 2) {
        return;
    }

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_precompute);

    // TODO(2048-b1): implement the inverse transpose / twiddle stage pair.
    // Keep the control flow parallel to the forward kernel above and mirror the
    // 4096 inverse structure once the 2048 row/col mapping is fixed.

    (void)laneid;
    (void)smem;
    (void)ept;
    (void)W_precompute;

}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N2048(vec2_t<T>* __restrict__ reg,
                          const vec2_t<T>* __restrict__ smem) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    // TODO(2048-b1): finalize the 2048 b1 smem -> reg mapping.
    // This placeholder keeps the scaffold compile-ready and follows the same
    // row/col helper style as 4096 rather than a block-linear implementation.
    for (int i = 0; i < ept / 2; ++i) {
        int row0 = i%4 + (laneid%4) * 4;
        int row1 = row0 + 16;
        int col = (laneid / 4) + (i/4) * 8 + warpid * 32;

        int idx0 = row0 * 64 + col;
        int idx1 = row1 * 64 + col;
        idx0 = idx0 + idx0 / sL::pad_period * sL::pad;
        idx1 = idx1 + idx1 / sL::pad_period * sL::pad;
        idx0 *= sL::elem_stride;
        idx1 *= sL::elem_stride;

        reg[i] = smem[idx0];
        reg[i+ept/2] = smem[idx1];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N2048(vec2_t<T>* __restrict__ smem,
                          const vec2_t<T>* __restrict__ reg) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    // TODO(2048-b1): finalize the 2048 b1 reg -> smem mapping to match the
    // kernel transpose stage once the row/col convention is fixed.
    for (int i = 0; i < ept / 2; ++i) {
        int row =  i + (laneid %4 ) * 16;
        int col0 = (laneid / 4) + warpid * 16;
        int col1 = col0+8;

        int idx0 = row * 32 + col0;
        int idx1 = row * 32 + col1;
        idx0 = idx0 + idx0 / sL::pad_period * sL::pad;
        idx1 = idx1 + idx1 / sL::pad_period * sL::pad;
        idx0 *= sL::elem_stride;
        idx1 *= sL::elem_stride;

        smem[idx0] = reg[i];
        smem[idx1] = reg[i + ept / 2];
    }

    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.x ==0 && threadIdx.y ==0){
    //     for(int i=0; i<64; i++) {
    //         for(int j=0; j<32; j++) {
    //             int idx = (i*32+j); idx += idx/ 16; 
    //             printf("%f %f, ", smem[idx].x, smem[idx].y);
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();
}

} // namespace thunderfft
