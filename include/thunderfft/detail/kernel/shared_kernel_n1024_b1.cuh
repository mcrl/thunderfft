namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 1024, 1, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W_precompute, void *workspace) {
    int laneid = threadIdx.x & 31;
    int block_id = blockIdx.x;

    int ept = 32; // N * batch / threads_per_warp

    vec2_t<float>* smem = (vec2_t<float>*)workspace;

    thunderfft::unit::fft_kernel_r64_b16<true>((float*) reg);

    using L_in = layout_t<64, 16, 1, 64, 16, 1, false>;
    ThunderFFT_reg2smem_N64<float, L_in>(smem, reg);

    __syncthreads();

    for(int i=0; i<ept/2; i++) {
        int row = (i%4) * 4 + (laneid%4);
        int rev_row = i%4 + (laneid%4)*4;

        
        int col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
        int col1 = col0 + 4;

        int index0 = row * 64 + col0; index0 += index0/L_in::pad_period * L_in::pad;    
        int index1 = row * 64 + col1; index1 += index1/L_in::pad_period * L_in::pad;
        

        reg[i] = smem[index0];
        reg[i+ept/2] = smem[index1];
        
        rotate(reg[i], col0 * rev_row, 1024);
        rotate(reg[i+ept/2], col1 * rev_row, 1024);
    }

    thunderfft::unit::fft_kernel_r16_b64<true>((float*) (reg));
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 1024, 1, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W_reg, void *workspace) {
    int laneid = threadIdx.x & 31;
    int block_id = blockIdx.x;

    int ept = 32; // N * batch / threads_per_warp

    vec2_t<float>* smem = (vec2_t<float>*)workspace;

    thunderfft::unit::fft_kernel_r64_b16<false>((float*) reg);

    using L_in = layout_t<64, 16, 1, 64, 16, 1, false>;
    ThunderFFT_reg2smem_N64<float, L_in>(smem, reg);

    __syncthreads();

    for(int i=0; i<ept/2; i++) {
        int row = i%4 * 4 + (laneid%4);
        int rev_row = reverse_bit_groups<2,4>(row);

        int col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
        int col1 = col0 + 4;

        int index0 = row * 64 + col0; index0 += index0/L_in::pad_period * L_in::pad;
        int index1 = row * 64 + col1; index1 += index1/L_in::pad_period * L_in::pad;

        reg[i] = smem[index0];
        reg[i+ept/2] = smem[index1];
        
        rotate(reg[i], -col0 * rev_row, 1024);
        rotate(reg[i+ept/2], -col1 * rev_row, 1024);
    }

    thunderfft::unit::fft_kernel_r16_b64<false>((float*) (reg));
}


template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 1024, 1, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W_reg, void *workspace) {
    int laneid = threadIdx.x;
    int ept = 32; // N * batch / threads_per_warp

    vec2_t<half>* smem = (vec2_t<half>*)workspace;

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_reg);

    using L_in = layout_t<64, 16, 1, 64, 32, 1, false>;
    ThunderFFT_reg2smem_N64<half, L_in>(smem, reg);

    __syncthreads();

    for(int i=0; i<ept/2; i++) {
        int row = i%4 * 4 + (laneid%4);
        int rev_row = i%4 + (laneid%4)*4;
        // int col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
        // int col1 = col0 + 4;
        
        int col0 = (laneid/4) %2 + (laneid/8) * 8 + (i/4) * 2;
        int col1 = col0 + 32;

        int index0 = row * 64 + col0; index0 += index0/L_in::pad_period * L_in::pad;
        int index1 = row * 64 + col1; index1 += index1/L_in::pad_period * L_in::pad;

        reg[i] = smem[index0];
        reg[i+ept/2] = smem[index1];
        rotate(reg[i], col0 * rev_row, 1024);
        rotate(reg[i+ept/2], col1 * rev_row, 1024);

        
        // __syncthreads();
        // if(blockIdx.x==0) {
        //     if(i==0) {
        //         printf("%d %d %d\n", laneid, index0, index0%32);
        //     }
        // }
        // __syncthreads();
    }

    thunderfft::unit_fp16::fft_kernel_r16_b64<true>(reg, W_reg);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 1024, 1, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W_reg, void *workspace) {
    int laneid = threadIdx.x & 31;
    int block_id = blockIdx.x;

    int ept = 32; // N * batch / threads_per_warp

    vec2_t<half>* smem = (vec2_t<half>*)workspace;

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_reg);

    using L_in = layout_t<64, 16, 1, 64, 32, 1, false>;
    ThunderFFT_reg2smem_N64<half, L_in>(smem, reg);

    __syncthreads();

    for(int i=0; i<ept/2; i++) {
        int row = i%4 * 4 + (laneid%4);
        int rev_row = reverse_bit_groups<2,4>(row);

        int col0 = (laneid/4) %2 + (laneid/8) * 8 + (i/4) * 2;
        int col1 = col0 + 32;

        int index0 = row * 64 + col0; index0 += index0/L_in::pad_period * L_in::pad;
        int index1 = row * 64 + col1; index1 += index1/L_in::pad_period * L_in::pad;

        reg[i] = smem[index0];
        reg[i+ept/2] = smem[index1];
        
        rotate(reg[i], -col0 * rev_row, 1024);
        rotate(reg[i+ept/2], -col1 * rev_row, 1024);
    }

    thunderfft::unit_fp16::fft_kernel_r16_b64<false>(reg, W_reg);
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N1024(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    int laneid = threadIdx.x & 31;
    int ept = 32;

    auto *s_0 = (smem+(laneid/4)*(64+64/sL::pad_period*sL::pad)*sL::elem_stride);
    auto *s_1 = (smem+(laneid/4+8)*(64+ 64/sL::pad_period*sL::pad)*sL::elem_stride);

    if constexpr (sL::reversed) {
        for (int i = 0; i < ept / 2; i++) { 
            reg[i] =
                s_0[(i * 4 + (laneid % 4) ) * sL::elem_stride];
            reg[i + ept / 2] =
                s_1[(i * 4 + (laneid % 4) ) * sL::elem_stride];
        }
    } else {
        for (int i = 0; i < ept; i++) {
            int row = (laneid %4)*4 + i%4;
            int col = (laneid/4)%4*4+ (laneid/16) + (i/4)%4*16 + (i/16)*2;
            int idx = row*64+col;
            idx *= sL::elem_stride;
            idx += idx/sL::pad_period*sL::pad;
            reg[i] = smem[idx];
        }
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
ThunderFFT_reg2smem_N1024(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {          
    int laneid = threadIdx.x & 31;
    int ept = 32; // N * batch / threads_per_warp
    
    // auto *s_0 = (smem+(laneid/4)*(64+64/sL::pad_period*sL::pad)*sL::elem_stride);
    // auto *s_1 = (smem+(laneid/4+8)*(64+ 64/sL::pad_period*sL::pad)*sL::elem_stride);
    
    // for (int i = 0; i < ept / 2; i++) {
    //     s_0[(i + (laneid % 4) * (17))*sL::elem_stride] = reg[i];
    //     s_1[(i + (laneid % 4) * (17))*sL::elem_stride] = reg[i + ept/2];
    // }

    for(int i=0; i<ept/2; i++) {
        int row = i%4 + (laneid%4)*4;
        
        
        // int col0 = (laneid/4) %4 + (laneid/16)*16 + (i/4) * 4;
        // int col1 = col0 + 32;

        // int col0, col1;
        // if constexpr(std::is_same_v<T, float>) {
        //     col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
        //     col1 = col0 + 4;
        // } else {
        //     col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
        //     col1 = col0 + 4;
        // }

        int col0, col1;
        if constexpr (std::is_same_v<T, float>) {
             col0 = (laneid/4) %4 + (laneid/16)*8 + (i/4) * 16;
             col1 = col0 + 4;
        } else {
            col0 = (laneid/4) %2 + (laneid/8) * 8 + (i/4) * 2;
            col1 = col0 + 32;
        }
        
        // int col0 = ((laneid/4) %4)*16 + (laneid/16)*2 + (i/4) * 4;
        // int col1 = col0 + 1;

        // int col0 = (laneid/4) %2 + (laneid/8)*8+ (i/4) * 2;
        // int col1 = col0 + 32;

        int idx0 = (row*64+col0)*sL::elem_stride;
        int idx1 = (row*64+col1)*sL::elem_stride;
        idx0+=idx0/sL::pad_period*sL::pad;
        idx1+=idx1/sL::pad_period*sL::pad;

        smem[idx0] = reg[i];
        smem[idx1] = reg[i+ept/2];
        
        // __syncthreads();
        // if(blockIdx.x==0) {
        //     if(i==0) {
        //         printf("%d %d %d %d %d\n", laneid, idx0, idx0%32, sL::pad_period, sL::pad);
        //     }
        // }
    }

    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.x>=28 && threadIdx.x<32) {
    //     for(int i=0; i<ept; i++) {
    //         printf("%d %d : %f %f\n", threadIdx.x, i, reg[i].x, reg[i].y);
    //     }
    
    // }
    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.x==0) {
    //     printf("---------------\n");
    // }
    // __syncthreads();    


    // __syncthreads();
    // if(threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<1024; i++) {
    //         int idx = i+i/16;
    //         printf("%f %f\n", smem[idx].x, smem[idx].y);
    //     }
    //     printf("---------------\n");
    // }

    // __syncthreads();
}
}
