namespace thunderfft {
namespace detail {
__device__ __forceinline__ int smem_index_4096(int row, int col) {
    int idx = row * 64 + col;
    idx += idx / 16;
    return idx;
}

#ifdef THUNDERFFT_DEBUG_FP16_4096
__device__ __forceinline__ void debug_print_complex_4096(const char* label,
                                                         int slot,
                                                         vec2_t<half> v) {
    float2 fv = __half22float2(v);
    printf("%s slot=%d (%f, %f)\n", label, slot, fv.x, fv.y);
}

__device__ __forceinline__ void debug_print_complex_4096(const char* label,
                                                         int slot,
                                                         vec2_t<float> v) {
    printf("%s slot=%d (%f, %f)\n", label, slot, v.x, v.y);
}

template <typename T>
__device__ __forceinline__ void debug_dump_reg_4096(const char* label,
                                                    vec2_t<T>* reg) {
    if (blockIdx.x != 0 || threadIdx.y != 0 || threadIdx.x >= 4) {
        return;
    }
    for (int i = 0; i < 4; ++i) {
        debug_print_complex_4096(label, i, reg[i]);
    }
}

template <typename T>
__device__ __forceinline__ void debug_dump_smem_4096(const char* label,
                                                     vec2_t<T>* smem,
                                                     int laneid,
                                                     int warpid) {
    if (blockIdx.x != 0 || warpid != 0 || laneid >= 4) {
        return;
    }
    int col = laneid;
    int row0 = laneid >> 2;
    int row1 = row0 + 8;
    debug_print_complex_4096(label, 0, smem[smem_index_4096(row0, col)]);
    debug_print_complex_4096(label, 1, smem[smem_index_4096(row1, col)]);
}
#endif
} // namespace detail

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, true>(vec2_t<float>* __restrict__ reg,
                                            vec2_t<float>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);


    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        
        smem[detail::smem_index_4096(row0, col)+warpid*4] = reg[i];
        smem[detail::smem_index_4096(row1, col)+warpid*4] = reg[i + ept / 2];
    }

    __syncthreads();



    // __syncthreads();
    // if(blockIdx.x==0 && threadIdx.x ==0 && threadIdx.y ==0){
    //     for(int i=0; i<64; i++) {
    //         for(int j=0; j<64; j++) {
    //             int idx = (i*64+j); idx += idx/ 16; 
    //             printf("(%f, %f) ", smem[idx].x, smem[idx].y);
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();

    for (int i = 0; i < ept / 2; ++i) {
        // int row = i * 4 + (laneid % 4);
        int row = (laneid %4) * 16 + i/4 + (i%4)*4;
        int col0 = warpid * 16 + (laneid / 4);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)+(laneid%4)*4];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)+(laneid%4)*4];
        rotate(reg[i], col0 * row, 4096);
        rotate(reg[i + ept / 2], col1 * row, 4096);
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

}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 4096, 1, false>(vec2_t<float>* __restrict__ reg,
                                             vec2_t<float>* W_precompute,
                                             void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<float>* smem = reinterpret_cast<vec2_t<float>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit::fft_kernel_r64_b16<false>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], -col0 * row, 4096);
        rotate(reg[i + ept / 2], -col1 * row, 4096);
    }

    thunderfft::unit::fft_kernel_r64_b16<false>((float*)reg);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i * 4 + (laneid & 3);
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        smem[detail::smem_index_4096(row, col0)] = reg[i];
        smem[detail::smem_index_4096(row, col1)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        reg[i] = smem[detail::smem_index_4096(row0, col)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row1, col)];
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, true>(vec2_t<half>* __restrict__ reg,
                                           vec2_t<half>* W_precompute,
                                           void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_precompute);
#ifdef THUNDERFFT_DEBUG_FP16_4096
    detail::debug_dump_reg_4096("fp16_r64_stage0", reg);
#endif

    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col) + warpid * 8] = reg[i];
        smem[detail::smem_index_4096(row1, col) + warpid * 8] = reg[i + ept / 2];
    }

    __syncthreads();
#ifdef THUNDERFFT_DEBUG_FP16_4096
    detail::debug_dump_smem_4096("fp16_store_stage1", smem, laneid, warpid);
#endif

    for (int i = 0; i < ept / 2; ++i) {
        int row = (laneid & 3) * 16 + i / 4 + (i % 4) * 4;
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0) + (laneid & 3) * 8];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1) + (laneid & 3) * 8];
        rotate(reg[i], col0 * row, 4096);
        rotate(reg[i + ept / 2], col1 * row, 4096);
    }
#ifdef THUNDERFFT_DEBUG_FP16_4096
    detail::debug_dump_reg_4096("fp16_reload_rotate_stage2", reg);
#endif

    thunderfft::unit_fp16::fft_kernel_r64_b16<true>(reg, W_precompute);
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 4096, 1, false>(vec2_t<half>* __restrict__ reg,
                                            vec2_t<half>* W_precompute,
                                            void* workspace) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    if (warpid >= 4) {
        return;
    }

    vec2_t<half>* smem = reinterpret_cast<vec2_t<half>*>(workspace);
    constexpr int ept = 32;

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_precompute);

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int col = i + (laneid & 3) * 16;
        int row0 = warpid * 16 + (laneid >> 2);
        int row1 = row0 + 8;
        smem[detail::smem_index_4096(row0, col)] = reg[i];
        smem[detail::smem_index_4096(row1, col)] = reg[i + ept / 2];
    }

    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = (laneid & 3) * 16 + i / 4 + (i % 4) * 4;
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;
        reg[i] = smem[detail::smem_index_4096(row, col0)];
        reg[i + ept / 2] = smem[detail::smem_index_4096(row, col1)];
        rotate(reg[i], -col0 * row, 4096);
        rotate(reg[i + ept / 2], -col1 * row, 4096);
    }

    thunderfft::unit_fp16::fft_kernel_r64_b16<false>(reg, W_precompute);
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N4096(vec2_t<T>* __restrict__ reg,
                          const vec2_t<T>* __restrict__ smem) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i/4+(i%4)*4 + (laneid & 3) * 16;
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;

        int idx0 = row * 64 + col0;
        int idx1 = row * 64 + col1;
        idx0 = idx0 + idx0 / sL::pad_period * sL::pad;
        idx1 = idx1 + idx1 / sL::pad_period * sL::pad;
        idx0 *= sL::elem_stride;
        idx1 *= sL::elem_stride;

        reg[i] = smem[idx0];
        reg[i + ept / 2] = smem[idx1];
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N4096(vec2_t<T>* __restrict__ smem,
                          const vec2_t<T>* __restrict__ reg) {
    int laneid = threadIdx.x & 31;
    int warpid = threadIdx.y;
    constexpr int ept = 32;

    #pragma unroll
    for (int i = 0; i < ept / 2; ++i) {
        int row = i + (laneid & 3) * 16;
        int col0 = warpid * 16 + (laneid >> 2);
        int col1 = col0 + 8;

        int idx0 = row * 64 + col0;
        int idx1 = row * 64 + col1;
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
    //         for(int j=0; j<64; j++) {
    //             int idx = (i*64+j); idx += idx/ sL::pad_period * sL::pad; 
    //             printf("%f %f, ", smem[idx].x, smem[idx].y);
    //         }
    //         printf("\n");
    //     }
    // }
    // __syncthreads();


}
} // namespace thunderfft
