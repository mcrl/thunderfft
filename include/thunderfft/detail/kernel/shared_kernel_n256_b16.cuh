namespace thunderfft {
template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 256, 16, true>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W_precompute, void *workspace) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    int ept = 32;

    vec2_t<float> *smem = (vec2_t<float>*)workspace;
    thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    // thunderfft::unit::reg2smem(reg, smem+((laneid/4) + warpid * 16)*(64+4), smem + ((laneid/4+8) + warpid*16)*(64+4), 1);
    {
        const int pad_period=64, pad=1;
        int batch0 = (laneid/4);
        int batch1 = (laneid/4+8);

        for (int i = 0; i < ept / 2; i++) {
            // int idx = i*4 + (laneid % 4)*64 + warpid;
            int idx = i + warpid * 16 + (laneid % 4)*64;

            int idx0 = batch0*256+idx;
            int idx1 = batch1*256+idx;
            idx0 = idx0 + idx0/pad_period*pad;
            idx1 = idx1 + idx1/pad_period*pad;

            smem[idx0] = reg[i];
            smem[idx1] = reg[i + ept / 2];
        }

        // __syncthreads();

        // if(threadIdx.y==0 && blockIdx.x==0) {
        //     int idx = batch0+4*((laneid % 4)*16) + warpid;
        //     idx = idx + idx/pad_period*pad;
        //     printf("%d %d %d\n", laneid, idx, idx%32);
        // }
        // __syncthreads();

        __syncthreads();

        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                // int x= (laneid%4) *16 + warpid*4 + i;
                int x= (laneid%4) + warpid*4 + i*16;
                // int index = j + 4*x;
                int index = (laneid%4) + warpid*4 + j*16 + i * 64;

                int index0 = batch0*256 + index;
                int index1 = batch1*256 + index;
                index0 = index0 + index0/pad_period*pad;
                index1 = index1 + index1/pad_period*pad;

                reg[i*4+j] = smem[index0];
                reg[i*4+j+16] = smem[index1];
                
                // reg[i*4+j] = s_out_0[0];
                // reg[i*4+j+16] = s_out_1[0];
                
                if (j != 0) {
                    rotate(reg[i*4+j], x * j, 256);
                    rotate(reg[i*4+j+16], x * j, 256);
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        auto a = reg[4 * i + 0];
        auto b = reg[4 * i + 1];
        auto c = reg[4 * i + 2];
        auto d = reg[4 * i + 3];

        // radix-4 butterfly
        auto t0 = a + c;
        auto t1 = a - c;
        auto t2 = b + d;
        auto t3 = b - d; t3 = {t3.y, -t3.x}; // multiply by -i

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    
    __syncthreads();

    // __syncthreads();
    // if(threadIdx.x == 1 && threadIdx.y==0 && blockIdx.x==0) {
    //     for(int i=0; i<32; i++) {
    //         printf("%f %f\n", reg[i].x, reg[i].y);
    //     }
    //     printf("-----------\n");
    // }
    // __syncthreads();

    // thunderfft::unit::fft_kernel_r64_b16<true>((float*)reg);

    // __syncthreads();
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<float, 256, 16, false>(vec2_t<float>* __restrict__ reg, vec2_t<float>* W, void *workspace) {
    // The unnormalized inverse is conj(FFT(conj(x))), with the forward layout.
    constexpr int ept = 32;
    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        reg[i].y = -reg[i].y;
    }
    ThunderFFT_kernel_reg<float, 256, 16, true>(reg, W, workspace);
    #pragma unroll
    for (int i = 0; i < ept; ++i) {
        reg[i].y = -reg[i].y;
    }
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 256, 16, true>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W_precompute, void *workspace) {
    const unsigned laneid = threadIdx.x;
    const unsigned warpid = threadIdx.y;

    int ept = 32;
    
    vec2_t<half> *smem = (vec2_t<half>*)workspace;
    thunderfft::unit_fp16::fft_kernel_r64_b16<true>((vec2_t<half>*)reg, W_precompute);

    {
        const int pad_period=64, pad=1;
        int batch0 = (laneid/4);
        int batch1 = (laneid/4+8);

        for (int i = 0; i < ept / 2; i++) {
            // int idx = i*4 + (laneid % 4)*64 + warpid;
            // One half2 is one bank word. With a 32 x 4 block,
            // i + 16*warpid stays below 64; fold the 64+1 skew into
            // the base instead of recomputing signed divisions per value.
            const unsigned base = 260 * (laneid / 4) + 65 * (laneid % 4)
                                + 16 * warpid;
            smem[base + i] = reg[i];
            smem[base + 2080 + i] = reg[i + ept / 2];
        }

        // __syncthreads();

        // if(threadIdx.y==0 && blockIdx.x==0) {
        //     int idx = batch0+4*((laneid % 4)*16) + warpid;
        //     idx = idx + idx/pad_period*pad;
        //     printf("%d %d %d\n", laneid, idx, idx%32);
        // }
        // __syncthreads();

        __syncthreads();

        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                // int x= (laneid%4) *16 + warpid*4 + i;
                int x= (laneid%4) + warpid*4 + i*16;
                // int index = j + 4*x;
                int index = (laneid%4) + warpid*4 + j*16 + i * 64;

                // The transpose has the same skew: i selects a 65-word
                // row and j a 16-word segment. The second batch is 8*260 away.
                const unsigned base = 260 * (laneid / 4) + (laneid % 4)
                                    + 4 * warpid + 65 * i + 16 * j;
                reg[i*4+j] = smem[base];
                reg[i*4+j+16] = smem[base + 2080];
                
                // reg[i*4+j] = s_out_0[0];
                // reg[i*4+j+16] = s_out_1[0];
                
                if (j != 0) {
                    const half2 cached = __ldg(ThunderFFT_get_twiddle<half>() + x * j);
                    const half2 factor = __hmul2(cached, __floats2half2_rn(1.f, -1.f));
                    reg[i*4+j] = cmul(reg[i*4+j], factor);
                    reg[i*4+j+16] = cmul(reg[i*4+j+16], factor);
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        auto a = reg[4 * i + 0];
        auto b = reg[4 * i + 1];
        auto c = reg[4 * i + 2];
        auto d = reg[4 * i + 3];

        // radix-4 butterfly
        auto t0 = a + c;
        auto t1 = a - c;
        auto t2 = b + d;
        auto t3 = __hmul2(__lowhigh2highlow(b - d), __floats2half2_rn(1.f, -1.f));

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    
    __syncthreads();
}

template<>
__device__ __forceinline__ void
ThunderFFT_kernel_reg<half, 256, 16, false>(vec2_t<half>* __restrict__ reg, vec2_t<half>* W, void *workspace) {
    const unsigned laneid = threadIdx.x;
    const unsigned warpid = threadIdx.y;

    int ept = 32;
    
    vec2_t<half> *smem = (vec2_t<half>*)workspace;
    thunderfft::unit_fp16::fft_kernel_r64_b16<false>((vec2_t<half>*)reg, W);

    {
        const int pad_period=64, pad=1;
        int batch0 = (laneid/4);
        int batch1 = (laneid/4+8);

        for (int i = 0; i < ept / 2; i++) {
            // One half2 is one bank word. With a 32 x 4 block,
            // i + 16*warpid stays below 64; fold the 64+1 skew into
            // the base instead of recomputing signed divisions per value.
            const unsigned base = 260 * (laneid / 4) + 65 * (laneid % 4)
                                + 16 * warpid;
            smem[base + i] = reg[i];
            smem[base + 2080 + i] = reg[i + ept / 2];
        }

        __syncthreads();

        for(int i=0; i<4; i++) {
            for(int j=0; j<4; j++) {
                int x= (laneid%4) + warpid*4 + i*16;
                int index = (laneid%4) + warpid*4 + j*16 + i * 64;

                // The transpose has the same skew: i selects a 65-word
                // row and j a 16-word segment. The second batch is 8*260 away.
                const unsigned base = 260 * (laneid / 4) + (laneid % 4)
                                    + 4 * warpid + 65 * i + 16 * j;
                reg[i*4+j] = smem[base];
                reg[i*4+j+16] = smem[base + 2080];
                
                if (j != 0) {
                    const half2 cached = __ldg(ThunderFFT_get_twiddle<half>() + x * j);
                    const half2 factor = cached;
                    reg[i*4+j] = cmul(reg[i*4+j], factor);
                    reg[i*4+j+16] = cmul(reg[i*4+j+16], factor);
                }
            }
        }
    }

    for (int i = 0; i < 8; i++) {
        auto a = reg[4 * i + 0];
        auto b = reg[4 * i + 1];
        auto c = reg[4 * i + 2];
        auto d = reg[4 * i + 3];

        // radix-4 butterfly
        auto t0 = a + c;
        auto t1 = a - c;
        auto t2 = b + d;
        auto t3 = __hmul2(__lowhigh2highlow(b - d), __floats2half2_rn(-1.f, 1.f));

        reg[4 * i + 0] = t0 + t2;
        reg[4 * i + 1] = t1 + t3;
        reg[4 * i + 2] = t0 - t2;
        reg[4 * i + 3] = t1 - t3;
    }
    
    __syncthreads();
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N256(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    // grid = {32, 4};
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;
    int ept = 32; // N * batch / (threads_per_warp * warps_per_block)

    int batch0 = (laneid/4);
    int batch1 = (laneid/4+8);

    for (int i = 0; i < ept / 2; i++) {
        int idx = i*4 + (laneid % 4) + warpid * 64;
        if constexpr( !sL::reversed ) {
            idx = reverse_bit_groups<2,8>(idx);
        }

        int idx0 = batch0*256+idx;
        int idx1 = batch1*256+idx;
        idx0 = idx0 + idx0/sL::pad_period*sL::pad;
        idx1 = idx1 + idx1/sL::pad_period*sL::pad;
        idx0 = idx0 * sL::elem_stride;
        idx1 = idx1 * sL::elem_stride;

        reg[i] = smem[idx0];
        reg[i + ept / 2] = smem[idx1];
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
ThunderFFT_reg2smem_N256(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    int laneid = threadIdx.x;
    int warpid = threadIdx.y;

    auto s_out_0 = smem + (laneid/4) * (sL::batch_stride + sL::batch_stride/sL::pad_period * sL::pad);
    auto s_out_1 = smem + (laneid/4+8) * (sL::batch_stride + sL::batch_stride/sL::pad_period * sL::pad);
    for(int i=0; i<4; i++) {
        int index_i = (laneid%4) + i*16 + warpid*4;
        // int index_i = (laneid%4) *16 + warpid*4 + i;


        for(int j=0; j<4; j++) {
            int index = (index_i + j*64) * sL::elem_stride;
            index += index / sL::pad_period * sL::pad;
            s_out_0[index] = reg[i*4+j];
            s_out_1[index] = reg[i*4+j+16];

        }
    }
}
}
