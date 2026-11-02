namespace thunderfft {
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_gmem2smem(vec2_t<T>* __restrict__ smem,
                      const vec2_t<T>* __restrict__ gmem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride= sL::batch_stride;
    constexpr int pad_period  = sL::pad_period;
    constexpr int pad         = sL::pad;

    int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = 0; i < batch; i++) {
        for(int _j=0; _j<N; _j+= threads_per_warp * warp_per_block<N>) {
            int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx/pad_period * pad;

            vec2_t<T> value;
            if constexpr(sL::reversed) {
                value = gmem[i * N + reverse_bit_groups<2,LOG2P_builtin<N>>(j)];
            } else {
                value = gmem[i * N + j];
            }

            // if constexpr (std::is_same_v<T, float>) {
            //     value.x = __uint_as_float(__float_as_uint(value.x) + 0x1000u);
            //     value.y = __uint_as_float(__float_as_uint(value.y) + 0x1000u);
            // }
            smem[smem_idx] = value;
              // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
        }
    }
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2gmem(vec2_t<T>* __restrict__ gmem,
                     const vec2_t<T>* __restrict__ smem) {
    constexpr int N = sL::N;
    constexpr int batch = sL::batch;
    constexpr int elem_stride = sL::elem_stride;
    constexpr int batch_stride= sL::batch_stride;
    constexpr int pad_period  = sL::pad_period;
    constexpr int pad         = sL::pad;

    int tidx = threadIdx.x + threadIdx.y * blockDim.x;
    int block_size = blockDim.x * blockDim.y;
    for (int i = 0; i < batch; i++) {
        for(int _j=0; _j<N; _j+= threads_per_warp * warp_per_block<N>) {
            int j = _j + tidx;
            int smem_idx = i * batch_stride + j * elem_stride;
            smem_idx = smem_idx + smem_idx/pad_period * pad;

            if constexpr(sL::reversed) {
                // assertion(false && "Output must not be reversed");
                // gmem[i * N + reverse_bit_groups<2,LOG2P_builtin<N>>(j)] = smem[smem_idx];
            } else {
                gmem[i * N + j] = smem[smem_idx];
            }
              // s_in[i * (N+pad)+ j] = d_input[b * N * batch_per_block + i * N + j];
        }
    }
}


template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N64(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N64(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

                    
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N128(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N128(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N256(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N256(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N512(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N512(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N1024(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N1024(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N2048(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N2048(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg_N4096(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem_N4096(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);


template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem) {
    constexpr int N = sL::N;
    if constexpr(N == 64) ThunderFFT_smem2reg_N64<T, sL>(reg, smem);
    else if constexpr(N == 128) ThunderFFT_smem2reg_N128<T, sL>(reg, smem);
    else if constexpr(N == 256) ThunderFFT_smem2reg_N256<T, sL>(reg, smem);
    else if constexpr(N == 512) ThunderFFT_smem2reg_N512<T, sL>(reg, smem);
    else if constexpr(N == 1024) ThunderFFT_smem2reg_N1024<T, sL>(reg, smem);
    else if constexpr(N == 2048) ThunderFFT_smem2reg_N2048<T, sL>(reg, smem);
    else if constexpr(N == 4096) ThunderFFT_smem2reg_N4096<T, sL>(reg, smem);
    // else static_assert(N == 64 || N == 128 || N == 256 || N == 512 || N == 1024 || N == 2048 || N == 4096, "Unsupported N");
}

template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg) {
    constexpr int N = sL::N;
    if constexpr(N == 64) ThunderFFT_reg2smem_N64<T, sL>(smem, reg);
    else if constexpr(N == 128) ThunderFFT_reg2smem_N128<T, sL>(smem, reg);
    else if constexpr(N == 256) ThunderFFT_reg2smem_N256<T, sL>(smem, reg);
    else if constexpr(N == 512) ThunderFFT_reg2smem_N512<T, sL>(smem, reg);
    else if constexpr(N == 1024) ThunderFFT_reg2smem_N1024<T, sL>(smem, reg);
    else if constexpr(N == 2048) ThunderFFT_reg2smem_N2048<T, sL>(smem, reg);
    else if constexpr(N == 4096) ThunderFFT_reg2smem_N4096<T, sL>(smem, reg);
    // else static_assert(N == 64 || N == 128 || N == 256 || N == 512 || N == 1024 || N == 2048 || N == 4096, "Unsupported N");
}

template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_gmem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ gmem) {
    // gmem points to the block-local base (caller applies blockIdx offset).
    using L = layout_t<N, batch, 1, N, 1, 0, false>;
    ThunderFFT_smem2reg<T, L>(reg, gmem);
}

template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_reg2gmem(vec2_t<T>* __restrict__ gmem,
                    const vec2_t<T>* __restrict__ reg) {
    // gmem points to the block-local base (caller applies blockIdx offset).
    using L = layout_t<N, batch, 1, N, 1, 0, false>;
    ThunderFFT_reg2smem<T, L>(gmem, reg);
}

} // namespace thunderfft
