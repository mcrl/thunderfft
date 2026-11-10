// thunderfft.cu
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <vector>
#include <string>

#include "util_arith.h"

namespace thunderfft {
    
template <int N>
inline constexpr int batch_per_block =
    (N == 64)   ? 16 :
    (N == 128)  ? 8  :
    (N == 256)  ? 16 :
    (N == 512)  ? 2  :
    (N == 1024 || N == 4096) ? 1 :
    (N == 2048) ? 1 :
    0;

template <int N>
inline constexpr int warp_per_block =
    (N == 64 || N == 128 || N == 1024) ? 1 :
    (N == 256 || N == 4096) ? 4 :
    (N == 2048) ? 2 :
    (N == 512) ? 1 :
    0;

constexpr int threads_per_warp = 32;


}
#include "thunderfft_impl_twiddle.cuh"
#include "thunderfft_impl_helper.cuh"
#include "thunderfft_impl_kernel.cuh"
