
#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "detail/util_cuda.h"

namespace thunderfft {

// Logical indexing:
//   i : element index within a signal
//   j : batch index
//
// Linear logical index (no padding):
//   idx = i * elem_stride + j * batch_stride
//
// Physical index with periodic padding:
//   - Every pad_period logical elements form one block
//   - Each block is followed by 'pad' padding elements
//
//   padded_idx = idx + (idx / pad_period) * pad

template<int N_, int Batch_, int ElemStride_, int BatchStride_, int PadPeriod_, int Pad_, bool Reversed_>
struct layout_t {
  static constexpr int N           = N_;
  static constexpr int batch       = Batch_;
  static constexpr int elem_stride = ElemStride_;
  static constexpr int batch_stride= BatchStride_;
  static constexpr int pad_period  = PadPeriod_;
  static constexpr int pad         = Pad_;
  static constexpr int reversed     = Reversed_;
};


// Twiddle factor management.

// Persistent twiddle management.
template <typename T>
void ThunderFFTInitialize(int N);

// Release internal twiddle tables.
template <typename T>
void ThunderFFTFinalize();

// Allocate and precompute W_N on device memory.
template <typename T>
static T* preprocess_W(int N);


// Device utilities: data movement (gmem/smem <-> registers).


// Global memory -> register file.
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_gmem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ gmem);


// Register file -> global memory.
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_reg2gmem(vec2_t<T>* __restrict__ gmem,
                    const vec2_t<T>* __restrict__ reg);


// Shared memory -> register file (layout-aware).
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);


// Register file -> shared memory (layout-aware).
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);


// Global memory -> shared memory (layout-aware).
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_gmem2smem(vec2_t<T>* __restrict__ smem,
                     const vec2_t<T>* __restrict__ gmem);


// Shared memory -> global memory (layout-aware).
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2gmem(vec2_t<T>* __restrict__ gmem,
                     const vec2_t<T>* __restrict__ smem);


// Public API: launch wrapper (global I/O).

template <typename T, int N, bool forward = true>
inline void ThunderFFT_global(vec2_t<T>* d_input,
                              vec2_t<T>* d_output,
                              int batch,
                              cudaStream_t stream = 0);


// FFT kernels: shared-memory / register variants.

// Shared-memory kernel (no padding layout assumed).
template <typename T>
__device__ __forceinline__ void
ThunderFFT_kernel_smem(vec2_t<T>* __restrict__ s_input,
                       vec2_t<T>* __restrict__ s_output,
                       const T* __restrict__ W_N,
                       int N,
                       int batch);


// Shared-memory kernel (layout-aware padding).
template <typename T, typename input_L, typename output_L>
__device__ __forceinline__ void
ThunderFFT_kernel_smem_pad(vec2_t<T>* __restrict__ s_input,
                           vec2_t<T>* __restrict__ s_output,
                           const T* __restrict__ W_N,
                           int N,
                           int batch);


// Register-resident kernel.
template <typename T, int N, int batch, bool forward>
__device__ __forceinline__ void
ThunderFFT_kernel_reg(vec2_t<T>* __restrict__ reg, vec2_t<T>* __restrict__ W, void *workspace);

}  // namespace thunderfft


#include "thunderfft/detail/thunderfft_impl.cuh"
