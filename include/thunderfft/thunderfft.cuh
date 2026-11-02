
#pragma once
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "detail/util_cuda.h"

namespace thunderfft {

// Public entry point for CUDA translation units; see sample/README.md for
// complete host-call and kernel-fusion examples and configuration limitations.
// T names the component type: float uses float2 (real, imaginary) storage and
// TF32 tensor-core arithmetic; half uses half2 (low=real, high=imaginary).
// float storage does not imply full FP32 arithmetic throughout the FFT.

// Block-local layout descriptor used by the device data-movement helpers.
// N_ and Batch_ describe one block's tile, not the total global batch.
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
// PadPeriod_ must be positive even when Pad_ is zero. Reversed_ controls the
// input permutation expected by each size-specific helper; it is not a generic
// bit-reversal operation. Only the documented sample layouts are demonstrated.

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


// Initialize internal twiddle tables for T=float or T=half and positive N on
// the current CUDA device before any FFT call. There is one cached N per T;
// initializing another N replaces that allocation. Initialization/finalization
// are not stream-ordered: finish all users before replacing or releasing it.
// The cache is not keyed by device or protected against concurrent callers;
// keep its lifetime on one current device and serialize cache changes.
template <typename T>
void ThunderFFTInitialize(int N);

// Release the cached allocation on the same current device after all dependent
// kernels finish. Call Initialize again before reuse; this is not a full reset
// of device symbols or the FP16 unit-table initialization flag.
template <typename T>
void ThunderFFTFinalize();

// Reserved declaration without a definition in this release; do not call.
// Use ThunderFFTInitialize/Finalize for managed twiddle storage instead.
template <typename T>
static T* preprocess_W(int N);


// Device utilities below are collective, size-specific fragment mappings.
// Use the same (N, block batch, launch shape) as ThunderFFT_kernel_reg, provide
// full tiles and non-overlapping source/destination storage, and insert the
// required synchronization around shared-memory producers/consumers. These
// helpers do not perform bounds checks or insert barriers.


// Global memory -> per-thread FFT fragment. gmem points to the block-local
// base of batch contiguous N-element signals in natural order; the caller
// applies the block offset. reg needs N*batch/(32*warp_per_block<N>) elements.
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_gmem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ gmem);


// Output FFT fragment -> natural-order, contiguous global-memory tile.
// gmem is block-local and reg has the same per-thread size as above.
template <typename T, int N, int batch>
__device__ __forceinline__ void
ThunderFFT_reg2gmem(vec2_t<T>* __restrict__ gmem,
                    const vec2_t<T>* __restrict__ reg);


// Shared memory described by sL -> the size-specific input FFT fragment.
// The fragment is distributed across lanes/warps, not a natural-order vector.
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2reg(vec2_t<T>* __restrict__ reg,
                    const vec2_t<T>* __restrict__ smem);


// Size-specific output FFT fragment -> shared memory described by sL.
// Input and output fragment mappings differ; use a store/load conversion when
// composing FFT stages as demonstrated in sample/README.md.
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_reg2smem(vec2_t<T>* __restrict__ smem,
                    const vec2_t<T>* __restrict__ reg);


// Contiguous natural-order global tile -> shared memory described by sL.
// gmem must already point to the block-local base; sL strides/padding describe
// shared storage. sL::reversed optionally permutes the input during this copy.
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_gmem2smem(vec2_t<T>* __restrict__ smem,
                     const vec2_t<T>* __restrict__ gmem);


// Shared memory described by sL -> contiguous global tile at a block-local
// base. Requires sL::reversed == false; the true branch does not write output.
template <typename T, typename sL>
__device__ __forceinline__ void
ThunderFFT_smem2gmem(vec2_t<T>* __restrict__ gmem,
                     const vec2_t<T>* __restrict__ smem);


// Host launch wrapper for batched complex-to-complex FFTs. Input and output
// are separate device buffers of batch*N complex elements, with signal j at
// offset j*N and natural-order input/output. forward=true uses exp(-2*pi*i*k/N);
// false selects an unnormalized inverse: scale its output by 1/N as needed.
//
// Implemented launch tuples (N, FFTs per block, warps per block):
// (64,16,1), (128,8,1), (256,16,4), (512,2,1), (1024,1,1),
// (2048,1,2), (4096,1,4). This lists implementation coverage, not a correctness
// guarantee; consult sample/README.md for verified configurations and limits.
// batch must be positive and a multiple of FFTs per block: the kernel processes
// complete tiles and does not mask a partial final block.
//
// N=512 has no inverse definition; N=2048 inverse is incomplete. The wrapper
// supplies W=nullptr internally, so half at N=64/128/256/1024/2048 cannot be used
// through this interface: those specializations require per-thread coefficients.
//
// Initialize<T>(N) first. The wrapper synchronizes stream before returning,
// including prior work in that stream, but discards the synchronization status
// and returns no status. Check CUDA errors in the caller; sample/README.md shows
// a checked call. The current wrapper is not an asynchronous launch interface.

template <typename T, int N, bool forward = true>
inline void ThunderFFT_global(vec2_t<T>* d_input,
                              vec2_t<T>* d_output,
                              int batch,
                              cudaStream_t stream = 0);


// Reserved shared-memory API declarations without definitions in this release.
// They cannot be instantiated; use the implemented data-movement helpers and
// ThunderFFT_kernel_reg for kernel fusion.
template <typename T>
__device__ __forceinline__ void
ThunderFFT_kernel_smem(vec2_t<T>* __restrict__ s_input,
                       vec2_t<T>* __restrict__ s_output,
                       const T* __restrict__ W_N,
                       int N,
                       int batch);


// Reserved declaration; no implementation is provided.
template <typename T, typename input_L, typename output_L>
__device__ __forceinline__ void
ThunderFFT_kernel_smem_pad(vec2_t<T>* __restrict__ s_input,
                           vec2_t<T>* __restrict__ s_output,
                           const T* __restrict__ W_N,
                           int N,
                           int batch);


// Device FFT for fusion into a caller-owned CUDA kernel. Here batch is the
// compile-time FFT count per block, not the global batch. Only the explicit
// (T,N,batch,forward) specializations included below are defined. N=512 inverse
// is undefined and N=2048 inverse is incomplete, as for the host wrapper.
//
// Launch a complete block of dim3(32, warp_per_block<N>) and call collectively
// with every participating lane/warp; several specializations contain block
// barriers. reg is a per-thread fragment of N*batch/(32*warp_per_block<N>)
// complex values prepared by the matching load helper. It is overwritten with
// the output fragment, which can require a store/load conversion before reuse.
//
// workspace is caller-owned shared memory, distinct from live register data.
// For the standard launch, reserve sizeof(vec2_t<T>)*(N+pad_h(N))*batch bytes,
// aligned for vec2_t<T>; consult the specialization for any different layout.
// W is specialization-specific: several FP16 paths need direction-specific
// per-thread MMA coefficients, while others build them internally. It is not
// the global N-element twiddle table; a null W is not valid for every variant.
// Follow the matching sample's coefficient setup and shared-memory barriers.
// Initialization and inverse normalization rules are the same as above.
template <typename T, int N, int batch, bool forward>
__device__ __forceinline__ void
ThunderFFT_kernel_reg(vec2_t<T>* __restrict__ reg, vec2_t<T>* __restrict__ W, void *workspace);

}  // namespace thunderfft


#include "thunderfft/detail/thunderfft_impl.cuh"
