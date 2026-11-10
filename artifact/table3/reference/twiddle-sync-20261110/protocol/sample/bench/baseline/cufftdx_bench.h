// cuFFTDx kernels for the paper timing adapter (paper_fft.cu).
// The candidate table below holds tuned FFTsPerBlock/ElementsPerThread
// configurations for SM 80/90/100 (A100/H100/B200); the cuFFTDx default Block
// configuration is added for each size/precision/placement.
// Input/output are natural-order, contiguous, N complex elements per FFT.
#pragma once
#include <cuda_fp16.h>
#include <cufftdx.hpp>
#include <algorithm>
#include <string>
#include <type_traits>

#ifndef THFF_CUFFTDX_SM
#error "THFF_CUFFTDX_SM (e.g. 800) must name the single cuFFTDx target SM"
#endif

namespace thff_cufftdx {

// FP16 RI <-> cuFFTDx RRII conversion (NVIDIA CUDALibrarySamples fp16_common.hpp).
__device__ __forceinline__ cufftdx::complex<__half2> to_rrii(__half2 a, __half2 b) {
  return cufftdx::complex<__half2>(__lows2half2(a, b), __highs2half2(a, b));
}
__device__ __forceinline__ __half2 to_ri1(cufftdx::complex<__half2> v) { return __lows2half2(v.x, v.y); }
__device__ __forceinline__ __half2 to_ri2(cufftdx::complex<__half2> v) { return __highs2half2(v.x, v.y); }

// Block FFT, register placement: load once, repeat FFT().execute on registers.
template <class FFT, class T>
__launch_bounds__(FFT::max_threads_per_block) __global__
void batch_reg(T* input, T* output, int batch, int repeat) {
  using C = typename FFT::value_type;
  constexpr int N = cufftdx::size_of<FFT>::value;
  constexpr int Stride = FFT::stride;
  constexpr bool Half = std::is_same_v<T, __half2>;
  const int fft = blockIdx.x * FFT::ffts_per_block + threadIdx.y;
  if (fft >= batch) return;
  extern __shared__ __align__(alignof(float4)) unsigned char smem[];
  C data[FFT::storage_size];
  // FP16 packs FFT y and y+FPB/2 of this block into one RRII lane pair.
  const unsigned offset = Half ? fft * N : fft * N / FFT::implicit_type_batching;
  constexpr unsigned pair = (FFT::ffts_per_block / 2) * N;
  unsigned index = offset + threadIdx.x;
  for (unsigned i = 0; i < FFT::elements_per_thread; ++i, index += Stride) {
    if (i * Stride + threadIdx.x < N) {
      if constexpr (Half) data[i] = to_rrii(input[index], input[index + pair]);
      else data[i] = reinterpret_cast<C*>(input)[index];
    }
  }
#pragma unroll 1
  for (int r = 0; r < repeat; ++r) FFT().execute(data, reinterpret_cast<void*>(smem));
  index = offset + threadIdx.x;
  for (unsigned i = 0; i < FFT::elements_per_thread; ++i, index += Stride) {
    if (i * Stride + threadIdx.x < N) {
      if constexpr (Half) { output[index] = to_ri1(data[i]); output[index + pair] = to_ri2(data[i]); }
      else reinterpret_cast<C*>(output)[index] = data[i];
    }
  }
}

// Block FFT, shared placement: load to shared memory, repeat the shared-memory API.
template <class FFT, class T>
__launch_bounds__(FFT::max_threads_per_block) __global__
void batch_smem(T* input, T* output, int batch, int repeat) {
  using C = typename FFT::value_type;
  constexpr int N = cufftdx::size_of<FFT>::value;
  constexpr bool Half = std::is_same_v<T, __half2>;
  if (int(blockIdx.x * FFT::ffts_per_block + threadIdx.y) >= batch) return;
  extern __shared__ __align__(alignof(float4)) unsigned char smem[];
  C* s = reinterpret_cast<C*>(smem);
  const unsigned offset = Half ? blockIdx.x * FFT::ffts_per_block * N
                               : blockIdx.x * FFT::ffts_per_block * N / FFT::implicit_type_batching;
  constexpr unsigned pair = (FFT::ffts_per_block / 2) * N;
  const unsigned stride = blockDim.x * blockDim.y, limit = blockDim.y * FFT::input_length;
  T* in = input + offset;
  T* out = output + offset;
  unsigned index = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < FFT::elements_per_thread; ++i, index += stride) {
    if (index < limit) {
      if constexpr (Half) s[index] = to_rrii(in[index], in[index + pair]);
      else s[index] = reinterpret_cast<C*>(in)[index];
    }
  }
  __syncthreads();
#pragma unroll 1
  for (int r = 0; r < repeat; ++r) FFT().execute(s);
  __syncthreads();
  index = threadIdx.y * blockDim.x + threadIdx.x;
  for (int i = 0; i < FFT::elements_per_thread; ++i, index += stride) {
    if (index < limit) {
      if constexpr (Half) { out[index] = to_ri1(s[index]); out[index + pair] = to_ri2(s[index]); }
      else reinterpret_cast<C*>(out)[index] = s[index];
    }
  }
}

// Thread FFT (register only): one FFT per thread; FP16 threads hold FFTs
// i and i+blockDim.x of their block as one RRII pair. Batch must fill blocks.
template <class FFT, class T>
__global__ void thread_reg(T* input, T* output, int, int repeat) {
  using C = typename FFT::value_type;
  constexpr int N = cufftdx::size_of<FFT>::value;
  constexpr bool Half = std::is_same_v<T, __half2>;
  C data[FFT::storage_size];
  const unsigned pair = blockDim.x * N;
  const unsigned index = (Half ? 2u : 1u) * blockIdx.x * blockDim.x * N + threadIdx.x * FFT::elements_per_thread;
  for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
    if constexpr (Half) data[i] = to_rrii(input[index + i], input[index + i + pair]);
    else data[i] = reinterpret_cast<C*>(input)[index + i];
  }
#pragma unroll 1
  for (int r = 0; r < repeat; ++r) FFT().execute(data);
  for (unsigned i = 0; i < FFT::elements_per_thread; ++i) {
    if constexpr (Half) { output[index + i] = to_ri1(data[i]); output[index + i + pair] = to_ri2(data[i]); }
    else reinterpret_cast<C*>(output)[index + i] = data[i];
  }
}

// One launchable configuration. A launch uses count/ffts_per_launch_block
// blocks of `block` threads and `shared` dynamic shared bytes.
template <class T>
struct Candidate {
  void (*kernel)(T*, T*, int, int);
  dim3 block;
  size_t shared;
  int ffts_per_launch_block;  // FFTs covered by one CUDA block
  int ffts_per_block;         // cuFFTDx FFTsPerBlock (thread FFT: same as above)
  int elements_per_thread;
  std::string label;          // block_f<FPB>_e<EPT>, block_default, thread_t<threads>
};

template <class T>
using precision_t = cufftdx::Precision<std::conditional_t<std::is_same_v<T, __half2>, __half, float>>;

template <class T, int N>
using common = decltype(cufftdx::Size<N>() + precision_t<T>() + cufftdx::Type<cufftdx::fft_type::c2c>() +
                        cufftdx::Direction<cufftdx::fft_direction::forward>());

// FPB=0/EPT=0 selects cuFFTDx's default Block configuration.
template <class T, int N, bool Shared, unsigned FPB, unsigned EPT, class Visit>
void block_candidate(Visit&& visit) {
  using Base = std::conditional_t<FPB == 0, decltype(common<T, N>() + cufftdx::Block()),
      decltype(common<T, N>() + cufftdx::FFTsPerBlock<FPB>() + cufftdx::ElementsPerThread<EPT>() + cufftdx::Block())>;
  if constexpr (cufftdx::is_supported<Base, THFF_CUFFTDX_SM>::value) {
    using FFT = decltype(Base() + cufftdx::SM<THFF_CUFFTDX_SM>());
    using C = typename FFT::value_type;
    // Room for the block's data as well as
    // FFT::shared_memory_size, for both placements.
    constexpr size_t data = FFT::ffts_per_block / FFT::implicit_type_batching * N * sizeof(C);
    Candidate<T> c;
    if constexpr (Shared) c.kernel = batch_smem<FFT, T>; else c.kernel = batch_reg<FFT, T>;
    c.block = FFT::block_dim;
    c.shared = std::max<size_t>(FFT::shared_memory_size, data);
    c.ffts_per_launch_block = c.ffts_per_block = FFT::ffts_per_block;
    c.elements_per_thread = FFT::elements_per_thread;
    c.label = FPB == 0 ? std::string("block_default")
                       : "block_f" + std::to_string(FPB) + "_e" + std::to_string(EPT);
    visit(c);
  }
}

template <class T, int N, unsigned Threads, class Visit>
void thread_candidate(Visit&& visit) {
  using Base = decltype(common<T, N>() + cufftdx::Thread());
  if constexpr (cufftdx::is_supported<Base, THFF_CUFFTDX_SM>::value) {
    using FFT = decltype(Base() + cufftdx::SM<THFF_CUFFTDX_SM>());
    Candidate<T> c;
    c.kernel = thread_reg<FFT, T>;
    c.block = dim3(Threads);
    c.shared = 0;
    c.ffts_per_launch_block = c.ffts_per_block = (std::is_same_v<T, __half2> ? 2 : 1) * Threads;
    c.elements_per_thread = FFT::elements_per_thread;
    c.label = "thread_t" + std::to_string(Threads);
    visit(c);
  }
}

// cuFFTDx default plus every tuned candidate for this build's SM.
template <class T, int N, bool Shared, class Visit>
void for_each_candidate(Visit&& visit) {
  block_candidate<T, N, Shared, 0, 0>(visit);
  // ---- tuned candidates per SM ----
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 8, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 1, 4>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
    block_candidate<T, N, Shared, 1, 32>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 1, 4>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 1, 32>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, float2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 32, 16>(visit);
    block_candidate<T, N, Shared, 16, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 32, 32>(visit);
    block_candidate<T, N, Shared, 32, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 8, 8>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 16, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
    block_candidate<T, N, Shared, 4, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
    block_candidate<T, N, Shared, 4, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
    block_candidate<T, N, Shared, 4, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 2, 32>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 2, 32>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 800 && std::is_same_v<T, __half2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 8, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 16, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 8, 16>(visit);
    block_candidate<T, N, Shared, 4, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
    block_candidate<T, N, Shared, 2, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
    block_candidate<T, N, Shared, 2, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
    block_candidate<T, N, Shared, 2, 32>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, float2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 32, 4>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 32, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 16, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 16, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
    block_candidate<T, N, Shared, 8, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 4, 8>(visit);
    block_candidate<T, N, Shared, 8, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 900 && std::is_same_v<T, __half2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 32, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 32, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 16, 2>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 16, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 4, 16>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 8, 16>(visit);
    block_candidate<T, N, Shared, 16, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
    block_candidate<T, N, Shared, 4, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
    block_candidate<T, N, Shared, 4, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, float2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 1, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 64 && Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 64 && !Shared) {
    block_candidate<T, N, Shared, 32, 8>(visit);
    block_candidate<T, N, Shared, 16, 8>(visit);
    thread_candidate<T, N, 16>(visit);
    thread_candidate<T, N, 32>(visit);
    thread_candidate<T, N, 64>(visit);
    thread_candidate<T, N, 128>(visit);
    thread_candidate<T, N, 256>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 128 && Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 8, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 128 && !Shared) {
    block_candidate<T, N, Shared, 16, 8>(visit);
    block_candidate<T, N, Shared, 32, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 256 && Shared) {
    block_candidate<T, N, Shared, 8, 8>(visit);
    block_candidate<T, N, Shared, 8, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 256 && !Shared) {
    block_candidate<T, N, Shared, 8, 16>(visit);
    block_candidate<T, N, Shared, 16, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 512 && Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
    block_candidate<T, N, Shared, 4, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 512 && !Shared) {
    block_candidate<T, N, Shared, 2, 8>(visit);
    block_candidate<T, N, Shared, 4, 8>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 1024 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 1024 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 2048 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 2048 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 4096 && Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  if constexpr (THFF_CUFFTDX_SM == 1000 && std::is_same_v<T, __half2> && N == 4096 && !Shared) {
    block_candidate<T, N, Shared, 2, 16>(visit);
  }
  // ---- end generated ----
}

}  // namespace thff_cufftdx
