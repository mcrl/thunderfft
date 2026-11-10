// thunderfft_impl_twiddle.cuh
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <map>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "util_cuda.h"

namespace thunderfft {

// ============================================================
// Compile-time constants
// ============================================================
constexpr int WARP_SIZE      = 32;
constexpr int UNIT_ENTRIES   = 36;
constexpr int UNIT_W_SIZE    = UNIT_ENTRIES * WARP_SIZE * 2;

// ============================================================
// Host-side twiddle cache (N-dependent)
// ============================================================
template <typename T>
struct TwiddleCache {
  using TwiddleT = std::conditional_t<std::is_same_v<T, float>, float2, half2>;
  struct Entry {
    TwiddleT* dW = nullptr;
    int N_cached = 0;
    bool unit_initialized = false;
  };
  inline static std::map<int, Entry> devices;
  inline static std::mutex mutex;
};

// ============================================================
// Device-side symbols
// ============================================================
namespace detail {

// N-dependent twiddle (now complex: (cos, sin))
__device__ __constant__ const float2* d_twiddle_f   = nullptr;
__device__ __constant__ int           d_twiddle_N_f = 0;

__device__ __constant__ const half2*  d_twiddle_h   = nullptr;
__device__ __constant__ int           d_twiddle_N_h = 0;

// N-independent unit twiddle (lane-dependent!) - keep as is
__device__ half2  d_unit_twiddle_half2[UNIT_W_SIZE];
__device__ float2 d_unit_twiddle_float2[UNIT_W_SIZE];

template <typename T>
inline void set_twiddle_symbols(typename TwiddleCache<T>::TwiddleT* pointer,
                               int N) {
  if constexpr (std::is_same_v<T, float>) {
    tf_check_cuda(cudaMemcpyToSymbol(d_twiddle_f, &pointer, sizeof(pointer)),
                  "MemcpyToSymbol(d_twiddle_f)");
    tf_check_cuda(cudaMemcpyToSymbol(d_twiddle_N_f, &N, sizeof(N)),
                  "MemcpyToSymbol(d_twiddle_N_f)");
  } else {
    tf_check_cuda(cudaMemcpyToSymbol(d_twiddle_h, &pointer, sizeof(pointer)),
                  "MemcpyToSymbol(d_twiddle_h)");
    tf_check_cuda(cudaMemcpyToSymbol(d_twiddle_N_h, &N, sizeof(N)),
                  "MemcpyToSymbol(d_twiddle_N_h)");
  }
}

} // namespace detail

// ============================================================
// N-dependent twiddle builders
//   Store W[k] = (cos(2πk/N), sin(2πk/N))  in (x, y)
// ============================================================
inline float2* preprocess_W_float(int N) {
  if (N <= 0) return nullptr;

  std::vector<float2> h(N);
  const double scale = 2.0 * M_PI / double(N);

  for (int k = 0; k < N; ++k) {
    const float theta = float(scale * k);
    h[k] = make_float2(std::cos(theta), std::sin(theta));
  }

  float2* d = nullptr;
  tf_check_cuda(cudaMalloc(&d, sizeof(float2) * N), "cudaMalloc dW (float2)");
  tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(float2) * N,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy dW (float2)");
  return d;
}

inline half2* preprocess_W_half(int N) {
  if (N <= 0) return nullptr;

  std::vector<half2> h(N);
  const double scale = 2.0 * M_PI / double(N);

  for (int k = 0; k < N; ++k) {
    const float theta = float(scale * k);
    const half  c     = __float2half(float(std::cos(theta)));
    const half  s     = __float2half(float(std::sin(theta)));
    h[k] = make_half2(c, s);
  }

  half2* d = nullptr;
  tf_check_cuda(cudaMalloc(&d, sizeof(half2) * N), "cudaMalloc dW (half2)");
  tf_check_cuda(cudaMemcpy(d, h.data(), sizeof(half2) * N,
                           cudaMemcpyHostToDevice),
                "cudaMemcpy dW (half2)");
  return d;
}

// ============================================================
// CPU-side unit twiddle builder (lane-aware, N-independent)
//   (UNCHANGED)
// ============================================================
template <typename PairT>
inline PairT make_pair(float a, float b);

template <>
inline half2 make_pair<half2>(float a, float b) {
  return make_half2(__float2half(a), __float2half(b));
}

template <>
inline float2 make_pair<float2>(float a, float b) {
  return make_float2(a, b);
}

template <typename PairT>
void build_unit_twiddle_host(std::vector<PairT>& W) {
  // -----------------------------
  // local helper (TEMPORARY)
  // -----------------------------
  auto f = [](int x) {
      return (x & 1) * 4 + (x >> 1);
  };

  W.resize(UNIT_W_SIZE);

  constexpr int n     = 64;
  constexpr int radix = 4;

  int entry = 0;

  for (int stage = 0; stage < 3; ++stage) {
    const int stride = 1 << (stage << 1);

    for (int _j = 0; _j < n / radix; ++_j) {
      if (_j % (1 << (2 - stage)) != 0) continue;

      int j = (_j % 4) * 4 + _j / 4;

      const int j_perm =
          (stride >= radix)
            ? ((j / (stride / radix)) / 2 * 2) % radix
            : 0;
      const int i_perm = ((j / stride) / 2 * 2) % radix;
      const int k      = j % stride;

      for (int lane = 0; lane < 32; ++lane) {
        for (int dir = 0; dir < 2; ++dir) {
          const bool forward = (dir == 0);

          int i0 = lane / 4;
          int i1 = lane / 4;
          int j0 = (lane % 4) * 2;
          int j1 = (lane % 4) * 2 + 1;

          if (stage == 0) {
            j0 = f(j0);
            j1 = f(j1);
          }
          if (stage == 2) {
            i0 = f(i0);
            i1 = f(i1);
          }

          i0 ^= i_perm;  i1 ^= i_perm;
          j0 ^= j_perm;  j1 ^= j_perm;

          i0 = (i0 % 4) * 2 + i0 / 4;
          i1 = (i1 % 4) * 2 + i1 / 4;
          j0 = (j0 % 4) * 2 + j0 / 4;
          j1 = (j1 % 4) * 2 + j1 / 4;

          const int index1 =
              (j0 / 2) * (k + stride * (i0 / 2)) +
              stride * (i0 & 1) - stride * (j0 & 1);
          const int index2 =
              (j1 / 2) * (k + stride * (i1 / 2)) +
              stride * (i1 & 1) - stride * (j1 & 1);

          float c1 = std::cos(2.0f * M_PI * index1 / float(4 * stride));
          float c2 = std::cos(2.0f * M_PI * index2 / float(4 * stride));

          if (!forward) {
            if ((i0 + j0) & 1) c1 = -c1;
            if ((i1 + j1) & 1) c2 = -c2;
          }

          const int idx = ((dir * UNIT_ENTRIES + entry) * 32 + lane);
          W[idx] = make_pair<PairT>(c1, c2);
        }
      }

      ++entry;
    }
  }

  for (int stage = 1; stage < 2; ++stage) {
    const int stride = 1 << (stage << 1);

    for (int _j = 0; _j < n / radix; ++_j) {
      if (_j % (1 << (2 - stage)) != 0) continue;

      int j = (_j % 4) * 4 + _j / 4;

      const int j_perm =
          (stride >= radix)
            ? ((j / (stride / radix)) / 2 * 2) % radix
            : 0;
      const int i_perm = 0;
      const int k      = j % stride;

      for (int lane = 0; lane < 32; ++lane) {
        for (int dir = 0; dir < 2; ++dir) {
          const bool forward = (dir == 0);

          int i0 = lane / 4;
          int i1 = lane / 4;
          int j0 = (lane % 4) * 2;
          int j1 = (lane % 4) * 2 + 1;

          i0 = f(i0);
          i1 = f(i1);

          i0 ^= i_perm;  i1 ^= i_perm;
          j0 ^= j_perm;  j1 ^= j_perm;

          i0 = (i0 % 4) * 2 + i0 / 4;
          i1 = (i1 % 4) * 2 + i1 / 4;
          j0 = (j0 % 4) * 2 + j0 / 4;
          j1 = (j1 % 4) * 2 + j1 / 4;

          const int index1 =
              (j0 / 2) * (k + stride * (i0 / 2)) +
              stride * (i0 & 1) - stride * (j0 & 1);
          const int index2 =
              (j1 / 2) * (k + stride * (i1 / 2)) +
              stride * (i1 & 1) - stride * (j1 & 1);

          float c1 = std::cos(2.0f * M_PI * index1 / float(4 * stride));
          float c2 = std::cos(2.0f * M_PI * index2 / float(4 * stride));

          if (!forward) {
            if ((i0 + j0) & 1) c1 = -c1;
            if ((i1 + j1) & 1) c2 = -c2;
          }

          const int idx = ((dir * UNIT_ENTRIES + entry) * 32 + lane);
          W[idx] = make_pair<PairT>(c1, c2);
        }
      }

      ++entry;
    }
  }

  assert(entry == UNIT_ENTRIES);
}

// ============================================================
// Public API
// ============================================================
template <typename T>
inline void ThunderFFTInitialize(int N) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                "ThunderFFTInitialize<T>: supported T are float and half only");

  if (N != 64 && N != 128 && N != 256 && N != 512 &&
      N != 1024 && N != 2048 && N != 4096) {
    throw std::invalid_argument("ThunderFFTInitialize<T>: unsupported FFT size");
  }

  std::lock_guard<std::mutex> lock(TwiddleCache<T>::mutex);
  int device = 0;
  tf_check_cuda(cudaGetDevice(&device), "cudaGetDevice (initialize)");
  auto& cache = TwiddleCache<T>::devices[device];

  // Unit twiddles live in device symbols, so initialize them on each device.
  if constexpr (std::is_same_v<T, half>) {
    if (!cache.unit_initialized) {
      std::vector<half2> h_unit;
      build_unit_twiddle_host(h_unit);

      tf_check_cuda(cudaMemcpyToSymbol(
          detail::d_unit_twiddle_half2,
          h_unit.data(),
          sizeof(half2) * UNIT_W_SIZE),
          "MemcpyToSymbol(d_unit_twiddle_half2)");

      cache.unit_initialized = true;
    }
  }

  if (cache.dW && cache.N_cached == N) return;

  // The caller must complete kernels using this device's cache before replacing it.
  if (cache.dW) {
    cache.N_cached = 0;
    detail::set_twiddle_symbols<T>(nullptr, 0);
    tf_check_cuda(cudaFree(cache.dW), "cudaFree cached twiddle (initialize)");
    cache.dW = nullptr;
  }

  if constexpr (std::is_same_v<T, float>) {
    cache.dW = preprocess_W_float(N);
  } else {
    cache.dW = preprocess_W_half(N);
  }
  detail::set_twiddle_symbols<T>(cache.dW, N);
  cache.N_cached = N;
}

// ============================================================
// Device-side accessors
// ============================================================
template <typename T>
__device__ __forceinline__ const typename TwiddleCache<T>::TwiddleT*
ThunderFFT_get_twiddle();

template <>
__device__ __forceinline__ const float2*
ThunderFFT_get_twiddle<float>() {
  return detail::d_twiddle_f;
}

template <>
__device__ __forceinline__ const half2*
ThunderFFT_get_twiddle<half>() {
  return detail::d_twiddle_h;
}

__device__ __forceinline__ half2*
ThunderFFT_get_unit_twiddle_half2() {
  return detail::d_unit_twiddle_half2;
}

__device__ __forceinline__ float2*
ThunderFFT_get_unit_twiddle_float2() {
  return detail::d_unit_twiddle_float2;
}

template <typename T>
inline void ThunderFFTFinalize() {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, half>,
                "ThunderFFTFinalize<T>: supported T are float and half only");

  std::lock_guard<std::mutex> lock(TwiddleCache<T>::mutex);
  int device = 0;
  tf_check_cuda(cudaGetDevice(&device), "cudaGetDevice (finalize)");
  const auto found = TwiddleCache<T>::devices.find(device);
  if (found != TwiddleCache<T>::devices.end()) {
    found->second.N_cached = 0;
  }
  detail::set_twiddle_symbols<T>(nullptr, 0);

  if (found != TwiddleCache<T>::devices.end()) {
    if (found->second.dW) {
      tf_check_cuda(cudaFree(found->second.dW), "cudaFree cached twiddle (finalize)");
    }
    TwiddleCache<T>::devices.erase(found);
  }
}

} // namespace thunderfft
