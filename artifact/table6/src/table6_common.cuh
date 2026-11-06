// Shared harness for the Table 6 drivers: inputs, FP64 reference, and the
// R/2R timing of sample/bench/paper_fft.cu.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#define T6_CUDA_OK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) \
  throw std::runtime_error(std::string(#expr) + ": " + cudaGetErrorString(e_)); } while (0)

namespace table6 {

constexpr int kDevice = 0;
constexpr int kN = 256;
constexpr double kRelL2Limit = 5e-3;

struct Options {
  int repeats = 1000, trials = 3, batch = 65536;
};

inline Options parse(int argc, char** argv, const char* usage) {
  Options o;
  std::vector<std::string> args(argv + 1, argv + argc);
  if (args.size() > 3) throw std::runtime_error(usage);
  if (args.size() > 0) o.repeats = std::stoi(args[0]);
  if (args.size() > 1) o.trials = std::stoi(args[1]);
  if (args.size() > 2) o.batch = std::stoi(args[2]);
  if (o.repeats <= 0 || o.trials <= 0 || o.batch <= 0) throw std::runtime_error(usage);
  T6_CUDA_OK(cudaSetDevice(kDevice));
  return o;
}

// Same inputs as paper_fft.cu.
inline void validation_input(std::vector<float>& re, std::vector<float>& im, size_t count, bool real_only) {
  re.resize(count); im.resize(count);
  for (size_t i = 0; i < count; ++i) {
    unsigned z = unsigned(i) * 1664525u + 1013904223u;
    float x = (float(int(z & 65535u)) - 32768.0f) / 32768.0f / std::sqrt(float(kN));
    z = z * 1664525u + 1013904223u;
    float y = (float(int(z & 65535u)) - 32768.0f) / 32768.0f / std::sqrt(float(kN));
    re[i] = __half2float(__float2half_rn(x));
    im[i] = real_only ? 0.0f : __half2float(__float2half_rn(y));
  }
}

inline float timed_value(size_t i) {
  return std::sin(float(6.2831853071795864769 * double(i % kN) / kN)) / std::sqrt(float(kN));
}

// Forward DFT in double precision.
inline void reference_dft(const std::vector<float>& re, const std::vector<float>& im, size_t batch,
                          std::vector<double>& out_re, std::vector<double>& out_im) {
  std::vector<double> c(kN), s(kN);
  for (int k = 0; k < kN; ++k) {
    c[k] = std::cos(2.0 * M_PI * k / kN);
    s[k] = -std::sin(2.0 * M_PI * k / kN);
  }
  out_re.assign(batch * kN, 0.0); out_im.assign(batch * kN, 0.0);
  for (size_t b = 0; b < batch; ++b)
    for (int k = 0; k < kN; ++k) {
      double sr = 0, si = 0;
      for (int j = 0; j < kN; ++j) {
        const int t = (j * k) % kN;
        const double xr = re[b * kN + j], xi = im[b * kN + j];
        sr += xr * c[t] - xi * s[t];
        si += xr * s[t] + xi * c[t];
      }
      out_re[b * kN + k] = sr; out_im[b * kN + k] = si;
    }
}

inline double rel_l2(const std::vector<__half2>& actual, const std::vector<double>& ref_re,
                     const std::vector<double>& ref_im) {
  double se = 0, sr = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double x = __half2float(actual[i].x), y = __half2float(actual[i].y);
    if (!std::isfinite(x) || !std::isfinite(y)) return INFINITY;
    se += (x - ref_re[i]) * (x - ref_re[i]) + (y - ref_im[i]) * (y - ref_im[i]);
    sr += ref_re[i] * ref_re[i] + ref_im[i] * ref_im[i];
  }
  return std::sqrt(se / sr);
}

inline void print_identity(const Options& o, const char* library) {
  cudaDeviceProp prop; T6_CUDA_OK(cudaGetDeviceProperties(&prop, kDevice));
  int driver = 0, runtime = 0;
  T6_CUDA_OK(cudaDriverGetVersion(&driver)); T6_CUDA_OK(cudaRuntimeGetVersion(&runtime));
  std::fprintf(stderr, "compiled_CUDART=%d runtime=%d driver=%d library=%s extra_R_warmups=3\n",
               CUDART_VERSION, runtime, driver, library);
  std::fprintf(stderr, "GPU=%s compute=%d.%d batch=%d R=%d trials=%d single_fft_relL2_limit=%g\n",
               prop.name, prop.major, prop.minor, o.batch, o.repeats, o.trials, kRelL2Limit);
  std::fputs("GPU_UUID=GPU-", stderr);
  for (int i = 0; i < 16; ++i) {
    if (i == 4 || i == 6 || i == 8 || i == 10) std::fputc('-', stderr);
    std::fprintf(stderr, "%02x", static_cast<unsigned char>(prop.uuid.bytes[i]));
  }
  std::fputc('\n', stderr);
}

inline const char* header() {
  return "N,precision,input,placement,batch,inside_repeats,trial,t_R_us,t_2R_us,compute_us,"
         "e2e_us,e2e_mode,single_fft_relL2,shared_bytes,library,candidate,ffts_per_block,threads_per_block";
}

template <class Launch> double elapsed_us(Launch launch, int runs) {
  cudaEvent_t begin, end;
  T6_CUDA_OK(cudaEventCreate(&begin)); T6_CUDA_OK(cudaEventCreate(&end));
  T6_CUDA_OK(cudaEventRecord(begin));
  for (int i = 0; i < runs; ++i) launch();
  T6_CUDA_OK(cudaEventRecord(end)); T6_CUDA_OK(cudaEventSynchronize(end));
  float ms = 0; T6_CUDA_OK(cudaEventElapsedTime(&ms, begin, end));
  T6_CUDA_OK(cudaEventDestroy(begin)); T6_CUDA_OK(cudaEventDestroy(end));
  return double(ms) * 1000.0 / runs;
}

struct Row {
  const char* input;
  const char* placement;
  const char* e2e_mode;
  const char* library;
  std::string candidate;
  int ffts_per_block, threads_per_block;
  size_t shared_bytes;
};

// Same schedule as paper_fft.cu: compute_us = (t_2R - t_R) / R.
template <class Repeated, class Once>
void time_and_print(const Options& o, double rel, const Row& row, Repeated repeated, Once once) {
  for (int w = 0; w < 3; ++w) { repeated(o.repeats); T6_CUDA_OK(cudaDeviceSynchronize()); }
  for (int t = 0; t < o.trials; ++t) {
    repeated(o.repeats); T6_CUDA_OK(cudaDeviceSynchronize());
    const double t_r = elapsed_us([&] { repeated(o.repeats); }, 1);
    repeated(2 * o.repeats); T6_CUDA_OK(cudaDeviceSynchronize());
    const double t_2r = elapsed_us([&] { repeated(2 * o.repeats); }, 1);
    if (!(std::isfinite(t_r) && std::isfinite(t_2r) && t_2r > t_r))
      throw std::runtime_error("nonpositive/nonfinite differential timing");
    once(); T6_CUDA_OK(cudaDeviceSynchronize());
    const double e2e = elapsed_us(once, 10);
    std::printf("%d,fp16,%s,%s,%d,%d,%d,%.9g,%.9g,%.9g,%.9g,%s,%.12g,%zu,%s,%s,%d,%d\n", kN, row.input,
                row.placement, o.batch, o.repeats, t, t_r, t_2r, (t_2r - t_r) / o.repeats, e2e, row.e2e_mode,
                rel, row.shared_bytes, row.library, row.candidate.c_str(), row.ffts_per_block,
                row.threads_per_block);
    std::fflush(stdout);
  }
}

}  // namespace table6
