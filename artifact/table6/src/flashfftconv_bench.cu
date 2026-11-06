// Table 6 FlashFFTConv driver: times the patched copy of FlashFFTConv's N=256
// convolution kernel (monarch_conv_fft_repeat.patch) as a complex forward FFT.
#include "table6_common.cuh"
#include "monarch_cuda_kernel.h"

namespace {

using complex_half = c10::complex<at::Half>;
constexpr int kSqrtN = 16;
constexpr size_t kShared = 6 * table6::kN * sizeof(half);

struct Matrices {
  const complex_half *dft, *twiddle, *idft, *itwiddle;
};

void launch(const Matrices& m, const complex_half* k_f, const complex_half* x, complex_half* out, int batch, int r) {
  monarch_conv_cuda_kernel_fft_repeat<32, 1, 256, 1, false, 1, 1><<<dim3(batch, 1), dim3(32, 1), kShared>>>(
      x, nullptr, k_f, m.dft, m.twiddle, m.idft, m.itwiddle, out, nullptr, batch, 1, table6::kN, kSqrtN, r);
  T6_CUDA_OK(cudaGetLastError());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const table6::Options o = table6::parse(argc, argv, "usage: table6_flashfftconv [repeats=1000] [trials=3] [batch=65536]");
    table6::print_identity(o, "flashfftconv");

    std::vector<complex_half> cm(4 * table6::kN);
    for (int r = 0; r < kSqrtN; ++r)
      for (int c = 0; c < kSqrtN; ++c) {
        const double f = 2.0 * M_PI * r * c / kSqrtN, t = 2.0 * M_PI * r * c / table6::kN;
        const int i = r * kSqrtN + c;
        cm[i] = complex_half(at::Half(float(std::cos(f))), at::Half(float(-std::sin(f))));
        cm[table6::kN + i] = complex_half(at::Half(float(std::cos(t))), at::Half(float(-std::sin(t))));
        cm[2 * table6::kN + i] = complex_half(at::Half(float(std::cos(f))), at::Half(float(std::sin(f))));
        cm[3 * table6::kN + i] = complex_half(at::Half(float(std::cos(t))), at::Half(float(std::sin(t))));
      }
    complex_half *d_cm = nullptr, *d_kf = nullptr;
    T6_CUDA_OK(cudaMalloc(&d_cm, cm.size() * sizeof(complex_half)));
    T6_CUDA_OK(cudaMemcpy(d_cm, cm.data(), cm.size() * sizeof(complex_half), cudaMemcpyHostToDevice));
    // k_f is read by the kernel but does not affect the FFT.
    T6_CUDA_OK(cudaMalloc(&d_kf, table6::kN * sizeof(complex_half)));
    T6_CUDA_OK(cudaMemset(d_kf, 0, table6::kN * sizeof(complex_half)));
    const Matrices m{d_cm, d_cm + table6::kN, d_cm + 2 * table6::kN, d_cm + 3 * table6::kN};
    T6_CUDA_OK(cudaFuncSetAttribute(monarch_conv_cuda_kernel_fft_repeat<32, 1, 256, 1, false, 1, 1>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize, int(kShared)));

    // Correctness gate on 64 FFTs.
    const int vb = 64;
    const size_t vcount = size_t(vb) * table6::kN;
    std::vector<float> re, im;
    table6::validation_input(re, im, vcount, false);
    std::vector<double> ref_re, ref_im;
    table6::reference_dft(re, im, vb, ref_re, ref_im);
    std::vector<__half2> host(vcount);
    for (size_t i = 0; i < vcount; ++i) host[i] = __floats2half2_rn(re[i], im[i]);
    complex_half *vi = nullptr, *vo = nullptr;
    T6_CUDA_OK(cudaMalloc(&vi, vcount * sizeof(complex_half)));
    T6_CUDA_OK(cudaMalloc(&vo, vcount * sizeof(complex_half)));
    T6_CUDA_OK(cudaMemcpy(vi, host.data(), vcount * sizeof(complex_half), cudaMemcpyHostToDevice));
    T6_CUDA_OK(cudaMemset(vo, 0xff, vcount * sizeof(complex_half)));
    launch(m, d_kf, vi, vo, vb, 1);
    T6_CUDA_OK(cudaDeviceSynchronize());
    std::vector<__half2> actual(vcount);
    T6_CUDA_OK(cudaMemcpy(actual.data(), vo, vcount * sizeof(__half2), cudaMemcpyDeviceToHost));
    T6_CUDA_OK(cudaFree(vi)); T6_CUDA_OK(cudaFree(vo));
    const double rel = table6::rel_l2(actual, ref_re, ref_im);
    std::fprintf(stderr, "FlashFFTConv single_fft_relL2=%.6g\n", rel);
    if (!std::isfinite(rel) || rel > table6::kRelL2Limit)
      throw std::runtime_error("FlashFFTConv correctness gate failed relL2=" + std::to_string(rel));

    // Timed input, as in Table 3.
    const size_t count = size_t(o.batch) * table6::kN;
    std::vector<__half2> timed(count);
    for (size_t i = 0; i < count; ++i) timed[i] = __floats2half2_rn(table6::timed_value(i), 0.0f);
    complex_half *in = nullptr, *out = nullptr;
    T6_CUDA_OK(cudaMalloc(&in, count * sizeof(complex_half)));
    T6_CUDA_OK(cudaMalloc(&out, count * sizeof(complex_half)));
    T6_CUDA_OK(cudaMemcpy(in, timed.data(), count * sizeof(complex_half), cudaMemcpyHostToDevice));
    std::printf("%s\n", table6::header());
    table6::Row row{"complex", "register_fragments", "out_of_place", "flashfftconv", "conv_kernel",
                    1, 32, kShared};
    table6::time_and_print(o, rel, row, [&](int r) { launch(m, d_kf, in, out, o.batch, r); },
                           [&] { launch(m, d_kf, in, out, o.batch, 1); });
    T6_CUDA_OK(cudaFree(in)); T6_CUDA_OK(cudaFree(out));
    T6_CUDA_OK(cudaFree(d_cm)); T6_CUDA_OK(cudaFree(d_kf));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
