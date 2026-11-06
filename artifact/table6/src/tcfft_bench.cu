// Table 6 tcFFT driver: times tcFFT's layer_256_0_A100 with a repeat loop
// (layer_256_0_A100_repeat.patch). tcFFT expects digit-reversed input.
#include "table6_common.cuh"
#include <cstring>
#include "tcfft_half.cu"

namespace {

constexpr int kCont = 32, kWarps = 8;
constexpr size_t kShared = size_t(table6::kN) * sizeof(half2) * kCont;

int digit_reversed(int i) { return (i % 16) * 16 + i / 16; }

void upload_permuted(half2* device, const std::vector<float>& re, const std::vector<float>& im, size_t batch) {
  std::vector<half2> host(batch * table6::kN);
  for (size_t b = 0; b < batch; ++b)
    for (int i = 0; i < table6::kN; ++i) {
      const size_t src = b * table6::kN + digit_reversed(i);
      host[b * table6::kN + i] = __floats2half2_rn(re[src], im[src]);
    }
  T6_CUDA_OK(cudaMemcpy(device, host.data(), host.size() * sizeof(half2), cudaMemcpyHostToDevice));
}

std::vector<__half2> download(const half2* device, size_t count) {
  std::vector<__half2> host(count);
  T6_CUDA_OK(cudaMemcpy(host.data(), device, count * sizeof(half2), cudaMemcpyDeviceToHost));
  return host;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const table6::Options o = table6::parse(argc, argv, "usage: table6_tcfft [repeats=1000] [trials=3] [batch=65536]");
    if (o.batch % kCont) throw std::runtime_error("batch must be a multiple of 32 (FFTs per tcFFT block)");
    table6::print_identity(o, "tcfft");
    std::printf("%s\n", table6::header());

    cudaDeviceProp prop; T6_CUDA_OK(cudaGetDeviceProperties(&prop, table6::kDevice));
    if (prop.major < 8) throw std::runtime_error("this driver times tcFFT's A100 (sm_80+) kernel");
    auto repeat_kernel = layer_256_0_A100_repeat<kCont, kWarps>;
    auto base_kernel = layer_256_0_A100<kCont, kWarps>;
    T6_CUDA_OK(cudaFuncSetAttribute(repeat_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kShared)));
    T6_CUDA_OK(cudaFuncSetAttribute(base_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(kShared)));
    const dim3 threads(32, kWarps);

    // Correctness gate on 64 FFTs.
    const int vb = 64;
    const size_t vcount = size_t(vb) * table6::kN;
    tcfftHandle vplan;
    tcfftCreate(&vplan, table6::kN, vb);
    if (vplan.n_mergings != 1 || vplan.mergings[0] != 0 || vplan.layer_0[0] != static_cast<void (*)(half2 *, half *, half *)>(layer_256_0<kCont, kWarps>))
      throw std::runtime_error("unexpected tcFFT plan for N=256");
    vplan.layer_0[0] = base_kernel;
    std::vector<float> re, im;
    table6::validation_input(re, im, vcount, false);
    std::vector<double> ref_re, ref_im;
    table6::reference_dft(re, im, vb, ref_re, ref_im);
    half2* v = nullptr;
    T6_CUDA_OK(cudaMalloc(&v, vcount * sizeof(half2)));
    upload_permuted(v, re, im, vb);
    tcfftExec(vplan, reinterpret_cast<half*>(v));
    T6_CUDA_OK(cudaGetLastError()); T6_CUDA_OK(cudaDeviceSynchronize());
    const auto library_out = download(v, vcount);
    upload_permuted(v, re, im, vb);
    repeat_kernel<<<vb / kCont, threads, kShared>>>(v, vplan.F_real, vplan.F_imag, 1);
    T6_CUDA_OK(cudaGetLastError()); T6_CUDA_OK(cudaDeviceSynchronize());
    const auto repeat_out = download(v, vcount);
    T6_CUDA_OK(cudaFree(v));
    const double rel = std::max(table6::rel_l2(library_out, ref_re, ref_im), table6::rel_l2(repeat_out, ref_re, ref_im));
    const bool identical = std::memcmp(library_out.data(), repeat_out.data(), vcount * sizeof(half2)) == 0;
    std::fprintf(stderr, "tcFFT single_fft_relL2=%.6g repeat_kernel_bitwise_equal_to_tcfftExec=%d\n", rel, int(identical));
    if (!identical) throw std::runtime_error("layer_256_0_A100_repeat(R=1) differs from tcfftExec");
    if (!std::isfinite(rel) || rel > table6::kRelL2Limit)
      throw std::runtime_error("tcFFT correctness gate failed relL2=" + std::to_string(rel));

    // Timed input, as in Table 3.
    tcfftHandle plan;
    tcfftCreate(&plan, table6::kN, o.batch);
    plan.layer_0[0] = base_kernel;
    const size_t count = size_t(o.batch) * table6::kN;
    std::vector<float> tre(count), tim(count, 0.0f);
    for (size_t i = 0; i < count; ++i) tre[i] = table6::timed_value(i);
    half2* data = nullptr;
    T6_CUDA_OK(cudaMalloc(&data, count * sizeof(half2)));
    upload_permuted(data, tre, tim, o.batch);
    const unsigned blocks = unsigned(size_t(o.batch) * table6::kN / kCont / table6::kN);
    table6::Row row{"complex", "register_fragments", "in_place", "tcfft", "layer_256_0_A100", kCont, 32 * kWarps, kShared};
    table6::time_and_print(
        o, rel, row,
        [&](int r) {
          repeat_kernel<<<blocks, threads, kShared>>>(data, plan.F_real, plan.F_imag, r);
          T6_CUDA_OK(cudaGetLastError());
        },
        [&] {
          base_kernel<<<blocks, threads, kShared>>>(data, plan.F_real, plan.F_imag);
          T6_CUDA_OK(cudaGetLastError());
        });
    T6_CUDA_OK(cudaFree(data));
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "ERROR: %s\n", e.what());
    return 1;
  }
}
