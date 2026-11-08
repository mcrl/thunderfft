// Full-image cuFFT: R2C of the input and of the flipped, zero-padded filter, product, C2R.
#include "common.cuh"

namespace figure5 {
namespace {

__global__ void pad_flip_filter(const float* __restrict__ filter, float* __restrict__ out, int N, int f, float scale) {
    const size_t total = size_t(N) * N;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < total; k += size_t(gridDim.x) * blockDim.x) {
        const int r = int(k / N), c = int(k % N), i = (N - r) % N, j = (N - c) % N;
        out[k] = i < f && j < f ? filter[i * f + j] * scale : 0.0f;
    }
}

__global__ void pointwise_multiply(cufftComplex* __restrict__ a, const cufftComplex* __restrict__ b, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
        const float ax = a[i].x, ay = a[i].y, bx = b[i].x, by = b[i].y;
        a[i].x = ax * bx - ay * by;
        a[i].y = ax * by + ay * bx;
    }
}

class CufftMethod final : public Method {
public:
    explicit CufftMethod(const Problem& p)
        : problem_(p), spectrum_count_(size_t(p.N) * (p.N / 2 + 1)), filter_(size_t(p.N) * p.N),
          result_(size_t(p.N) * p.N), input_fft_(spectrum_count_), filter_fft_(spectrum_count_) {
        FIG5_CUFFT(cufftPlan2d(&r2c_, p.N, p.N, CUFFT_R2C));
        FIG5_CUFFT(cufftPlan2d(&c2r_, p.N, p.N, CUFFT_C2R));
        FIG5_CUFFT(cufftSetStream(r2c_, p.stream));
        FIG5_CUFFT(cufftSetStream(c2r_, p.stream));
    }
    ~CufftMethod() override {
        cufftDestroy(r2c_);
        cufftDestroy(c2r_);
    }
    void run() override {
        const Problem& p = problem_;
        pad_flip_filter<<<4096, 256, 0, p.stream>>>(p.d_filter, filter_.data, p.N, p.f, 1.0f / (float(p.N) * float(p.N)));
        FIG5_CUFFT(cufftExecR2C(r2c_, const_cast<float*>(p.d_input), input_fft_.data));
        FIG5_CUFFT(cufftExecR2C(r2c_, filter_.data, filter_fft_.data));
        pointwise_multiply<<<4096, 256, 0, p.stream>>>(input_fft_.data, filter_fft_.data, spectrum_count_);
        FIG5_CUDA(cudaGetLastError());
        FIG5_CUFFT(cufftExecC2R(c2r_, input_fft_.data, result_.data));
    }
    const float* output() const override { return result_.data; }
    size_t output_pitch() const override { return size_t(problem_.N); }
    bool filter_fft_included() const override { return true; }

private:
    Problem problem_;
    size_t spectrum_count_;
    DeviceBuffer<float> filter_, result_;
    DeviceBuffer<cufftComplex> input_fft_, filter_fft_;
    cufftHandle r2c_ = 0, c2r_ = 0;
};

}  // namespace

std::unique_ptr<Method> make_cufft(const Problem& p) { return std::make_unique<CufftMethod>(p); }

}  // namespace figure5
