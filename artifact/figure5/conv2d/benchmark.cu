// Figure 5 driver: one library per process over a list of filter sizes; each size is checked
// against a direct FP64 reference before it is timed.
//   thff_conv2d N f[,f...] cufft|cufftdx|cudnn|thunderfft [warmup=5] [runs=20] [trials=1] [device=0]
// x (N x N) and h (f x f) are U[0,1] from minstd_rand seeds 42 and 42 + f (real parts of the complex
// sequence used by the earlier evaluation).
#include "common.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <string>

namespace figure5 {
namespace {

constexpr double kTolerance = 5e-3;

std::vector<std::pair<int, int>> sample_points(int M, int f, int count) {
    std::set<std::pair<int, int>> points;
    const int valid = 64 - f + 1;
    std::vector<int> edges{0, 1, M / 2, M - 2, M - 1};
    for (int k = 1; k <= 3; ++k) {
        edges.push_back(k * valid - 1);
        edges.push_back(k * valid);
    }
    const int last_tile = ((M - 1) / valid) * valid;
    edges.push_back(last_tile - 1);
    edges.push_back(last_tile);
    for (int r : edges)
        for (int c : edges)
            if (r >= 0 && r < M && c >= 0 && c < M) points.insert({r, c});
    std::minstd_rand rng(7 + f);
    std::uniform_int_distribution<int> pick(0, M - 1);
    while (int(points.size()) < count + int(edges.size() * edges.size()) && int64_t(points.size()) < int64_t(M) * M)
        points.insert({pick(rng), pick(rng)});
    return {points.begin(), points.end()};
}

double rel_l2(const Method& method, const Problem& p, const std::vector<float>& x, const std::vector<float>& h,
              const std::vector<std::pair<int, int>>& points) {
    std::vector<float> out(size_t(p.M) * p.M);
    FIG5_CUDA(cudaMemcpy2D(out.data(), sizeof(float) * p.M, method.output(), sizeof(float) * method.output_pitch(),
                           sizeof(float) * p.M, p.M, cudaMemcpyDeviceToHost));
    for (float v : out)
        if (!std::isfinite(v)) return NAN;
    double num = 0.0, den = 0.0;
    for (const auto& [i, j] : points) {
        double ref = 0.0;
        for (int a = 0; a < p.f; ++a)
            for (int b = 0; b < p.f; ++b) ref += double(x[size_t(i + a) * p.N + j + b]) * h[a * p.f + b];
        const double diff = out[size_t(i) * p.M + j] - ref;
        num += diff * diff;
        den += ref * ref;
    }
    return std::sqrt(num / den);
}

std::unique_ptr<Method> make(const std::string& library, const Problem& p) {
    if (library == "cufft") return make_cufft(p);
    if (library == "thunderfft") return make_thunderfft_c2c(p);
#ifdef THFF_WITH_CUFFTDX
    if (library == "cufftdx") return make_cufftdx_c2c(p);
#endif
#ifdef THFF_WITH_CUDNN
    if (library == "cudnn") return make_cudnn(p);
#endif
    return nullptr;
}

int run(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " N f[,f...] cufft|cufftdx|cudnn|thunderfft [warmup=5] [runs=20] [trials=1] [device=0]\n";
        return 2;
    }
    const int N = std::atoi(argv[1]);
    std::vector<int> filters;
    for (const char* s = argv[2]; *s;) {
        filters.push_back(std::atoi(s));
        while (*s && *s != ',') ++s;
        if (*s) ++s;
    }
    const std::string library = argv[3];
    const int warmup = argc >= 5 ? std::atoi(argv[4]) : 5;
    const int runs = argc >= 6 ? std::atoi(argv[5]) : 20;
    const int trials = argc >= 7 ? std::atoi(argv[6]) : 1;
    const int device = argc >= 8 ? std::atoi(argv[7]) : 0;
    const bool known = library == "cufft" || library == "cufftdx" || library == "cudnn" || library == "thunderfft";
    const bool bad_filter = filters.empty() || std::any_of(filters.begin(), filters.end(), [N](int f) {
        return f < 1 || f > 63 || f > N;
    });
    if (!known || bad_filter || N < 64 || N % 2 || warmup < 0 || runs <= 0 || trials <= 0) {
        std::cerr << "invalid arguments\n";
        return 2;
    }

    FIG5_CUDA(cudaSetDevice(device));
    cudaDeviceProp prop{};
    FIG5_CUDA(cudaGetDeviceProperties(&prop, device));
    char uuid[48];
    int at = std::snprintf(uuid, sizeof(uuid), "GPU-");
    for (int i = 0; i < 16; ++i) {
        at += std::snprintf(uuid + at, sizeof(uuid) - at, "%02x", (unsigned char)prop.uuid.bytes[i]);
        if (i == 3 || i == 5 || i == 7 || i == 9) uuid[at++] = '-', uuid[at] = 0;
    }
    std::cerr << "GPU=" << prop.name << " compute=" << prop.major << "." << prop.minor << " batch=1 seed=42\n"
              << "GPU_UUID=" << uuid << "\n";

    std::vector<float> x(size_t(N) * N);
    std::minstd_rand rng(42);
    for (auto& v : x) v = make_float2(float(rng()) / float(rng.max()), float(rng()) / float(rng.max())).x;

    Problem p;
    p.N = N;
    FIG5_CUDA(cudaStreamCreateWithFlags(&p.stream, cudaStreamNonBlocking));
    DeviceBuffer<float> d_input(x.size());
    FIG5_CUDA(cudaMemcpy(d_input.data, x.data(), sizeof(float) * x.size(), cudaMemcpyHostToDevice));
    p.d_input = d_input.data;
    cudaEvent_t start, stop;
    FIG5_CUDA(cudaEventCreate(&start));
    FIG5_CUDA(cudaEventCreate(&stop));

    for (const int f : filters) {
        std::cout << "Filter size: " << f << "\n";
        std::vector<float> h(size_t(f) * f);
        std::minstd_rand filter_rng(42 + f);
        for (auto& v : h) v = float(filter_rng()) / float(filter_rng.max());
        DeviceBuffer<float> d_filter(h.size());
        FIG5_CUDA(cudaMemcpy(d_filter.data, h.data(), sizeof(float) * h.size(), cudaMemcpyHostToDevice));
        p.f = f;
        p.M = N - f + 1;
        p.d_filter = d_filter.data;

        std::unique_ptr<Method> method = make(library, p);
        if (!method) {
            std::cerr << "built without " << library << "\n";
            return 78;
        }
        if (!method->details().empty()) std::cout << method->details() << "\n";
        method->run();
        FIG5_CUDA(cudaStreamSynchronize(p.stream));
        const auto points = sample_points(p.M, f, 2048);
        const double rel = rel_l2(*method, p, x, h, points);
        const bool passed = std::isfinite(rel) && rel <= kTolerance;
        std::cout << std::setprecision(6) << "Validation 2D (" << library << ") vs direct reference: L2_rel=" << rel
                  << " samples=" << points.size() << "\n"
                  << "Convolution validation: " << (passed ? "PASS" : "FAIL") << " (relative L2 limit=" << kTolerance
                  << ")\n";
        if (!passed) return EXIT_FAILURE;

        for (int i = 0; i < warmup; ++i) method->run();
        for (int t = 0; t < trials; ++t) {
            FIG5_CUDA(cudaEventRecord(start, p.stream));
            for (int r = 0; r < runs; ++r) method->run();
            FIG5_CUDA(cudaEventRecord(stop, p.stream));
            FIG5_CUDA(cudaEventSynchronize(stop));
            float ms = 0.0f;
            FIG5_CUDA(cudaEventElapsedTime(&ms, start, stop));
            std::cout << std::fixed << std::setprecision(6) << library << " 2D conv (N=" << N << ", f=" << f
                      << ", filter_fft_included=" << int(method->filter_fft_included()) << ", trial=" << t
                      << ") time: " << ms / runs << " ms\n" << std::defaultfloat;
        }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}

}  // namespace
}  // namespace figure5

int main(int argc, char** argv) {
    try {
        return figure5::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << std::endl;
        return 1;
    }
}
