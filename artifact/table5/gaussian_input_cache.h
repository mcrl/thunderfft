#ifndef THUNDERFFT_TABLE5_GAUSSIAN_INPUT_CACHE_H
#define THUNDERFFT_TABLE5_GAUSSIAN_INPUT_CACHE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace table5 {

// Preserve the unrounded Gaussian samples and the double normalization factor.
// Caching already-normalized FP32 values would round before alpha is applied and
// would not reproduce generate_input_fp32 for arbitrary alpha values.
class GaussianInputCache {
public:
    GaussianInputCache(int n, int batch, std::uint64_t seed)
        : n_(n), base_(static_cast<std::size_t>(n) * batch), scale_(batch) {
        std::normal_distribution<float> normal(0.0f, 1.0f);
        for (int b = 0; b < batch; ++b) {
            std::mt19937_64 rng(seed + static_cast<std::uint64_t>(b));
            double norm2 = 0.0;
            for (int j = 0; j < n; ++j) {
                const float re = normal(rng);
                const float im = normal(rng);
                base_[static_cast<std::size_t>(b) * n + j] = {re, im};
                norm2 += static_cast<double>(re) * re + static_cast<double>(im) * im;
            }
            if (norm2 <= 0.0) norm2 = 1.0;
            const double target = std::sqrt(static_cast<double>(n));
            scale_[b] = target / std::sqrt(norm2);
        }
    }

    template <typename Complex>
    void fill(std::vector<Complex>& output, float alpha) const {
        output.resize(base_.size());
        for (std::size_t b = 0; b < scale_.size(); ++b) {
            const double scale = scale_[b] * static_cast<double>(alpha);
            for (int j = 0; j < n_; ++j) {
                const std::size_t i = b * n_ + j;
                output[i].x = static_cast<float>(base_[i].x * scale);
                output[i].y = static_cast<float>(base_[i].y * scale);
            }
        }
    }

private:
    struct ComplexValue { float x, y; };
    int n_;
    std::vector<ComplexValue> base_;
    std::vector<double> scale_;
};

}  // namespace table5

#endif  // THUNDERFFT_TABLE5_GAUSSIAN_INPUT_CACHE_H
