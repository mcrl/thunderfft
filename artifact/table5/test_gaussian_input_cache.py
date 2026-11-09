#!/usr/bin/env python3
"""Compare cached sweep inputs bit for bit with the unchanged native generator."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


class GaussianInputCacheChecks(unittest.TestCase):
    @unittest.skipUnless(shutil.which("g++"), "host C++ compiler is required")
    def test_all_paper_exponents_match_original_generator_exactly(self):
        # Compile the actual accuracy-path generator, with only CUDA's float2
        # declaration replaced. This also catches later drift between the two
        # input paths, without requiring a GPU or CUDA toolkit for the test.
        source = (HERE / "main.cu").read_text()
        generator = source[source.index("static void generate_input_fp32("):
                           source.index("static void pack_half2(")]
        program = r'''
#include "gaussian_input_cache.h"
#include <cstring>
#include <iostream>
#include <limits>

struct float2 { float x, y; };
float2 make_float2(float x, float y) { return {x, y}; }
enum class InputMode { Gaussian, AllOnes, Impulse };
''' + generator + r'''
int main() {
    std::vector<float> alphas;
    for (int exponent = 0; exponent <= 127; ++exponent)
        alphas.push_back(std::ldexp(1.0f, exponent));
    // Non-powers of two expose premature FP32 rounding; tiny alphas exercise
    // subnormal and signed-zero outputs that a numerical tolerance would miss.
    for (float alpha : {1.234567f, 3.125f, std::ldexp(1.0f, -149),
                        std::ldexp(1.0f, -126), std::ldexp(1.0f, -20)})
        alphas.push_back(alpha);
    std::size_t compared = 0, premature_rounding_differences = 0;
    for (int n : {64, 128, 256, 512, 1024, 2048, 4096}) {
        for (std::uint64_t seed : {std::uint64_t{0}, std::uint64_t{1234},
                                  std::numeric_limits<std::uint64_t>::max() - 1}) {
            constexpr int batch = 3;
            table5::GaussianInputCache cache(n, batch, seed);
            std::vector<float2> original, cached, normalized;
            generate_input_fp32(normalized, n, batch, InputMode::Gaussian, 1.0f, seed);
            for (float alpha : alphas) {
                generate_input_fp32(original, n, batch, InputMode::Gaussian, alpha, seed);
                cache.fill(cached, alpha);
                if (cached.size() != original.size() ||
                    std::memcmp(cached.data(), original.data(), original.size() * sizeof(float2))) {
                    std::cerr << "Input mismatch: N=" << n << " seed=" << seed
                              << " alpha=" << alpha << '\n';
                    return 1;
                }
                if (alpha == 1.234567f) {
                    for (std::size_t i = 0; i < original.size(); ++i) {
                        const float rounded = normalized[i].x * alpha;
                        premature_rounding_differences +=
                            std::memcmp(&rounded, &original[i].x, sizeof(float)) != 0;
                    }
                }
                compared += original.size();
            }
            // A caller may reuse a differently sized output vector.
            cached.resize(1);
            cache.fill(cached, alphas.back());
            if (cached.size() != original.size() ||
                std::memcmp(cached.data(), original.data(), original.size() * sizeof(float2)))
                return 2;
        }
    }
    if (premature_rounding_differences == 0) return 3;
    std::cout << "Compared " << compared << " complex inputs bit for bit; "
              << premature_rounding_differences << " premature-rounding differences detected\n";
}
'''
        with tempfile.TemporaryDirectory(prefix="table5-input-cache-") as directory:
            root = Path(directory)
            (root / "check.cpp").write_text(program)
            build = subprocess.run(
                ["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror",
                 "-I", str(HERE), str(root / "check.cpp"), "-o", str(root / "check")],
                capture_output=True, text=True, timeout=60)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            result = subprocess.run([str(root / "check")], capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Compared 9729216 complex inputs bit for bit", result.stdout)


if __name__ == "__main__":
    unittest.main()
