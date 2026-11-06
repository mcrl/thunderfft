// Minimal stand-in for <torch/extension.h> used only by the Table 6
// FlashFFTConv driver. FlashFFTConv's device headers need c10::complex and
// at::Half, which are header-only; the PyTorch extension machinery (Python,
// pybind11, libtorch) is not needed for a standalone CUDA executable.
#pragma once
#include <c10/util/Half.h>
#include <c10/util/complex.h>

namespace at {
using c10::Half;
}
