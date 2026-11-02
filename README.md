# ThunderFFT

CUDA FFT kernels and convolution experiments using NVIDIA Tensor Cores.

## Repository layout

- `include/thunderfft/`: header-only library; include `<thunderfft/thunderfft.cuh>`.
- `sample/`: CMake sample programs and the active FFT benchmark.
- `bench/`: earlier benchmark sources, currently outside the CMake build.
- `artifact/figure5/`: 1D convolution benchmark driver and existing local CSV output.
- `artifact/table5/`: precision/range benchmark, layout sweeps and existing local CSV outputs.
- `artifact/flashfftconv/`: notes on the external FlashFFTConv dependency.

## Build

Use a CUDA toolkit with a C++17 host compiler and CMake 3.18 or newer. The default
CUDA architecture is SM 80 (Ampere); set `THUNDERFFT_ARCHS` for another supported
target. The repository migration is checked locally with CUDA 12.9, GCC 9.4 and
CMake 4.1.

From the repository root:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80
cmake --build build --target benchmark benchmark_table5_precision thff_conv1d_fullfft4096 -j
```

If `nvcc` is not on `PATH`, pass
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` and
`-DCUDAToolkit_ROOT=/path/to/cuda` when configuring.

The 2D convolution target additionally requires NVIDIA MathDX, including its
cuFFTDx and bundled CUTLASS/CuTe headers. Configure with
`-DMATHDX_ROOT=/path/to/nvidia/mathdx/25.06` before building `thff_conv` or all
targets. The native `thff_conv1d_compare` target is available when the cuDNN
library and development headers are installed.

## Artifact experiments

See [Figure 5](artifact/figure5/README.md) and [Table 5](artifact/table5/README.md)
for their commands. Figure 5 additionally needs a Python environment with
PyTorch and a separately installed FlashFFTConv clone.

The included CSVs are historical local outputs preserved during repository
preparation. They have not been revalidated as paper reference results. Numerical
correctness and cross-implementation equivalence are still being investigated;
this repository reorganization does not resolve the previously identified
correctness issues.

## Git history

The original `main` history is preserved. Existing local work was saved in a
separate commit before the directory moves, and the moves were committed without
content changes. Build outputs are excluded from version control.
