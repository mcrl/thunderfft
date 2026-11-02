# ThunderFFT

CUDA FFT kernels and convolution experiments using NVIDIA Tensor Cores.

## Repository layout

- `include/thunderfft/`: header-only library; include `<thunderfft/thunderfft.cuh>`.
- `sample/`: CMake sample programs and the active FFT benchmark.
- `bench/`: earlier benchmark sources, currently outside the CMake build.

## Build

Use a CUDA toolkit with a C++17 host compiler and CMake 3.18 or newer. The default
CUDA architecture is SM 80 (Ampere); set `THUNDERFFT_ARCHS` for another supported
target. The repository migration is checked locally with CUDA 12.9, GCC 9.4 and
CMake 4.1.

From the repository root:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80
cmake --build build --target benchmark thff_conv1d_fullfft4096 -j
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

Paper reproduction scripts, the Table 5 benchmark and historical local CSV
outputs are maintained on the `artifact` branch. Switch branches to use them:

```bash
git switch artifact
```

That branch includes the experiment instructions under `artifact/figure5/` and
`artifact/table5/`. Numerical correctness and cross-implementation equivalence
are still being investigated.

## Git history

The original `main` history is preserved. Existing local work was saved in a
separate commit before the directory moves, and the moves were committed without
content changes. Build outputs are excluded from version control.
