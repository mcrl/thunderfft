# Figure 5: 2D convolution

Status: runnable with partial collection and offline assembly. Measures full-image
cuFFT and tiled ThunderFFT without optional dependencies; also measures cuFFTDx
when `MATHDX_ROOT` is configured and cuDNN when installed. All paths use the
native direct-reference correctness gate.

Paper p. 10, Sections 5.4/6.6: H100; cuDNN's automatically selected best
algorithm, full-image cuFFT, tiled cuFFTDx and tiled ThunderFFT. FFT tile size
is 64 and each tile produces `64-f+1` valid outputs per dimension. All FFT
results include filter FFT construction. The earlier H100 evaluation used a
16384 x 16384 image and odd `f=3…53`, which the paper profile follows.

## Run

```bash
MATHDX_ROOT=/path/to/nvidia/mathdx/25.06 THUNDERFFT_ARCHS=90 ./run.sh figure5 --profile paper  # H100
MATHDX_ROOT=/path/to/nvidia/mathdx/25.06 ./run.sh figure5 --profile smoke                     # 2048^2, f=3,33,53
```

cuFFTDx is compiled for the first `THUNDERFFT_ARCHS` entry. Overrides:
`THUNDERFFT_FIGURE5_FILTERS`, `_TRIALS`, `_N` (smoke only), `_BACKENDS` and
`_LIBRARIES=auto` (all available backends). A comma-separated `_LIBRARIES` subset
of `cuFFT,cuFFTDx,ThunderFFT,cuDNN` selects independent measurement parts.
Explicit requests for an unavailable backend fail.

## Operation

Valid 2D cross-correlation `y[i][j] = sum_{a,b<f} x[i+a][j+b] h[a][b]`,
`(N-f+1)^2` outputs, row-major `x` (N x N) and `h` (f x f) from `std::minstd_rand` (seed 42 for `x`, `42+f` for `h`) with
`U[0,1]` real and imaginary parts. As in the earlier evaluation:

| Library | Input | Pipeline |
| --- | --- | --- |
| cuDNN | real FP32, 1 channel (real parts) | `CUDNN_CROSS_CORRELATION`, fastest of the top three heuristic algorithms (timed); `cudnnFind` grew host memory past 300 GB at f≥31 |
| cuFFT | real FP32 (real parts) | full-image R2C of image and N x N padded filter, product, C2R (as the earlier `cufft.cu`) |
| cuFFTDx | complex FP32 | tiled (64) fused kernel, `sample/conv2d/cufftdx.cu`; 64 x 64 filter FFT (cuFFT) |
| ThunderFFT | complex FP32 (TF32) | tiled (64), fused current public N=64 TF32 FFT kernels; cuFFT filter transform included |

cuDNN and cuFFT use real input while the tiled FFT paths are complex; this
follows the earlier evaluation code (its sheet labels cuFFT "c2c", but the code
is R2C, and only R2C matches its timing). Every timed FFT iteration includes the filter FFT; the
zero-padded, flipped spatial filter layout and cuFFT plans are prepared once.

`thff_conv N f[,f...] cufft|cufftdx|cudnn|thunderfft [warmup] [runs] [trials]` checks finite relL2 ≤ 5e-3 against a direct
double-precision reference before timing. For N≤256, the gate checks every output; larger images use samples. The sampled
outputs cover every
corner, the last row/column and both sides of every tile boundary (crossed with
a strided subset). Each trial is the mean of 3–20 launches (about 0.2 s of work) after up to 5
warmups; the median of
3 trials is plotted.

The ThunderFFT adapter is `sample/conv2d/thunder_tiled64.cu`: it uses FP32
complex storage, TF32 tensor-core arithmetic, a padded 64×65 shared tile and
four warps. Each iteration includes the 64×64 cuFFT filter transform. The driver
handles partial tiles and every odd paper filter. This is an artifact
implementation using the current public kernels; equivalence to the original
paper binary and paper-scale performance remain to be measured.

## Split collection and offline assembly

```bash
THUNDERFFT_FIGURE5_LIBRARIES=cuFFT,ThunderFFT ./run.sh figure5 --profile paper --collection conv2d --part fft
MATHDX_ROOT=/path/to/mathdx THUNDERFFT_FIGURE5_LIBRARIES=cuFFTDx ./run.sh figure5 --profile paper --collection conv2d --part cufftdx
THUNDERFFT_FIGURE5_LIBRARIES=cuDNN ./run.sh figure5 --profile paper --collection conv2d --part cudnn
./run.sh figure5 --profile paper --collection conv2d --assemble
```

Filter subsets may be collected separately with `_FILTERS`. GPU UUID, driver,
shape, operation and each backend's source/binary identity must agree across its
parts. Missing cells remain in `coverage.json` and gaps in `figure.pdf`.

## Optional backend CSVs

Columns `gpu,shape,filter,library,precision,repeat,time_ms,filter_fft_included,rel_l2,validated,passed,operation` with
`shape=N=16384` and `operation` equal to `FIGURE5["operation"]` in `measure.py`.
External adapters must use the same input generation/reference check and set
`filter_fft_included=true` for FFT paths (`n/a` for cuDNN).

## Notes

The paper figure is H100. A100 results show the trends only; compare absolute
values with the paper on H100.

## Run time and optional truncation

One native process per library sweeps every filter size: the input is generated
once (`std::minstd_rand`, seed 42; each filter from seed `42+f`), each size is
gated once, and the trials then run back to back. On the A100 the paper profile
takes about 4 minutes (cuFFT 54 s, cuFFTDx 54 s, cuDNN 143 s).
Each trial times enough launches for about 0.2 s of work (3 to 20 launches), so
slow cuDNN sizes do not dominate. The CPU reference check uses OpenMP when CMake
finds it and otherwise runs on one thread with the same result.

`THUNDERFFT_FIGURE5_SLOW_RATIO=r` (default `0`, off) stops cuDNN after the first
size at which its median exceeds `r` times the fastest other library; larger
sizes are then left empty in the plot and listed in `metadata.json` `truncated`.

## Reference results (A100 80GB PCIe, CUDA 12.9, cuDNN 9.12, 2026-11-07)

Paper profile, N=16384, 234 gated measurements (max relL2 8.2e-7, trial spread ≤3%).
Against the earlier H100 evaluation: cuFFT 14.75 ms at every f (H100 15.68 ms,
0.94x; R2C is bandwidth-bound and both GPUs have about 2 TB/s); cuDNN 3.19 →
554.2 ms (1.28–1.71x H100); cuFFTDx 6.71 → 94.9 ms (1.48–1.90x H100). The trends
match; the compute-bound paths need an H100 run for an absolute comparison.

Run-order sensitivity: for f ≤ 19 the tiled cuFFTDx time depends on what ran
before it on the GPU. Directly after the cuFFT sweep (the order `run.sh` uses)
it measured 5.8–6.4 ms; started on an idle GPU, 6.7–7.5 ms. Both are stable within
a run (spread < 1%) and a longer time-based warmup did not change either, so the
difference is not clock ramp-up; large filters (f ≥ 21) agree. Keep the default
order when comparing runs, and recheck on H100.

Native result assembly reparses every recorded stdout/log and checks the saved
CSV rows against the successful commands and correctness gates. Imported backend
CSVs are copied byte-for-byte into `raw/backends/`, hashed, and revalidated from
that copy; the result bundle remains usable after moving it or deleting the
original import file. Numeric and unset CUDA visibility masks use a context-free
CUDA device UUID query, so physical identity does not depend on `nvidia-smi`
ordinal ordering.
