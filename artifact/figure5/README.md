# Figure 5: 2D convolution

Status: ready subset. Measures cuDNN, full-image cuFFT and tiled cuFFTDx after a
correctness gate; the tiled ThunderFFT kernel is joined from a separately
produced CSV. Requires `MATHDX_ROOT`. Not yet supported: offline collection
assembly.

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
`THUNDERFFT_FIGURE5_FILTERS`, `_TRIALS`, `_N` (smoke only) and `_BACKENDS`.

## Operation

Valid 2D cross-correlation `y[i][j] = sum_{a,b<f} x[i+a][j+b] h[a][b]`,
`(N-f+1)^2` outputs, row-major `x` (N x N) and `h` (f x f) from `srand(42)` with
`U[0,1]` real and imaginary parts. As in the earlier evaluation:

| Library | Input | Pipeline |
| --- | --- | --- |
| cuDNN | real FP32, 1 channel (real parts) | `CUDNN_CROSS_CORRELATION`, fastest of the top three heuristic algorithms (timed); `cudnnFind` grew host memory past 300 GB at f≥31 |
| cuFFT | real FP32 (real parts) | full-image R2C of image and N x N padded filter, product, C2R (as the earlier `cufft.cu`) |
| cuFFTDx | complex FP32 | tiled (64) fused kernel, `sample/conv2d/cufftdx.cu`; 64 x 64 filter FFT (cuFFT) |
| ThunderFFT | complex FP32 (TF32) | tiled (64), same pipeline as cuFFTDx; backend CSV |

cuDNN and cuFFT use real input while the tiled FFT paths are complex; this
follows the earlier evaluation code (its sheet labels cuFFT "c2c", but the code
is R2C, and only R2C matches its timing). Every timed FFT iteration includes the filter FFT; the
zero-padded, flipped spatial filter layout and cuFFT plans are prepared once.

`thff_conv N f cufft|cufftdx|cudnn` checks finite relL2 ≤ 5e-3 against a direct
double-precision reference before timing. The checked outputs cover every
corner, the last row/column and both sides of every tile boundary (crossed with
a strided subset). Timing is the mean of 20 runs after 5 warmups; the median of
3 trials is plotted.

## Backend CSVs (tiled ThunderFFT)

Columns `gpu,shape,filter,library,repeat,time_ms,rel_l2,passed,operation` with
`shape=N=16384` and `operation` equal to `FIGURE5["operation"]` in `measure.py`.
The tiled ThunderFFT driver should use the same input generation and reference
check, and include the filter FFT in its timing.

## Notes

The paper figure is H100. A100 results show the trends only; compare absolute
values with the paper on H100.

## Reference results (A100 80GB PCIe, CUDA 12.9, cuDNN 9.12, 2026-11-05)

Paper profile, N=16384, 234 gated runs (max relL2 8.2e-7, trial spread ≤0.9%).
Against the earlier H100 evaluation: cuFFT 14.75 ms at every f (H100 15.68 ms,
0.94x; R2C is bandwidth-bound and both GPUs have about 2 TB/s); cuDNN 3.07 →
548.5 ms (1.28–1.71x H100); cuFFTDx 6.70 → 95.0 ms (1.48–1.90x H100). The trends
match; the compute-bound paths need an H100 run for an absolute comparison.
