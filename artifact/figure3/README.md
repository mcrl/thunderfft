# Figure 3: End-to-end FFT performance

Status: ready subset. Measures communication-only, ThunderFFT, cuFFT and
(with `MATHDX_ROOT`) cuFFTDx on one GPU and plots performance normalized by the
communication-only time. Not yet supported: offline collection assembly.

Paper p. 10, Section 6.5: N=64/256/1024/4096, batch 65,536. The
communication-only kernel performs vectorized global loads and stores without
FFT arithmetic; it is the ideal baseline (normalized performance 1). Section 6.5
does not name the GPU or precision; the earlier A100 evaluation used FP32
storage, which is the default here (ThunderFFT TF32, cuFFT/cuFFTDx FP32).

## Run

```bash
./run.sh figure3 --profile smoke
MATHDX_ROOT=/path/to/nvidia/mathdx/25.06 ./run.sh figure3 --profile paper
```

`THUNDERFFT_FIGURE3_PRECISIONS=tf32,fp16` adds FP16 storage (ThunderFFT FP16,
cuFFTDx FP16; cuFFT is FP32 only). `THUNDERFFT_FIGURE3_N_LIST`,
`_BATCH`, `_REPEATS` and `_TRIALS` override the smoke settings; the paper
profile requires batch 65,536. Without MathDX the cuFFTDx bars are absent and
listed in `metadata.json` `missing`.

## What is measured

All paths run through `sample/bench/paper_fft.cu` (`thff_paper_fft`, or
`thff_paper_fft_cufftdx` for cuFFTDx) with the same input and timing:

| Library | Native call | Kernel |
| --- | --- | --- |
| Comm. only | `--library copy N tf32 global` | `int4` load/store, one word per thread, full grid |
| Comm. only (scalar) | `--library copy-scalar ...` | one complex element per thread (reported, not the denominator) |
| ThunderFFT | `N tf32 register` | `ThunderFFT_benchmark_reg_e2e` (global → FFT → global) |
| cuFFT | `--library cufft N fp32 global` | out-of-place `cufftExecC2C`; plan creation excluded |
| cuFFTDx | `--library cufftdx N fp32 register` | Table 3 candidates; the fastest median e2e candidate is kept |

- Correctness before timing: copies must be byte-exact; FFTs must pass relL2
  ≤ 5e-3 against cuFFT FP64 for both the single-pass and repeated kernels.
- Timing: `e2e_out_of_place_us`, the mean of 10 back-to-back single-pass
  launches per trial; median over trials (3 in the paper profile). Twiddle
  initialization, allocation and input generation are excluded.
- Bytes: `2 * N * batch * element bytes` (read + write) for every path.
- `normalized = Comm. only median / implementation median`, paired by N and bytes.

## Outputs

`data.csv` (raw trials), `baselines.csv` (cuFFT/cuFFTDx rows only),
`summary.csv` (medians and normalized values), `figure.pdf`, `metadata.json`
(settings, commands, hashes, cuFFTDx candidate selection) and `raw/`.

## Reference results (A100 80GB PCIe, CUDA 12.9, 2026-11-05)

Paper profile, normalized to the vector copy:

| N | ThunderFFT | cuFFT | cuFFTDx |
| ---: | ---: | ---: | ---: |
| 64 | 0.975 | 0.931 | 0.923 |
| 256 | 0.953 | 0.966 | 0.954 |
| 1024 | 0.995 | 0.977 | 0.968 |
| 4096 | 1.007 | 0.987 | 0.982 |

Communication-only times are within 2.5% of the earlier evaluation
(0.0439/0.1639/0.6428/2.5537 ms). ThunderFFT reaches at least 95% of the
communication-only baseline; cuFFT and cuFFTDx fall below 95% at N=64, as in
the earlier evaluation. Values within about 5% are treated as run-to-run noise.
