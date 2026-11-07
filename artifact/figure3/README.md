# Figure 3: End-to-end FFT performance

Status: ready subset. Measures communication-only, ThunderFFT, cuFFT and
(with `MATHDX_ROOT`) cuFFTDx on one GPU and plots performance normalized by the
communication-only time. Libraries and sizes can be collected separately and
assembled offline.

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

The paper profile takes about 15 seconds on the A100: only the single-pass e2e
timing is used, so the in-kernel R/2R loop runs with R=16
(`THUNDERFFT_FIGURE3_REPEATS`).

`THUNDERFFT_FIGURE3_PRECISIONS=tf32,fp16` adds FP16 storage (ThunderFFT FP16,
cuFFTDx FP16; cuFFT is FP32 only). `THUNDERFFT_FIGURE3_N_LIST`,
`_BATCH`, `_REPEATS` and `_TRIALS` override the smoke settings; the paper
profile requires batch 65,536. Without MathDX the cuFFTDx bars are absent and
listed in `metadata.json` `missing`.

## Collect separately and assemble

`THUNDERFFT_FIGURE3_LIBRARIES` selects comma-separated native names:
`copy`, `copy-scalar`, `thunderfft`, `cufft`, `cufftdx`. By default all installed
paths are selected; an explicitly selected cuFFTDx path requires MathDX.
`copy-scalar` is a diagnostic series and is not required to complete the figure.

```bash
THUNDERFFT_FIGURE3_LIBRARIES=copy,thunderfft \
  ./run.sh figure3 --profile paper --collection paper-a100 --part communication-fft
THUNDERFFT_FIGURE3_LIBRARIES=cufft \
  ./run.sh figure3 --profile paper --collection paper-a100 --part cufft
MATHDX_ROOT=/path/to/mathdx THUNDERFFT_FIGURE3_LIBRARIES=cufftdx \
  ./run.sh figure3 --profile paper --collection paper-a100 --part cufftdx
./run.sh figure3 --profile paper --collection paper-a100 --assemble
```

`THUNDERFFT_FIGURE3_N_LIST=64,256` also splits collection by FFT size. Store each
non-overlapping subset under its own `--part`. You can collect FFT rows before
the communication baseline: times are retained, normalization stays blank,
and the run produces a pending plot until compatible copy rows are available.
Offline assembly creates `summary.csv`, `baselines.csv` and `figure.pdf` under
`result/collections/paper-a100/assembled/figure3/`; missing cells stay pending
and absent bars remain gaps. Multi-GPU results use separate plot panels.

Assembly reparses the original native CSVs (including all cuFFTDx candidates),
verifies saved rows and source/device/runtime provenance, and rejects overlapping
configurations. For each GPU family, use the same physical GPU, source hashes,
driver/CUDA versions, and batch/repeat/trial settings across parts. The required
paper coverage is all four sizes with TF32 storage; explicitly collected FP16
storage adds its corresponding required series. FP32 cuFFT pairs only with the
TF32-storage copy, never with the FP16 copy.

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
