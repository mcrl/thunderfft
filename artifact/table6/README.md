# Table 6: Prior Tensor Core FFT comparison

Paper p. 9, Section 6.3: 256-point FP16 forward FFT, batch 65,536, on A100.
Compares cuFFTDx, tcFFT, FlashFFTConv and ThunderFFT (xFFT) by time and
speedup over cuFFTDx.

## Run

```sh
MATHDX_ROOT=/path/to/nvidia/mathdx/25.06 THUNDERFFT_ARCHS=80 \
  ./run.sh table6 --backend local --profile paper
```

tcFFT and FlashFFTConv are not vendored. On every run the script clones these
commits from GitHub into `build_artifact/artifact_table6/`, replacing any
previous clone, applies the patches and builds the drivers:

| Library | Repository | Commit |
| --- | --- | --- |
| tcFFT | https://github.com/rox906/tcFFT | `3e87a7083577741815e21b2a3c2bed22520786ce` |
| FlashFFTConv | https://github.com/HazyResearch/flash-fft-conv | `b8771028717f46d5b22cbb8e12833f35033d621b` |

Requirements:

- MathDX for cuFFTDx. Without it the task reports `pending`.
- Network access to GitHub.
- PyTorch (headers only) for FlashFFTConv. Set `FLASHFFTCONV_PYTHON` if it is
  not in `python3`.

`paper` uses batch 65536, R=1000 and 3 trials. `smoke` uses batch 4096, R=64
and 2 trials.

## Method

Each kernel repeats the FFT R times on resident data in a `#pragma unroll 1`
loop. The FFT time is `(t_2R - t_R) / R`, the median over trials. Before
timing, every path is checked against an FP64 DFT (relL2 ≤ 0.005).

| Library | Measured kernel |
| --- | --- |
| cuFFTDx | Table 3 block FFT, `block_f16_e16` (fastest on A100) |
| tcFFT | `layer_256_0_A100` with a repeat loop ([patch](tcfft/layer_256_0_A100_repeat.patch)) |
| FlashFFTConv | its N=256 conv kernel with a repeat loop, complex input, FFT output only ([patch](flashfftconv/monarch_conv_fft_repeat.patch)) |
| ThunderFFT | Table 3 register kernel |

Notes:

- tcFFT's default `layer_256_0` is incorrect on A100, so the driver uses
  tcFFT's `layer_256_0_A100`.
- FlashFFTConv has no FFT-only API, so the FFT part of its convolution kernel
  is timed.

## A100 result (2026-11-06)

| Library | Time (ms) | Speedup | Paper (ms) | Paper speedup |
| --- | ---: | ---: | ---: | ---: |
| cuFFTDx | 0.0229 | 1.00 | 0.023 | 1.00 |
| tcFFT | 0.0218 | 1.05 | 0.021 | 1.11 |
| FlashFFTConv | 0.0181 | 1.27 | 0.018 | 1.28 |
| ThunderFFT | 0.0158 | 1.45 | 0.016 | 1.46 |

## Outputs

Written to the run's result directory:

- `table.csv`: the table above (time, speedup over cuFFTDx, paper values).
- `data.csv`: every trial.
- `metadata.json`: settings, pinned sources and build commands.
- `run.log`: console output of the build and every driver.
