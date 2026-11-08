# Figure 5: A100 2D convolution comparison

Measures the runtime of a valid FP32 2D cross-correlation of a 16384 x 16384
image as the filter size grows, for cuDNN, full-image cuFFT, tiled (64) cuFFTDx
and tiled (64) ThunderFFT (TF32) (paper p. 10). Every library passes a
correctness check against a direct reference before it is timed, and every FFT
timing includes the filter FFT.

cuFFTDx needs `MATHDX_ROOT`; cuDNN is used when found (`CUDNN_HOME` or
`CUDNN_ROOT`).

## Run

Run locally on an A100 80GB PCIe GPU:

```bash
./run.sh figure5 --profile smoke   # 2048 x 2048 image, f = 3, 33, 53, 1 trial
./run.sh figure5 --profile paper   # 16384 x 16384 image, f = 3, 5, ..., 53, 3 trials
```

To keep parts and assemble them later without a GPU:

```bash
./run.sh figure5 --profile paper --collection fig5 --part a100
./run.sh figure5 --assemble --collection fig5 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_FIGURE5_LIBRARIES` | `auto` or comma list of `cuFFT`, `cuFFTDx`, `ThunderFFT`, `cuDNN` | `auto` (all available) |
| `THUNDERFFT_FIGURE5_FILTERS` | comma list of odd filter sizes from 3 to 59 | `3,33,53` / `3,5,...,53` |
| `THUNDERFFT_FIGURE5_TRIALS` | positive integer (median over trials) | 1 / 3 |
| `THUNDERFFT_FIGURE5_N` | image size; smoke only (paper accepts only 16384) | 2048 / 16384 |
| `THUNDERFFT_FIGURE5_SLOW_RATIO` | `r` > 0 stops cuDNN after the first filter size slower than `r` times the fastest other library; `0` disables | 0 / 0 |
| `THUNDERFFT_FIGURE5_TIMEOUT` | seconds per native process | 1800 / 1800 |

## Output

A run writes to `result/figure5/<run_id>/`:

- `data.csv`: per-trial time and relative L2 error for each library and filter size
- `summary.csv`: median, minimum and maximum time per library and filter size
- `figure.pdf`: runtime versus filter size
- `metadata.json`: settings, GPU, source and binary hashes, truncated sizes
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `summary.csv`, `figure.pdf`, `metadata.json` and
`coverage.json` (missing cells) to `result/collections/fig5/assembled/figure5/`.
