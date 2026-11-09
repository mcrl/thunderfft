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
./run.sh figure5 --profile smoke   # 2048 x 2048; cuDNN f = 3; others f = 3, 33, 53
./run.sh figure5 --profile paper   # 16384 x 16384; cuDNN f = 3..11; others f = 3..53
./run.sh figure5 --profile paper --cudnn-mode full  # every library, all 26 odd filter sizes
```

The smoke profile uses one trial; paper uses three. Default `--cudnn-mode fast`
keeps cuDNN filters up to 11 and all requested filters for cuFFT, cuFFTDx and
ThunderFFT. The cutoff precedes the first observed 10x slowdown versus the
fastest other library in the retained 2026-11-09 A100 sweep. Each library's
selected filters run in one native process. The expected prepared-account
time is about 3–5 minutes in fast mode (provisional estimate), versus
9–12 minutes with every cuDNN filter.

`--cudnn-mode full` restores all requested cuDNN filters and disables
`THUNDERFFT_FIGURE5_SLOW_RATIO`. `--full` also selects Table 3's complete
cuFFTDx candidate search when Table 3 is included in the command. Figure 5's
tiled cuFFTDx implementation is unchanged by `--cufftdx-mode`; that option
controls Table 3 only.

To keep parts and assemble them later without a GPU:

```bash
./run.sh figure5 --profile paper --collection fig5 --part a100
./run.sh figure5 --assemble --collection fig5 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_FIGURE5_LIBRARIES` | `auto` or comma list of `cuFFT`, `cuFFTDx`, `ThunderFFT`, `cuDNN` | `auto` (all available) |
| `THUNDERFFT_CUDNN_MODE` | `fast` (cuDNN filters <=11) or `full`; explicit `--cudnn-mode` takes precedence | `fast` / `fast` |
| `THUNDERFFT_FIGURE5_FILTERS` | comma list of odd filter sizes from 3 to 59, before the cuDNN mode cutoff | `3,33,53` / `3,5,...,53` |
| `THUNDERFFT_FIGURE5_TRIALS` | positive integer (median over trials) | 1 / 3 |
| `THUNDERFFT_FIGURE5_N` | image size; smoke only (paper accepts only 16384) | 2048 / 16384 |
| `THUNDERFFT_FIGURE5_SLOW_RATIO` | optional additional fast-mode cutoff: `r` > 0 stops after the first cuDNN size slower than `r` times the fastest other library; `0` disables; ignored in full mode | 0 / 0 |
| `THUNDERFFT_FIGURE5_TIMEOUT` | seconds per native process | 1800 / 1800 |

## Output

A run writes to `result/figure5/<run_id>/`:

- `data.csv`: per-trial time and relative L2 error for each library and filter size
- `summary.csv`: median, minimum and maximum time per library/filter, plus arithmetic means over measured filter sizes
- `figure.pdf`: runtime versus filter size, with the skipped cuDNN tail left as a gap and annotated
- `metadata.json`: settings, GPU, source and binary hashes, `policy_skipped` and any additional truncated sizes
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `summary.csv`, `figure.pdf`, `metadata.json` and
`coverage.json` (unexpected missing cells and intentional `policy_skipped`
cells) to `result/collections/fig5/assembled/figure5/`. Fast paper mode omits
21 cuDNN filter sizes; `coverage_complete` is false even when all selected
measurements complete. Plots retain those positions as NaN gaps; raw CSVs
contain measured rows only. cuDNN's paper-grid arithmetic mean is `mean(5)`;
the three FFT libraries use `mean(26)`. These means cover different size sets.
Full mode gives all four libraries `mean(26)`.
