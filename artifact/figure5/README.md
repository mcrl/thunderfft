# Figure 5: H100 2D convolution comparison

Measures the runtime of a valid FP32 2D cross-correlation of a 16384 x 16384
image as the filter size grows, for cuDNN, full-image cuFFT, tiled (64) cuFFTDx
and tiled (64) ThunderFFT (TF32) (paper p. 10). The paper figure is an H100
measurement. Every library passes a correctness check against a direct reference
before it is timed, and every FFT timing includes the filter FFT.

## Run

On a machine with an H100:

```bash
./run.sh figure5 --profile smoke   # 2048 x 2048 image, f = 3, 33, 53, 1 trial
./run.sh figure5 --profile paper   # 16384 x 16384 image, f = 3, 5, ..., 53, 3 trials
```

Without a local H100, rent one on Vast.ai (credentials: [cloud README](../cloud/README.md#credentials)),
then assemble locally without a GPU:

```bash
./run.sh figure5 --backend vast --gpu H100 --profile paper --collection fig5 --part h100
./run.sh figure5 --assemble --collection fig5 --profile paper
```

Appending `--cloud-plan` to the first command checks the offer and cost without
renting. The Vast.ai worker installs MathDX and cuDNN and sets the build
variables itself. Collect all four libraries in one command so that they are
measured on the same physical GPU.

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_FIGURE5_LIBRARIES` | `auto` or comma list of `cuFFT`, `cuFFTDx`, `ThunderFFT`, `cuDNN` | `auto` (all available) |
| `THUNDERFFT_FIGURE5_FILTERS` | comma list of odd filter sizes from 3 to 59 | `3,33,53` / `3,5,...,53` |
| `THUNDERFFT_FIGURE5_TRIALS` | positive integer (median over trials) | 1 / 3 |
| `THUNDERFFT_FIGURE5_N` | image size; smoke only (paper accepts only 16384) | 2048 / 16384 |
| `THUNDERFFT_FIGURE5_SLOW_RATIO` | `r` > 0 stops cuDNN after the first filter size slower than `r` times the fastest other library; `0` disables | 0 / 0 |
| `THUNDERFFT_FIGURE5_TIMEOUT` | seconds per native process | 1800 / 1800 |

These variables, when exported, are also forwarded to the Vast.ai worker.

## Output

A local run writes to `result/figure5/<run_id>/`; a Vast.ai run downloads the
same directory under `result/cloud/thunderfft-ae-<session>/download/result/` and
adds it to the collection.

- `data.csv`: per-trial time and relative L2 error for each library and filter size
- `summary.csv`: median, minimum and maximum time per library and filter size
- `figure.pdf`: runtime versus filter size
- `metadata.json`: settings, GPU, source and binary hashes, truncated sizes
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `summary.csv`, `figure.pdf`, `metadata.json` and
`coverage.json` (missing cells) to `result/collections/fig5/assembled/figure5/`.
