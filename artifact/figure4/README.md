# Figure 4: A100 1D convolution comparison

Measures the runtime of an FP16 depthwise circular 1D convolution (L = 4096,
batch 64, 768 channels; forward FFT, pointwise product, inverse FFT) as the
filter size grows, for ThunderFFT, cuDNN and FlashFFTConv (paper p. 10). Runs on
the local A100. Every library passes a correctness check against a direct
reference before it is timed.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh figure4 --profile smoke   # batch 4, 64 channels, K = 3, 9, 53, 1 trial
./run.sh figure4 --profile paper   # batch 64, 768 channels, K = 3, 5, ..., 53, 3 trials
```

FlashFFTConv is not vendored. The adapter fetches
`https://github.com/HazyResearch/flash-fft-conv.git` at commit
`b8771028717f46d5b22cbb8e12833f35033d621b` (network access to GitHub is needed)
and builds its unmodified L = 4096 kernel against the CPU PyTorch headers that
`artifact/setup.sh` installs. If PyTorch cannot be imported, FlashFFTConv is
skipped; selecting it explicitly then fails.

To collect libraries in separate runs, add `--collection NAME --part LABEL`
(a new label per run) and then assemble offline:

```bash
THUNDERFFT_FIGURE4_LIBRARIES=ThunderFFT ./run.sh figure4 --profile paper --collection fig4 --part thunderfft
THUNDERFFT_FIGURE4_LIBRARIES=cuDNN,FlashFFTConv ./run.sh figure4 --profile paper --collection fig4 --part cudnn-flash
./run.sh figure4 --assemble --collection fig4 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_FIGURE4_LIBRARIES` | `auto` or comma list of `ThunderFFT`, `cuDNN`, `FlashFFTConv` | `auto` (all available) |
| `THUNDERFFT_FIGURE4_FILTERS` | comma list of filter sizes (1 to 4096) | `3,9,53` / `3,5,...,53` |
| `THUNDERFFT_FIGURE4_TRIALS` | positive integer (median over trials) | 1 / 3 |
| `THUNDERFFT_FIGURE4_VALIDATE_CHANNELS` | number of (batch, channel) sequences checked against the direct reference | 1024 / 1024 |
| `THUNDERFFT_FIGURE4_SLOW_RATIO` | `r` > 0 stops cuDNN after the first filter size slower than `r` times the fastest other library; `0` disables | 0 / 0 |
| `THUNDERFFT_FIGURE4_TIMEOUT` | seconds per native process | 1800 / 1800 |
| `FLASHFFTCONV_PYTHON` | optional: another Python with PyTorch for the FlashFFTConv build | the runner's Python |

## Output

Each run writes to `result/figure4/<run_id>/`:

- `data.csv`: per-trial time and relative L2 error for each library and filter size
- `summary.csv`: median, minimum and maximum time per library and filter size
- `figure.pdf`: runtime versus filter size
- `metadata.json`: settings, GPU, source and binary hashes, FlashFFTConv build record, truncated sizes
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `flash-build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `summary.csv`, `figure.pdf`, `metadata.json` and
`coverage.json` (missing cells) to `result/collections/NAME/assembled/figure4/`.
