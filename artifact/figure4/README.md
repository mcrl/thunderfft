# Figure 4: A100 1D convolution comparison

Measures the runtime of an FP16 depthwise circular 1D convolution (L = 4096,
batch 64, 768 channels; forward FFT, pointwise product, inverse FFT) as the
filter size grows, for ThunderFFT, cuDNN and FlashFFTConv (paper p. 10). Runs on
the local A100. Every library passes a correctness check against a direct
reference before it is timed.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh figure4 --profile smoke   # batch 4, 64 channels; cuDNN K = 3, 9; others also K = 53
./run.sh figure4 --profile paper   # batch 64, 768 channels; cuDNN K = 3..25; others K = 3..53
./run.sh figure4 --profile paper --cudnn-mode full  # every library, all 26 odd filter sizes
```

The smoke profile uses one trial; paper uses three. Default `--cudnn-mode fast`
keeps cuDNN filters up to 25 and all requested filters for ThunderFFT and
FlashFFTConv. The cutoff precedes the first observed 3x slowdown versus the
fastest other library in the retained 2026-11-09 A100 sweep. The selected
filters are batched into one process per library, preserving input reuse.
The expected prepared-account time is about 1–2 minutes in fast mode
(provisional estimate), versus 2–3 minutes with every cuDNN filter.

`--cudnn-mode full` restores all requested cuDNN filters and disables
`THUNDERFFT_FIGURE4_SLOW_RATIO`. `--full` also selects Table 3's complete
cuFFTDx candidate search when Table 3 is included in the command; Figure 4
itself has no cuFFTDx candidate search. `--cufftdx-mode all` alone leaves
Figure 4 in its default fast mode.

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
| `THUNDERFFT_CUDNN_MODE` | `fast` (cuDNN filters <=25) or `full`; explicit `--cudnn-mode` takes precedence | `fast` / `fast` |
| `THUNDERFFT_FIGURE4_FILTERS` | comma list of filter sizes (1 to 4096), before the cuDNN mode cutoff | `3,9,53` / `3,5,...,53` |
| `THUNDERFFT_FIGURE4_TRIALS` | positive integer (median over trials) | 1 / 3 |
| `THUNDERFFT_FIGURE4_VALIDATE_CHANNELS` | number of (batch, channel) sequences checked against the direct reference | 1024 / 1024 |
| `THUNDERFFT_FIGURE4_SLOW_RATIO` | optional additional fast-mode cutoff: `r` > 0 stops after the first cuDNN size slower than `r` times the fastest other library; `0` disables; ignored in full mode | 0 / 0 |
| `THUNDERFFT_FIGURE4_TIMEOUT` | seconds per native process | 1800 / 1800 |
| `FLASHFFTCONV_PYTHON` | optional: another Python with PyTorch for the FlashFFTConv build | the runner's Python |

## Output

Each run writes to `result/figure4/<run_id>/`:

- `data.csv`: per-trial time and relative L2 error for each library and filter size
- `summary.csv`: median, minimum and maximum time per library/filter, plus arithmetic means over measured filter sizes
- `figure.pdf`: runtime versus filter size, with the skipped cuDNN tail left as a gap and annotated
- `metadata.json`: settings, GPU, source and binary hashes, FlashFFTConv build record, `policy_skipped` and any additional truncated sizes
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `flash-build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `summary.csv`, `figure.pdf`, `metadata.json` and
`coverage.json` (unexpected missing cells and intentional `policy_skipped`
cells) to `result/collections/NAME/assembled/figure4/`. Fast paper mode omits
14 cuDNN filter sizes; `coverage_complete` is false even when all selected
measurements complete. Plots retain those positions as NaN gaps; raw CSVs
contain measured rows only. cuDNN's paper-grid arithmetic mean is `mean(12)`;
ThunderFFT and FlashFFTConv use `mean(26)`. These means cover different size
sets. Full mode gives all three libraries `mean(26)`.
