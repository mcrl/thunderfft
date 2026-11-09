# Table 5: Numerical accuracy and finite dynamic range

Measures relative L2 error against a cuFFT FP64 reference at `alpha=1` and
finite outputs at input scale `alpha=2^k` (paper p. 9), on an A100. Each FFT
size has four rows, each with fresh accuracy and finite-range measurements:
xFFT/ThunderFFT FP16, cuFFTDx FP16, xFFT/ThunderFFT TF32 and cuFFT FP32.
cuFFTDx requires MathDX; its range scan is now included alongside its
accuracy measurement.

The default `--table5-range-mode boundary` checks four exponents `k-1..k+2`
around each saved last-finite exponent `k`. If the observed boundary shifts,
the first window is widened and checked again. If the window does not contain
adjacent finite/non-finite exponents, it expands until they are found or the
profile's entire interval has been checked. A missing bracket then fails the
boundary check; full mode can instead report a censored bound. Every reported
value comes from the new run; saved boundaries only choose starting windows.

`--table5-range-mode full` checks the complete profile interval. `smoke` uses
N=64, batch 64 and intervals `8..13` for both FP16 implementations and
`120..127` for TF32/FP32. `paper` uses N=64/256/1024/4096, batch 65536 and
`0..127` for all four implementations. Boundary windows stay within these
profile intervals. Accuracy always uses the original `alpha=1` computation
and FP64 reference, regardless of range mode.

With unchanged saved boundaries, default paper mode produces 16 accuracy
rows plus 64 finite checks (80 raw rows). Full paper mode produces 16 accuracy
rows plus 2,048 finite checks (2,064 raw rows). The initial prepared-account
time estimates are 2–4 minutes for boundary mode and 22–28 minutes for full
mode, pending measurement of this revision.

The starting values and source hashes are in
[`boundary_presets.json`](boundary_presets.json). ThunderFFT FP16/TF32 and
cuFFT FP32 values come from the retained `6864641` A100 full scan on
2026-11-09. cuFFTDx FP16 values come from a separate `730e768` A100 probe
at exponents `6..13`, batch 65536 and seed 1234: the last-finite exponents
were 11/9/8/7 for N=64/256/1024/4096. That cuFFTDx probe established local
brackets, not a complete `0..127` scan.

Finite-output sweeps generate the Gaussian samples and double-precision
normalization once per size, then reuse them for every exponent. Scaling still
rounds to FP32 only after applying alpha, so the inputs and every measured
exponent match the original generator exactly. Accuracy/reference runs retain
their original input path; the sweep cache uses an additional 2 GiB of host
memory at the largest paper size.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh table5                    # smoke, fresh boundary checks
./run.sh table5 --profile paper    # paper accuracy + fresh boundary checks
./run.sh table5 --profile paper --table5-range-mode full  # every exponent 0..127
./run.sh --full                    # whole workflow with all three exhaustive modes
```

`--full` selects `--cufftdx-mode all --cudnn-mode full --table5-range-mode full`.
The individual Table 5 flag changes only its range checks.

The paper profile can be collected in parts (by phase or size) and assembled
without a GPU:

```bash
THUNDERFFT_TABLE5_PHASES=accuracy_reduced,accuracy_fp32_cufft,accuracy_fp16_cufftdx \
  ./run.sh table5 --profile paper --collection t5 --part accuracy
THUNDERFFT_TABLE5_PHASES=range_fp16,range_tf32,range_fp32_cufft,range_fp16_cufftdx \
  ./run.sh table5 --profile paper --collection t5 --part range
./run.sh table5 --assemble --collection t5 --profile paper
```

Use the same range mode, batch, seed and source in every part. Assembly rejects
mixed boundary/full modes. Cells that were not collected stay blank.

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE5_RANGE_MODE` | `boundary` or `full`; explicit `--table5-range-mode` takes precedence | `boundary` |
| `THUNDERFFT_TABLE5_PHASES` | comma-separated subset of `accuracy_reduced`, `accuracy_fp32_cufft`, `accuracy_fp16_cufftdx`, `range_fp16`, `range_tf32`, `range_fp32_cufft`, `range_fp16_cufftdx` | all seven (both cuFFTDx phases require MathDX) |
| `THUNDERFFT_TABLE5_N_LIST` | comma-separated subset of `64,128,256,512,1024,2048,4096` | `64` / `64,256,1024,4096` |
| `THUNDERFFT_TABLE5_BATCH` | positive multiple of 16 | `64` / `65536` |
| `THUNDERFFT_TABLE5_SEED` | nonnegative integer | `1234` |
| `THUNDERFFT_TABLE5_TIMEOUT` | seconds per native invocation, including each boundary-window attempt | `600` |

## Output

Written to `result/table5/<run_id>/`:

- `table.csv` (same content as `table5_summary.csv`) and `table.md`: four rows per size, with error and largest finite / first non-finite exponent observed in the measured window
- `data.csv`: all raw rows combined
- `accuracy_reduced.csv`, `accuracy_fp32_cufft.csv`, `accuracy_fp16_cufftdx.csv`: error at `alpha=1` (FP16+TF32, cuFFT FP32, cuFFTDx FP16)
- `range_fp16.csv`, `range_tf32.csv`, `range_fp32_cufft.csv`, `range_fp16_cufftdx.csv`: per-exponent finite-output scans; boundary mode retains each size's final measured window
- `raw/`: CSVs, native logs and environment records for every invocation, including earlier boundary-window attempts
- `<phase>.log`, `<phase>.environment.json`: native output and GPU/CUDA/cuFFT identity per phase
- `comparison.csv`: differences from the reference values in `artifact/table5/table5_accuracy_range_summary_with_fp32.csv`
- `validation.json`: completeness and numerical sanity checks
- `metadata.json`, `commands.txt`: settings, build, native commands, preset provenance and boundary-window attempts
- `status.json`, `run.log`: runner status and console log

Assembly writes `table.csv`, `table5_summary.csv`, `table.md`, `data.csv`, `comparison.csv`,
`validation.json` and `metadata.json` to `result/collections/<name>/assembled/table5/`;
missing cells are listed in `result/collections/<name>/assembled/missing.csv`.

`range_coverage=boundary_recheck` means the displayed window was measured and
contains an adjacent finite/non-finite pair. It does not establish global
monotonicity or the largest finite exponent across unmeasured values.
`range_coverage=full_profile_scan` identifies a full profile scan; for the
paper profile that is `0..127`. `full_range_scan` and the measured scan limits
remain explicit in tables and metadata. Accuracy remains a fresh measurement
in both modes, and no skipped exponent is filled with an inferred result.
