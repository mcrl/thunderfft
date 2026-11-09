# Table 7: 64-point optimization ablation

Measures N=64 TF32 unit-kernel compute time per batch (`(t_2R - t_R) / R`,
median over trials) and Tensor Core utilization against the A100 peak of
156 TFLOPS for kernels taken from historical ThunderFFT commits (paper p. 9),
on an A100.

| Variant | Source commit | Kernel | Mode |
|---|---|---|---|
| `register_branch` | `51e346c` | register FFT with lane-dependent branches; complete FFT | validated or timing-only |
| `register_uniform` | `5e37821` | register FFT with remapped fragments; complete FFT | validated or timing-only |
| `historical_e4aa1b0` | `e4aa1b0` | shared-memory round trip; MMA operand setup omitted (partial) | timing-only |
| `historical_652808b` | `652808b` | register path; MMA operand setup and stage permutation omitted (partial) | timing-only |

The `+ reducing communication` row uses **`e4aa1b0`**
(`historical_e4aa1b0`). Its A100 timing rounds to **0.0079 ms**, with about
**33%** Tensor Core utilization. `variants.json` records this artifact selection
in `paper_row_sources`; it does not establish the original paper experiment's
exact identity. The native `register_branch` ID still identifies `51e346c` and
remains available as a separate diagnostic. It never substitutes for a missing
`e4aa1b0` measurement in the selected row.

Timing-only mode is the default, so normal Table 7 collection includes the
selected partial kernel. It skips numerical checks and marks rows
`accuracy_status=unchecked`. `e4aa1b0` omits MMA B-fragment initialization and
includes a shared-memory round trip; its reported time measures that instruction
path rather than a complete FFT. The table retains this qualification.
With the estimated tcFFT baseline available, the identified `e4aa1b0` and
`5e37821` rows also show their timing ratios. These retain the `†` unchecked
marker and are not correctness-validated FFT speedups.
Explicit validated mode (`THUNDERFFT_TABLE7_TIMING_ONLY=0`) checks the two
complete candidates at R=1 against an FP64 DFT; it cannot supply the selected
partial-kernel row.

The [three-trial A100 confirmation](reference/a100-e4aa1b0-20261109/README.md)
records a median of 0.0078878722 ms and 32.7225% utilization, with the original
CSV, native log and source/binary hashes.

The first paper row, `radix16`, is an **estimate**, calculated during paper
assembly from validated Table 6 tcFFT measurements on A100 at batch 65536.
Assuming the same Tensor Core utilization is retained when moving from N=256
FP16 to N=64 TF32:

```
t_N64_TF32 = t_tcFFT_N256_FP16 * (12288 / 65536) * (312 / 156)
           = 0.375 * t_tcFFT_N256_FP16
```

The N64 target uses the theoretical radix-16 model `32*N*log2(N) = 12,288`
FLOPs per FFT; the measured N256 tcFFT kernel uses 65,536. A100 dense
Tensor Core peaks are 156 TFLOPS for TF32 and 312 TFLOPS for FP16. Utilization
comes from the measured Table 6 time, not a fixed 70% assumption. This reuses
collected evidence without GPU work and labels the row as estimated rather
than measured. Without compatible Table 6 tcFFT evidence, the first row stays
blank. The second row, `radix4_compute` (stage-wise shared-memory exchange),
still has no recovered source and remains blank.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh table7                    # smoke: all four available kernels, timing-only
./run.sh table7 --profile paper    # paper: all four available kernels, timing-only

# Optional numerical checks for the two complete-FFT candidates
THUNDERFFT_TABLE7_TIMING_ONLY=0 ./run.sh table7 --profile paper
```

To get the paper-shaped table, collect the variants into a collection and
assemble without a GPU. If that collection also contains Table 6 tcFFT evidence,
assembly revalidates and reuses it for the first row; selecting Table 7 alone
does not launch a Table 6 measurement:

```bash
./run.sh table7 --profile paper --collection t7 --part timing
./run.sh table7 --assemble --collection t7 --profile paper
```

Variants can also be split across parts with `THUNDERFFT_TABLE7_VARIANTS`
(same build and settings in every part).
To add only the selected 0.0079 ms kernel to an existing collection:

```bash
THUNDERFFT_TABLE7_VARIANTS=historical_e4aa1b0 THUNDERFFT_TABLE7_TIMING_ONLY=1 \
  ./run.sh table7 --profile paper --collection t7 --part e4aa1b0
./run.sh table7 --assemble --collection t7 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE7_VARIANTS` | comma-separated subset of `register_branch`, `register_uniform`, `historical_e4aa1b0`, `historical_652808b` | all four; `register_branch,register_uniform` when `TIMING_ONLY=0` |
| `THUNDERFFT_TABLE7_TIMING_ONLY` | `0` or `1` | `1` |
| `THUNDERFFT_TABLE7_TRIALS` | 1..100 | `2` / `3` |
| `THUNDERFFT_TABLE7_BATCH` | positive multiple of 16 (paper requires `65536`) | `1024` / `65536` |
| `THUNDERFFT_TABLE7_REPEATS` | R, positive integer (paper requires `1000`) | `256` / `1000` |

## Output

Written to `result/table7/<run_id>/`:

- `kernel_comparison.csv`: one row per measured variant (median/min/max time, achieved TFLOPS, utilization, accuracy status, source commit)
- `table.csv`: same rows as `kernel_comparison.csv`
- `data.csv`: every trial
- `<variant>.csv`, `<variant>.log`: native output and GPU identity per variant
- `coverage.json`, `metadata.json`, `commands.txt`: selection, settings, hashes and commands
- `configure.log`, `build.log`: CMake output
- `status.json`, `run.log`: runner status and console log

Assembly writes `table.csv` (paper roles with explicit `source_variant`, the
estimated baseline when available, missing stages blank, and supplemental rows), `kernel_comparison.csv`
(collected variants only), `data.csv` and `metadata.json` to
`result/collections/<name>/assembled/table7/`.
`baseline_projection` in the metadata records the assumption, formula and input
provenance. The measured timing/utilization fields remain empty for the
estimated row; its projected values have separate columns. The pipeline records
this estimate as a coverage limitation in both fast and full modes.

`kernel_comparison.csv` and `data.csv` retain native measurement IDs. Selecting
`historical_e4aa1b0` for the `register_branch` paper role changes presentation
only; it does not rename a kernel or rewrite any archived measurement.
