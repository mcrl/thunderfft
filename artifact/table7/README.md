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

Validated mode checks each kernel at R=1 against an FP64 DFT before timing.
Timing-only mode (`THUNDERFFT_TABLE7_TIMING_ONLY=1`) skips that check and marks
rows `accuracy_status=unchecked`; the two partial kernels run only in this mode.

The paper stages `radix16` (radix-16 baseline) and `radix4_compute` (radix-4
with stage-wise shared-memory exchange) have no recovered source. They cannot
be collected and stay blank in the assembled `table.csv`.

## Run

```bash
./run.sh table7                    # smoke: register_branch, register_uniform (validated)
./run.sh table7 --profile paper    # paper: register_branch, register_uniform (validated)

# All four available kernels, timing-only
THUNDERFFT_TABLE7_TIMING_ONLY=1 ./run.sh table7 --profile paper

# Without a local A100
./run.sh table7 --backend vast --gpu A100
THUNDERFFT_TABLE7_TIMING_ONLY=1 ./run.sh table7 --profile paper --backend vast --gpu A100
```

To get the paper-shaped table with the missing stages as blank rows, collect
the variants into a collection and assemble without a GPU:

```bash
THUNDERFFT_TABLE7_TIMING_ONLY=1 \
  ./run.sh table7 --profile paper --collection t7 --part timing
./run.sh table7 --assemble --collection t7 --profile paper
```

Variants can also be split across parts with `THUNDERFFT_TABLE7_VARIANTS`
(same build and settings in every part).

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE7_VARIANTS` | comma-separated subset of `register_branch`, `register_uniform`, `historical_e4aa1b0`, `historical_652808b` | `register_branch,register_uniform`; all four when `TIMING_ONLY=1` |
| `THUNDERFFT_TABLE7_TIMING_ONLY` | `0` or `1` | `0` |
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

Assembly writes `table.csv` (paper rows, missing stages blank, plus any collected
partial kernels), `kernel_comparison.csv` (collected variants only), `data.csv` and
`metadata.json` to `result/collections/<name>/assembled/table7/`.
