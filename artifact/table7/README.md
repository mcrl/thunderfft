# Table 7: 64-point optimization ablation

Measures N=64 TF32 unit-kernel compute time per batch (`(t_2R - t_R) / R`,
median over trials) and Tensor Core utilization against the A100 peak of
156 TFLOPS for historical kernels and a new stage-wise shared-memory reconstruction (paper p. 10),
on an A100.

| Variant | Source commit | Kernel | Mode |
|---|---|---|---|
| `shared_stage_radix4` | new reconstruction; source SHA-256 | stage-wise shared-memory radix-4 FFT; no register-fragment permutation | validated or timing-only |
| `register_branch_cached` | `51e346c` mapping + `652808b` permutation; source SHA-256 | complete register FFT; cached initialized coefficients; lane-dependent permutations | validated or timing-only |
| `register_branch` | `51e346c` | register FFT with lane-dependent branches; complete FFT | validated or timing-only |
| `register_uniform` | `5e37821` | register FFT with remapped fragments; complete FFT | validated or timing-only |
| `historical_e4aa1b0` | `e4aa1b0` | shared-memory round trip; MMA operand setup omitted (partial) | timing-only |
| `historical_652808b` | `652808b` | register path; MMA operand setup and stage permutation omitted (partial) | timing-only |

The `+ reducing communication` row selects **`register_branch_cached`**, a
complete-FFT reconstruction using the valid `51e346c` data mapping and the
three conditional-swap permutation from checkpoint `652808b`, preceding
`403c6ae` (`reduce divergence`). Each condition now covers all eight disjoint
quartets in a stage, reducing repeated branch handling while preserving the
lane-dependent branches and exact permutation. It restores every stage
permutation and MMA operand; the original checkpoint comments out both. This is a derived
implementation, not an unmodified historical kernel. Its 28 distinct coefficient
pairs are prepared once before repetition, matching `shared_stage_radix4`; the
difference-based timing excludes that preparation for both rows. The `§` marker
and source/binary hashes identify this new implementation; its original paper-stage identity remains unconfirmed.

The [native matched A100 comparison](reference/a100-register-grouped-20261110/README.md)
measures **6.5403 µs** for the grouped-branch candidate versus **9.3051 µs**
for the previous per-quartet implementation, approximately **29.7% less time**
(N=64, batch=65536, R=1000). Coefficient preparation and all FFT operations
have the same timing boundary in this comparison. These measurements describe
repeated computation, not full FFT calls.
Separate R=1 Gaussian and tone checks pass with maximum per-transform relative
L2 error **0.00151072429021**. Register use drops from **194 to 160**, with no
local-memory spills. The repeat body retains all **48 HMMA** instructions and
no shared/global loads/stores; its static branch-control counts drop from
16 to 6 BSSY instructions and from 33 to 9 BRA instructions. Static totals
include divergent alternatives and are not general dynamic instruction counts.
The host W64 LUT follows the same construction as the shared and uniform
candidates; the original branch's float-PI construction remains unchanged.

The [previous 9.30 µs measurement archive](reference/a100-register-cached-20261110/README.md)
is preserved unchanged, including historical restoration comparisons. Its
source hashes describe the earlier per-quartet implementation and cannot fill
the current selected row. The original checkpoint's approximately **3.48 µs**
partial-kernel timing cannot be used as an FFT result.

Normal collection includes the four complete candidates. Timing-only remains
the default and marks each run `accuracy_status=unchecked`.
`THUNDERFFT_TABLE7_TIMING_ONLY=0` validates all four at R=1 against an FP64 DFT.
The native `register_branch` ID retains the unmodified `51e346c` implementation
as a separate diagnostic, with coefficient preparation in the repeated path.
It never substitutes for missing `register_branch_cached` evidence.

The old `e4aa1b0` and `652808b` partial kernels remain available only through
explicit `THUNDERFFT_TABLE7_VARIANTS` selection with timing-only mode. Their
operand initialization is incomplete; their timings are not complete FFT
measurements and cannot populate the current reduced-communication row.
[Archived e4aa1b0 evidence](reference/a100-e4aa1b0-20261109/README.md) is preserved
unchanged with its original selection history and approximately 7.9 µs timing.

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
blank.

The second paper role, `radix4_compute`, explicitly selects the new
`shared_stage_radix4` reconstruction. It retains the three radix-4 stages and
Tensor Core work of the register implementation, exchanging values through
shared-memory loads/stores at every stage instead of register-fragment
permutation. The 48 MMA instructions per warp (16 FFTs) account for 6,144
Tensor Core FLOPs per FFT. Its measured time is not forced to the paper's
0.0120 ms. The optimized implementation uses a conflict-free 8 KiB shared
layout, vector loads/stores, and groups of four independent butterflies.
It prepares the 21 invariant MMA coefficient pairs once per kernel launch,
before the repetition loop. Consequently, `(t_2R - t_R) / R` excludes this
preparation and measures steady-state computation rather than full FFT launch
latency. The recorded scope is
`repeated_kernel_compute_coefficients_prepared_once`; the table's `‡` note
also states this timing boundary.

The original source is still unrecovered: metadata keeps
`implementation_kind=reconstructed`, `paper_identity=unconfirmed`, and the
source/binary SHA-256 values, with no invented historical commit. A `‡` note
identifies it in the comparison table. Existing bundles without this exact
source remain readable but leave its selected row blank; they never substitute
another native variant. Numerical validation is recorded per run; timing-only
runs retain their unchecked status even after a separate validation succeeds.

The [optimized A100 measurement bundle](reference/a100-shared-optimized-20261110/README.md)
records approximately **10.98 µs** in five validated paper-protocol trials,
below the paper's 12.0 µs reference. R=1 Gaussian/tone checks retain a maximum
per-transform relative L2 error of 0.001514. This improvement includes moving
invariant coefficient preparation outside the repeated timing interval, along
with shared-memory layout and scheduling changes. All 48 MMAs and every
stage's shared-memory loads/stores remain in the repeated interval.

The [pre-optimization measurement bundle](reference/a100-shared-stage-20261110/README.md)
preserves the older source/binary hashes and its **22.8055 µs** validated timing
(**22.7942 µs** in default timing-only mode). That implementation used a 9,344 B
shared buffer and prepared coefficients inside the loop. Those retained results
describe the older implementation, not the current default.

## Run

Run locally on an A100 80GB PCIe GPU.

```bash
./run.sh table7                    # smoke: four complete candidates, timing-only
./run.sh table7 --profile paper    # paper: four complete candidates, timing-only

# Optional numerical checks for the four complete-FFT candidates
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
To collect only the selected communication-reduction candidate:

```bash
THUNDERFFT_TABLE7_VARIANTS=register_branch_cached THUNDERFFT_TABLE7_TIMING_ONLY=0 \
  ./run.sh table7 --profile paper --collection t7 --part register-cached
./run.sh table7 --assemble --collection t7 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE7_VARIANTS` | comma-separated subset of `register_branch`, `register_uniform`, `shared_stage_radix4`, `register_branch_cached`, `historical_e4aa1b0`, `historical_652808b` | `register_branch,register_uniform,shared_stage_radix4,register_branch_cached` |
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
`register_branch_cached` for the `register_branch` paper role changes presentation
only; it does not rename a kernel or rewrite any archived measurement.

The same explicit selection applies to `shared_stage_radix4` for the
`radix4_compute` role. Only newly collected, matching source evidence fills that
row; original raw CSV identities, timing protocol, and accuracy status remain
intact. `full_paper_comparison` remains false because reconstruction does not
establish the original paper experiment's identity.
