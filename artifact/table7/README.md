# Table 7: optimization ablation and unit-kernel measurements

This task measures four historical N=64 TF32 kernel paths. Two are complete-FFT
candidates; two intentionally retain partial computations from earlier commits.
The results identify the measured function, included operations, source commit,
and accuracy status. A partial kernel can supply a unit-kernel timing without
claiming to implement a complete FFT.

| Variant | Source | Operations measured |
|---|---|---|
| `register_branch` | `51e346c` | LUT-based register FFT with lane-dependent permutation; complete-FFT candidate |
| `register_uniform` | `5e37821` | LUT-based remapped register FFT with complete output stores; complete-FFT candidate |
| `historical_e4aa1b0` | `e4aa1b0` | Historical shared-memory round trip inside each R iteration; MMA B-fragment initialization omitted |
| `historical_652808b` | `652808b` | Register-retained historical path; MMA B-fragment initialization and inter-stage permutation omitted |

The last two variants require timing-only mode. Their uninitialized MMA operand
values depend on the compiler; the reported work is the emitted instruction
path, with no claim of a mathematically valid FFT. They keep their own IDs and
operation-coverage fields in every output.

The original paper rows `radix16` and `radix4_compute` remain unrecovered and
stay blank in the paper-shaped table. Exact identity between any selected
kernel and the paper's original four-stage experiment is unconfirmed. The
selected kernels also differ in layout, permutation and loop unrolling, so the
comparison does not isolate branch divergence alone.

## Measure and assemble

Builds require CUDA Toolkit, a C++17 compiler and a GPU with compute capability
8.0 or newer. No cuFFTDx tuning parameters or external FFT checkout are needed.
The artifact uses `-O3` without `--use_fast_math`.

Run all four unit kernels at the paper-sized workload, then assemble their CSVs:

```sh
THUNDERFFT_TABLE7_VARIANTS=historical_e4aa1b0,historical_652808b,register_branch,register_uniform \
THUNDERFFT_TABLE7_TIMING_ONLY=1 THUNDERFFT_TABLE7_TRIALS=3 \
  ./run.sh table7 --profile paper --collection a100-unit-kernels --part four-kernels

# File processing only; no GPU work.
./run.sh table7 --profile paper --collection a100-unit-kernels --assemble
```

For a quick measurement, set `THUNDERFFT_TABLE7_TRIALS=1` and use a new
collection name. One trial is one sample, not an estimate of run-to-run spread.
Timing-only mode skips R1 accuracy and output-guard checks. Raw errors are
`nan`, check counts are `-1`, and `passed=0`; a successful exit means the timing
completed. These rows are collected and assembled as `collected_unvalidated`
with `accuracy_status=unchecked` and the measured operation coverage.

The main comparison output is
`result/collections/a100-unit-kernels/assembled/table7/kernel_comparison.csv`.
It contains only collected kernel rows: four when the command above completes.
`table.csv` additionally keeps the missing original paper rows and appends the
two partial kernels under their distinct IDs. Missing measurements remain blank;
`full_paper_comparison` stays false.

To collect independently, run the same build and settings in separate parts:

```sh
THUNDERFFT_TABLE7_VARIANTS=historical_e4aa1b0,historical_652808b \
THUNDERFFT_TABLE7_TIMING_ONLY=1 THUNDERFFT_TABLE7_TRIALS=3 \
  ./run.sh table7 --profile paper --collection a100-unit-split --part partial-kernels

THUNDERFFT_TABLE7_VARIANTS=register_branch,register_uniform \
THUNDERFFT_TABLE7_TIMING_ONLY=1 THUNDERFFT_TABLE7_TRIALS=3 \
  ./run.sh table7 --profile paper --collection a100-unit-split --part register-kernels

./run.sh table7 --profile paper --collection a100-unit-split --assemble
```

Without an explicit variant list, the normal runner selects the two available
complete-FFT candidates; timing-only mode selects all four available kernels.
`THUNDERFFT_TABLE7_VARIANTS=all` explicitly requests the missing paper rows too
and reports pending before building or launching GPU work.

## Measurement scope and protocol

Paper-sized mode uses N=64, batch65536 and R1000. Smoke uses batch1024 and R256.
The default is three trials for paper-sized mode and two for smoke;
`THUNDERFFT_TABLE7_TRIALS` can request one or more. Batch and R can be overridden
for smoke with `THUNDERFFT_TABLE7_BATCH` and `THUNDERFFT_TABLE7_REPEATS`.
Batch must be a positive multiple of 16. Each kernel launches one warp per block
and processes 16 FFT-sized input slots.

Timing-only mode initializes every input element to `(0.125, -0.25)` and uses
one warmup for each repeat count. The ordinary validation mode uses bounded
Gaussian timing input and three warmups. Both reset immutable input outside
the CUDA event interval and report `(time(2R)-time(R))/R`. Repeated outputs may
overflow; they are not used as numerical validation. The result is repeated
kernel compute time per batch, not end-to-end latency of a single FFT call.

To run the complete candidates with their independent accuracy gate:

```sh
THUNDERFFT_TABLE7_VARIANTS=register_branch,register_uniform \
  ./run.sh table7 --profile paper --collection a100-unit-validated --part register-kernels
```

This mode checks R=1 bounded Gaussian and tone inputs at batch64 against a CPU
FP64 DFT of the represented input. Each transform must have relative L2 at most
0.005, finite outputs, and unchanged 16-element guards on each side. Failure
stops timing. `register_branch` previously passed local A100 R1 checks;
`register_uniform` has timing evidence, with R1 validation pending for this
selected source. The two partial kernels cannot use this complete-FFT gate.

Each result bundle records source and binary hashes, commands, GPU UUID,
compiler/runtime, input pattern, trials, warmups, accuracy status and operation
coverage. Standalone bundles are under `result/table7/<run_id>/`; collection
parts are under `result/collections/<name>/parts/table7/`.

Assembly retains both validated and unchecked timings with their original
labels. Different trial counts, warmups or input patterns are recorded per part
and set `protocol_heterogeneous=true`; they are not presented as one matched
protocol. Source/binary identity, GPU, compiler, batch, R and precision must be
compatible. Conflicting measurements of the same variant require choosing a
part; missing variants can be collected later. Original missing-stage speedups
remain blank.

## Utilization

Each of the four kernels issues 48 `m16n8k8` MMA instructions per warp. With
16 FFT-sized slots, this is `48 * (2*16*8*8) / 16 = 6144` Tensor Core FLOPs per
slot. For the partial kernels this counts issued MMA work, including operations
whose operand setup was omitted; it is not a count of correct FFTs completed.

```text
achieved_tflops = 6144 * batch / (median_compute_us * 1e6)
utilization_pct = 100 * achieved_tflops / 156
```

The denominator is A100 hardware peak **156 TFLOPS**. R is already removed from
`compute_us`. Timing-only utilization retains the unchecked label. Other GPUs
retain achieved throughput with utilization pending until a hardware reference
is specified. No profiling CSV is required. `collect_profile.py` is an optional
standalone diagnostic; its counters do not feed this calculation.

## Historical source and measurements

The self-contained headers and their adjacent provenance JSONs are under
`ablation/kernels/`. Kernel statements are retained from the named commits;
namespace wrapping and same-valued constants isolate the historical symbols.
The selected complete candidates use the original host LUT formulas: float PI
for `51e346c`, double PI converted to float theta for `5e37821`. LUT construction
is outside timing; the uniform launch retains its original 8704-byte dynamic
shared-memory reservation.

The older `403c6ae` branch source uses a 256-byte local array in original SM80
builds with CUDA 11.7, 12.8 and 12.9, as well as in a saved historical executable.
It remains source evidence but is not the selected register-resident candidate.
The selected `5e37821` already contains the complete output stores omitted by
older uniform versions. The surviving shared-memory baseline is radix8, despite
its 16-by-16 real matrix tile; it does not fill the missing radix16 paper row.

`history/manifest.json` indexes 16 archived timing-only measurements, raw CSVs,
native logs, source/binary hashes and known operation omissions. The compact
coverage map links 50 legacy commits to independently measured source groups.
These are unit-kernel measurement records. Their archived raw format is not a
current collection bundle; use the runner commands above to collect the selected
kernels with current metadata and assembly support.

The archived register sweep used constant input, one warmup and a common
float-PI LUT. Its shared radix8 family used seeded bounded Gaussian input.
An intervening selected-source timing-only run also used Gaussian input;
the current timing-only runner uses constant input for all selected kernels.
The source hashes and per-part input/protocol metadata distinguish these runs.
Similar times do not establish identical input, included operations or paper
stage identity.

The earlier `radix16.cuh` and `radix4_shared.cuh` reconstructions remain explicit
native diagnostics (`reconstructed_radix16`, `reconstructed_radix4_compute`).
They are not recovered original sources and cannot fill the pending paper rows.

## Build without GPU execution

```sh
CUDACXX=/usr/local/cuda-12.9/bin/nvcc THUNDERFFT_ARCHS=80 \
  bash artifact/table7/build.sh
./build_artifact/table7/thff_ablation --help
python3 artifact/table7/test_contract.py
python3 artifact/table7/test_utilization.py
python3 artifact/table7/test_assemble.py
python3 -m unittest discover -s artifact/table7/history -p test_history.py
```

## Optional baseline projection from Table 6

`project_baseline.py` calculates a separate normalized comparison without GPU
work. It keeps A100 hardware-peak utilization constant when mapping the patched
N256 FP16 tcFFT/FlashFFTConv compute timings to an N64 TF32 work model:

```text
source utilization = (65536 * batch / source_time_us / 1e6) / 312
projected_time_us = source_time_us * (12288 / 65536) * (312 / 156)
```

The source count is eight 16-by-16-by-16 WMMA operations per FFT. The target
count, `32*N*log2(N) = 12288`, is a theoretical normalization, not an implemented
N64 radix16 kernel. The rounded Table 6 README times give tcFFT **63.15%** and
**8.175 us estimated**, and FlashFFTConv **76.05%** and **6.7875 us estimated**.
These projections do not establish a measured N64 baseline or recover the
original paper stages.

The included [A100 comparison](reference/a100-20261107/projection/README.md)
combines the archived four-kernel timing run with these two documented-input
projections; its CSVs remain separate from the canonical paper-shaped table.

```sh
python3 artifact/table7/project_baseline.py --documented-table6 \
  --table7-csv artifact/table7/reference/a100-20261107/kernel_comparison.csv \
  --output result/table7/normalized-a100-20261107
```

The documented-input mode reads the public Table 6 rounded timing record and
labels that source explicitly; it does not manufacture raw timing trials.
To use a collected Table 6 result, replace `--documented-table6` with
`--table6-csv result/table6/<run_id>/table.csv`; its adjacent `data.csv` must
confirm the N256 FP16 complex batch65536 workload and supported kernel.
For an explicit timing, use `--library FlashFFTConv --source-time-us 18.1
--source-note 'description of the timing source'`.

Outputs are `baseline_estimates.csv`, `projection_metadata.json`, and optional
`comparison.csv`. The comparison preserves measured source IDs, accuracy and
operation-coverage fields; estimated rows and speedups against projected
baselines have explicit labels. Input hashes and the scaling assumption are
recorded. Canonical `table.csv` is untouched, `full_paper_comparison` remains
false, and existing output files are never overwritten.
