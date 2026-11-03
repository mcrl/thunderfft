# ThunderFFT artifact tasks

This branch organizes reproduction work for the supplied PPoPP'27 draft,
*xFFT: Accelerating FFT with Tensor Cores*. The paper's **xFFT** denotes the
implementation named **ThunderFFT** in this repository. Paper numbers below
refer to that draft; confirm them if the paper changes.

The folders are independent evaluations; some adapters are still incomplete.
Each task documents its implemented scope and missing measurements.
Figure 1 is a design diagram
and Table 1 is a literature survey, so neither has a GPU evaluation task.

| Task | Paper page | Deliverable | Starting point / dependency |
| --- | ---: | --- | --- |
| [validation](validation/README.md) | — | FFT correctness and memory guards | Ready: 88 smoke / 500 full cases plus 18 host API checks |
| [figure2](figure2/README.md) | 9 | A100 FFT kernel time plot | Ready: ThunderFFT-only subset derived from this run's Table 3 |
| [figure3](figure3/README.md) | 10 | End-to-end FFT normalized performance | `sample/bench/`; communication-only and baseline work pending |
| [figure4](figure4/README.md) | 10 | A100 1D convolution plot | Native correctness gate available; paper workload and comparable semantics pending |
| [figure5](figure5/README.md) | 10 | H100 2D convolution plot | `sample/conv2d/`; full benchmark pending |
| [table2](table2/README.md) | 3 | A100 performance model and measured reference | Ready: calculated rows; measured cuFFTDx row excluded |
| [table3](table3/README.md) | 8 | FFT kernel latency matrix | Ready: ThunderFFT 28 configurations/GPU; cuFFTDx comparison excluded |
| [table4](table4/README.md) | 8 | MMA and unit-kernel throughput | MMA microbenchmark pending |
| [table5](table5/README.md) | 9 | Numerical accuracy and dynamic range | Ready: measurements and tables; FP32 is cuFFT; published provenance unconfirmed |
| [table6](table6/README.md) | 9 | Prior Tensor Core FFT comparison | External baseline integration pending |
| [table7](table7/README.md) | 9 | Optimization ablation | Two recovered candidates can be collected separately; first two variants remain unavailable |

The general paper environment (p. 7) is one GPU with an AMD EPYC-class host:
A100 PCIe 80GB with Ubuntu 20.04/CUDA 12.9, and H100 PCIe or B200 with Ubuntu
24.04/CUDA 13.0. This does not define missing per-experiment settings; resolve
those explicitly in the task README.

## Run from the repository root

Use Python 3.9 or newer, CMake 3.18 or newer, a C++17 compiler and the CUDA
toolkit for the GPU being evaluated. The tested A100 setup uses CUDA 12.9 and
`THUNDERFFT_ARCHS=80`. Only the selected CMake targets are built, so the
optional convolution targets' MathDX/cuDNN dependencies are not required.
Install the plotting dependency once:

```bash
python3 -m pip install -r artifact/requirements.txt
bash artifact/check.sh          # CPU contracts and failure-path checks
```

```bash
./run.sh --list
./run.sh --dry-run
./run.sh                         # available tasks, smoke profile
./run.sh all                     # also report unfinished tasks as pending
./run.sh validation --profile paper
./run.sh figure4 table5          # selected tasks
./run.sh table3 --profile paper --device 0 --jobs 8
./run.sh table5 --result-dir /path/to/results --build-dir /path/to/build
```

By default, `run.sh` selects **validation, Table 2, Table 3, Table 5 and Figure 2**,
in dependency order, with no downloaded FFT library or cuFFTDx tuning data.
CUDA Toolkit's cuFFT is still used as an FP64 numerical reference and Table 5's
explicitly labeled FP32 path; no cuFFT performance tuning is required.
`./run.sh all` includes unfinished assignments; `./run.sh --list` lists every
assignment, while `./run.sh --dry-run` previews the available default batch.

The runnable Table 2/3/Figure 2 adapters cover the subsets in the table above.
Successful execution does not imply that the excluded cuFFTDx comparison or
the complete multi-GPU paper experiment has been reproduced. Each task's
`coverage` and `full_paper_comparison=false` are retained in runner status and
the batch manifest. B200/H100 measurements require their own compatible GPU
and native architecture build; Figure 2 requires A100 source measurements.

`smoke` checks the implemented path with small workloads. `paper` requests the
documented paper settings. Adapters must record effective arguments, including
overrides, and must not label a small smoke run as a paper reproduction.
CUDA/CMake and their development libraries must be available for implemented GPU
tasks. Set `CUDACXX=/path/to/cuda/bin/nvcc` and
`CUDAToolkit_ROOT=/path/to/cuda` when the toolkit is not discoverable. Table 5's
paper profile uses the retained evaluator's operational batch 65536 and seed
1234, whose exact published provenance is unconfirmed. For a bounded full-size
and exponent scan use
`THUNDERFFT_TABLE5_BATCH=64 ./run.sh table5 --profile paper`; the override is
recorded and does not establish the original population.

The 2026-11-03 local A100 check completed all five default smoke tasks. An
additional run passed the 500-case correctness matrix, measured all 28
ThunderFFT configurations at batch 65536/R=1000/three trials, generated the
corresponding Figure 2 PDF, and checked all four Table 5 sizes over exponents
0 through 127 at batch 64 (1548 raw rows including accuracy checks). This
establishes execution of the declared subsets on that A100; it does not
establish the excluded comparisons, other GPUs or Table 5's original population.

The runner creates a unique run directory for each evaluation:

```text
result/
  <evaluation>/<run_id>/          # task output, runner logs, status.json
  runs/<run_id>/
    manifest.json
    summary.csv
```

Completed smoke tasks produce `data.csv`. Completed paper tasks additionally
produce `figure.pdf` for figures or `table.csv` for tables. The runner checks that
these files exist and are nonempty. Extra raw files, validation reports, and tuning results belong in the
same task result directory. Existing source-adjacent CSVs are historical inputs,
not results of the new runner; their original measurements have not been
revalidated against this branch. Keep them separate from fresh measurements.

## Collect CSVs separately and assemble later

A named collection stores independent result bundles. Each bundle contains its
CSVs, raw logs, correctness evidence and measurement settings. Missing inputs
remain pending; no placeholder timing or utilization values are inserted.
Collection and assembly are separate operations:

```bash
# Collect two parts on the same A100 with the same build and settings.
THUNDERFFT_TABLE3_PRECISIONS=fp16 \
  ./run.sh table3 --profile paper --collection ae --part a100-fp16
THUNDERFFT_TABLE3_PRECISIONS=tf32 \
  ./run.sh table3 --collection ae --part a100-tf32

# Collect each available ablation candidate independently.
THUNDERFFT_TABLE7_VARIANTS=register_branch \
  ./run.sh table7 --collection ae --part ablation-branch-time
THUNDERFFT_TABLE7_VARIANTS=register_uniform \
  ./run.sh table7 --collection ae --part ablation-uniform-time

# Generate the tables/figures from whatever has been collected. No GPU work.
./run.sh --collection ae --assemble
```

`--profile` defaults to the collection's existing profile, or `smoke` for a new
collection. Use a different collection for smoke and paper data or incompatible
measurement settings. Table 3 supports `THUNDERFFT_TABLE3_N_LIST`,
`THUNDERFFT_TABLE3_PRECISIONS`, and `THUNDERFFT_TABLE3_PLACEMENTS`. Table 5 supports
`THUNDERFFT_TABLE5_N_LIST` and `THUNDERFFT_TABLE5_PHASES`; Table 7 supports
`THUNDERFFT_TABLE7_VARIANTS` for separate timing collections.
Their READMEs list the exact values and measurement commands. Collecting Table 7
without an explicit variant list in collection mode selects its two available
candidates and leaves the missing original stages pending.

Table 7 utilization is derived from the operation count and collected timing,
with A100 hardware peak **156 TFLOPS** as the denominator. Each recovered TF32
candidate issues 6144 MMA FLOPs per FFT. No separate utilization collection or
profiler CSV is needed; assembly calculates the values from valid timing rows.
Other GPUs retain achieved throughput with the utilization reference pending.

Existing successful runs can be imported without measuring again. Transfer the
whole evaluation directory, including its metadata and raw evidence, when
collecting on another machine:

```bash
./run.sh --collection ae --part a100-existing \
  --import-result table3=/path/to/completed/table3/run-directory
./run.sh --collection ae --assemble figure2
```

The second command assembles Table 3's collected parts and generates Figure 2
from its A100 register data. It does not run Table 3 again. During collection,
selecting Figure 2 selects its Table 3 source measurements; figures are generated
by the separate assembly command.

Table 3 also retains each fully validated configuration as a checkpoint. If a
later configuration fails or the runner is interrupted, completed configurations
can still be imported; the unsuccessful run keeps its original failure status.
Running bundles and incomplete trials are never accepted. Collect the remaining
configurations into a new part with the selection variables above.

```text
result/collections/ae/
  manifest.json                     # explicit part list and file hashes
  parts/<evaluation>/<part>/         # immutable CSV/log/metadata snapshots
  assembled/
    status.csv                      # complete, partial or pending per evaluation
    missing.csv                     # all still-uncollected inputs
    <evaluation>/table.csv          # collected cells plus blank pending cells
    figure2/figure.pdf              # plotted when at least one valid point exists
```

Assembly does not automatically pick the latest run or merge unrelated CSVs.
Adapters validate input correctness and reject incompatible settings, source or
binary hashes, GPU identity and software versions. Table 3 keeps different GPU
families in separate rows. Duplicate/conflicting parts are rejected unless an
adapter explicitly verifies they are identical. Choose nonoverlapping selections
when collecting a matrix. Existing parts cannot be overwritten; start a new
collection to compare a revised implementation.

`--assemble` can also initialize an empty collection and produce its missing-input
list. It returns **2** while requested measurements remain pending, **1** for
invalid or conflicting evidence, and **0** when its requested scope is complete.
An invalid assembly preserves the previous assembled output and records
`assembly-error.json`; that previous output does not incorporate the rejected
parts. When updating selected evaluations, previously assembled evaluations are
also refreshed, so existing tables and derived figures are preserved.
Missing Figure 2 points are gaps, not zero values or interpolated lines.
Unsupported evaluations and external comparisons remain listed as pending.
The implemented subset and full-paper completeness are recorded separately.

## Adapter contract

Implement `artifact/<evaluation>/run.sh`. The root runner supplies:

| Variable | Meaning |
| --- | --- |
| `THUNDERFFT_ROOT` | Absolute repository root |
| `THUNDERFFT_RESULT_DIR` | Unique output directory for this task and run |
| `THUNDERFFT_BUILD_DIR` | Build directory selected by the runner |
| `THUNDERFFT_PROFILE` | `smoke` or `paper` |
| `THUNDERFFT_DEVICE` | CUDA ordinal within the inherited visible GPU set; preserve scheduler visibility |
| `THUNDERFFT_JOBS` | Build parallelism |
| `THUNDERFFT_PYTHON` | Python executable |
| `THUNDERFFT_RUN_ID` | Unique batch run ID |
| `THUNDERFFT_RESULTS_ROOT` | Root result directory |
| `THUNDERFFT_DEPENDENCY_RESULTS` | JSON mapping of dependency IDs to their result directories |
| `THUNDERFFT_COLLECTION` | Named collection when collecting independent parts; unset for ordinary runs |

Write all new results under `THUNDERFFT_RESULT_DIR`; do not overwrite checked-in
CSVs. Return `0` only after the task's checks and promised outputs succeed.
Return **78** while the task is unfinished or a required dependency/configuration
is unavailable; other nonzero values denote failures. The root runner records
these states and returns **0** when every selected task completes, **1** for an
actual failure, **2** for unfinished/blocked tasks without an actual failure,
and **130** for interruption. Actual failures take precedence over pending work.
Each task's original exit code is retained in `status.json`.
A pending
adapter must explain the missing work rather than fabricate timings or copy
historical CSVs into a successful run.

Selecting Figure 2 automatically selects Table 3 first. If Table 3 does not
complete, Figure 2 is recorded as blocked. Its implementation can find the same
batch's Table 3 directory in `THUNDERFFT_DEPENDENCY_RESULTS["table3"]` after
decoding the environment variable as JSON; it should verify the data and record
the source file and hash. Selecting Figure 2 alone does not make old CSVs new data.

`completed` describes successful execution and the output contract. The runner
records `scientific_validation: not_assessed`; the adapter's own validation
report and the task owner's review establish numerical correctness and faithful
paper reproduction. It does not infer those claims from exit code zero.

For incremental assembly, provide `artifact/<evaluation>/assemble.py` with
`assemble(parts, output, profile)`. `parts` contains immutable result-directory
paths; `output` is an empty staging directory. This function must only read and
validate collected evidence and write derived files, without GPU queries,
compilation or measurement. Return `status` (`complete`, `partial`, or `pending`),
`missing` (objects with `item` and `reason`), and `full_paper_comparison`.
Validate numerical gates, configuration completeness and compatible provenance
before combining values. Keep output references relative when they point to
other assembled files because the staging directory is renamed on publication.
Adapters without this implementation remain pending in the collection.

## Completion and provenance

Before timing an implementation, compare it with the agreed reference using the
same inputs, output definition, normalization, and precision policy. Record
non-finite outputs and enforce documented pass/fail tolerances. Convolution
baselines must agree on padding, filter orientation, output region, and channel
semantics. Accuracy failures block performance claims.

For every result, retain the Git revision and dirty state, GPU model, software
and baseline versions, exact commands/settings, seeds, correctness results, units,
timing scope, warmup/repetition counts, and aggregation rule. Keep full tuning
data when reporting a best configuration. Use raw measurements to generate the
plot/table and distinguish paper targets, historical data, and new measurements.

Unknown settings in the paper must be confirmed by the owner and documented
before calling the result a faithful reproduction. Each folder's proposed CSV
schema is an implementation guide; finalize and document it with the adapter.
