# ThunderFFT artifact tasks

This branch organizes reproduction work for the supplied PPoPP'27 draft,
*xFFT: Accelerating FFT with Tensor Cores*. The paper's **xFFT** denotes the
implementation named **ThunderFFT** in this repository. Paper numbers below
refer to that draft; confirm them if the paper changes.

The folders are assignment units, not a claim that every experiment is ready.
Assign an owner in each task's README and `registry.json`, resolve its unknown settings, and finish
its adapter before reporting reproduction results. Figure 1 is a design diagram
and Table 1 is a literature survey, so neither has a GPU evaluation task.

| Task | Paper page | Deliverable | Starting point / dependency |
| --- | ---: | --- | --- |
| [figure2](figure2/README.md) | 9 | A100 FFT kernel time plot | Derive from Table 3; no duplicate measurement |
| [figure3](figure3/README.md) | 10 | End-to-end FFT normalized performance | `sample/bench/`; communication-only and baseline work pending |
| [figure4](figure4/README.md) | 10 | A100 1D convolution plot | Existing 1D driver; correctness and comparable semantics pending |
| [figure5](figure5/README.md) | 10 | H100 2D convolution plot | `sample/conv2d/`; full benchmark pending |
| [table2](table2/README.md) | 3 | A100 performance model and measured reference | Model plus measured cuFFTDx rows |
| [table3](table3/README.md) | 8 | FFT kernel latency matrix | ThunderFFT helpers exist; cuFFTDx sweep pending |
| [table4](table4/README.md) | 8 | MMA and unit-kernel throughput | MMA microbenchmark pending |
| [table5](table5/README.md) | 9 | Numerical accuracy and dynamic range | Dedicated source exists; FP32 path is cuFFT |
| [table6](table6/README.md) | 9 | Prior Tensor Core FFT comparison | External baseline integration pending |
| [table7](table7/README.md) | 9 | Optimization ablation | Baseline variants and profiling pending |

The general paper environment (p. 7) is one GPU with an AMD EPYC-class host:
A100 PCIe 80GB with Ubuntu 20.04/CUDA 12.9, and H100 PCIe or B200 with Ubuntu
24.04/CUDA 13.0. This does not define missing per-experiment settings; resolve
those explicitly in the task README.

## Run from the repository root

```bash
./run.sh --list
./run.sh --dry-run
./run.sh                         # all tasks, smoke profile
./run.sh figure4 table5          # selected tasks
./run.sh table3 --profile paper --device 0 --jobs 8
./run.sh table5 --result-dir /path/to/results --build-dir /path/to/build
```

`smoke` checks the implemented path with small workloads. `paper` requests the
documented paper settings. Adapters must record effective arguments, including
overrides, and must not label a small smoke run as a paper reproduction.
CUDA/CMake and their development libraries must be available for implemented GPU
tasks. Set `CUDACXX=/path/to/cuda/bin/nvcc` and
`CUDAToolkit_ROOT=/path/to/cuda` when the toolkit is not discoverable. Table 5's
paper profile additionally requires an explicitly confirmed
`THUNDERFFT_TABLE5_BATCH`; the default smoke profile uses only a small population.

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

## Adapter contract for task owners

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
