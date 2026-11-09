# ThunderFFT artifact

Evaluations for *xFFT: Accelerating FFT with Tensor Cores* (PPoPP'27). xFFT in
the paper is ThunderFFT in this repository.

Start with [AE_README.txt](../AE_README.txt) for the one-command workflow,
default/full modes and estimated times; this page provides the detailed runner guide.

| Evaluation | Paper | GPU | Complete workflow |
| --- | --- | --- | --- |
| [validation](validation/README.md) | correctness | any | A100 stage |
| [table2](table2/README.md) | Table 2, p. 3 | CPU, A100 for the measured row | A100 stage |
| [table3](table3/README.md) | Table 3, p. 8 | A100, H100, B200 | all three GPU stages |
| [table4](table4/README.md) | Table 4, p. 8 | A100, H100, B200 | all three GPU stages |
| [table5](table5/README.md) | Table 5, p. 9 | A100 | A100 stage |
| [table6](table6/README.md) | Table 6, p. 9 | A100 | A100 stage |
| [table7](table7/README.md) | Table 7, p. 9 | A100 | A100 stage |
| [figure2](figure2/README.md) | Figure 2, p. 9 | A100 | A100 stage |
| [figure3](figure3/README.md) | Figure 3, p. 10 | A100 | A100 stage |
| [figure4](figure4/README.md) | Figure 4, p. 10 | A100 | A100 stage |
| [figure5](figure5/README.md) | Figure 5, p. 10 | A100 | A100 stage |

Figure 1 and Table 1 have no evaluation.

## Requirements

- Linux x86_64 with an NVIDIA GPU and driver
- CUDA Toolkit 12.9 for A100 (`/usr/local/cuda-12.9`, `/usr/local/cuda` or `CUDA_HOME`)
- C++17 host compiler, Python 3.9 or newer with `venv`
- Internet access to NVIDIA, PyPI and GitHub
- For `--backend vast`: credentials in `~/cloud_key/thunderfft/` ([Vast.ai backend](cloud/README.md))

## Setup

Once per account, no root access needed:

```bash
unzip thunderfft-<revision>.zip
cd thunderfft
bash artifact/setup.sh
source ~/.thunderfft_env
bash artifact/check.sh
```

`setup.sh` installs under `$HOME`: a venv at `~/venv` (matplotlib, CMake, CPU
PyTorch headers), MathDX 25.06.1 and cuDNN 9.12. It writes `~/.thunderfft_env`,
which activates the venv and sets `CUDA_HOME`, `PATH`, `MATHDX_ROOT`,
`CUDNN_HOME`, `LD_LIBRARY_PATH` and `THUNDERFFT_ARCHS` (GPU 0's architecture),
and sources it from `~/.bashrc`. `CUDA_HOME`, `THUNDERFFT_ARCHS`,
`THUNDERFFT_VENV`, `THUNDERFFT_MATHDX_ARCHIVE` and `THUNDERFFT_CUDNN_ARCHIVE`
(local archive or directory) override the defaults. `check.sh` runs CPU-only
tests.

## Complete reproduction with one command

```bash
./run.sh                               # paper workflow: preset cuFFTDx, fast cuDNN, Table 5 boundary checks
# Equivalent: ./run.sh all
./run.sh --full                         # all candidates, all cuDNN filters, full Table 5 exponent scans
./run.sh --cufftdx-mode all             # Table 3 candidate search; retain the short cuDNN sweep
./run.sh --cudnn-mode full              # all cuDNN filters; retain Table 3 saved parameters
./run.sh --table5-range-mode full       # full Table 5 scan; retain the other default modes
```

The launcher loads `~/.thunderfft_env` when present. With no evaluation names,
or with `all`, it uses the paper profile and runs these stages in order:

1. Run validation, Tables 2--7 and Figures 2--5 on an A100, using the local GPU
   when available and a rented A100 80GB PCIe otherwise.
2. Rent an H100 80GB PCIe for Tables 3 and 4, download its results and destroy it.
3. Rent a B200 for Tables 3 and 4, download its results and destroy it.
4. Assemble the collected measurements into the final tables and figures.

Table 3 defaults to `--cufftdx-mode preset`: remeasure the saved best cuFFTDx
parameters for each A100/H100/B200 configuration. `--cufftdx-mode all` repeats
the complete candidate search. Both modes check correctness and collect fresh
timings with the same paper settings; Table 2 and Figure 2 reuse those results.
See [Table 3](table3/README.md) for preset provenance and compatibility checks.

Figures 4 and 5 default to `--cudnn-mode fast`: cuDNN measures filters up to
25 in Figure 4 and up to 11 in Figure 5. Every other library measures the
whole requested filter grid. Each library's selected filters run in one
native process. The fixed cutoffs precede the first observed 3x/10x cuDNN
slowdown relative to the fastest other library in the retained A100 sweep.
The plot annotates the omitted cuDNN tail and leaves it blank; no timings
are filled in for skipped filters.

Table 5 defaults to `--table5-range-mode boundary`. For every FFT size, it
measures fresh `alpha=1` accuracy and finite-range results for four rows:
xFFT/ThunderFFT FP16, cuFFTDx FP16, xFFT/ThunderFFT TF32 and cuFFT FP32.
The range check measures `k-1..k+2` around each retained last-finite exponent
`k`, expanding the window if the boundary shifts or is not bracketed. The
saved values choose the starting window; they do not supply the reported
accuracy or finite flags. On the default paper grid, unchanged boundaries
give 16 accuracy rows and 64 finite checks. This is a local boundary recheck;
`--table5-range-mode full` scans every exponent `0..127` for all four rows
(2,064 raw rows including accuracy). See [Table 5](table5/README.md).

`--full` combines `--cufftdx-mode all --cudnn-mode full
--table5-range-mode full`. The individual options affect Table 3, Figures 4/5
and Table 5 respectively. `THUNDERFFT_CUDNN_MODE`
also accepts `fast` or `full`; an explicit CLI option takes precedence.
Full cuDNN mode measures every requested filter and disables the optional
`THUNDERFFT_FIGURE4_SLOW_RATIO` / `THUNDERFFT_FIGURE5_SLOW_RATIO` cutoff.

Approximate paper-workflow times on the prepared evaluation account, with
dependencies and local builds already available:

| Step, in execution order | GPU | Default (`preset` + `fast` + `boundary`) | `--full` |
| --- | --- | --- | --- |
| Validation | A100 | Under 1 minute | Under 1 minute |
| Table 3 | A100 | 4–6 minutes | 8–12 minutes |
| Table 2 | A100 | Under 1 minute | Under 1 minute |
| Table 4 | A100 | Under 1 minute | Under 1 minute |
| Table 5 | A100 | 3–5 minutes (measured: 4m03s) | 22–28 minutes (estimate) |
| Table 6 | A100 | Under 1 minute | Under 1 minute |
| Table 7 | A100 | Under 1 minute | Under 1 minute |
| Figure 3 | A100 | Under 1 minute | Under 1 minute |
| Figure 4 | A100 | 1–2 minutes | 2–3 minutes |
| Figure 5 | A100 | 2–4 minutes | 9–12 minutes |
| Tables 3 and 4, complete cloud stage | H100 PCIe | 8–12 minutes | 10–15 minutes |
| Tables 3 and 4, complete cloud stage | B200 | 8–12 minutes | 10–15 minutes |
| Final table assembly and Figure 2–5 PDFs | CPU | Under 1 minute | Under 1 minute |
| **Total** |  | **About 35–55 minutes (estimate)** | **About 65–85 minutes (estimate)** |

Cloud-stage estimates include startup, builds, measurements, result download
and cleanup. Initial setup/local builds, GPU waiting and cloud retries can add
time. Figure 2 reuses Table 3 measurements during final assembly.
The expanded Table 5 full-mode time, H100/B200 preset-stage times and both
workflow totals remain estimates. Table 5 full mode adds the cuFFTDx finite-range scan,
which the older full run did not include. The complete new
`preset` + `fast` + `boundary` workflow has not been rerun end to end on all
three GPUs; neither has the expanded `--full` workflow.
A100 Table 3 preset measurement took 4 minutes 56 seconds, plus 2 minutes
9 seconds for build/setup, compared with the previous full search's
8 minutes 41 seconds of measurement. See the Table 3 README for details.

The retained 2026-11-09 run at commit `6864641` completed the former default
(`all` cuFFTDx candidates and every cuDNN filter) in 61 minutes 52 seconds
on A100/H100/B200, including automatic assembly. At commit `317593c`, a
separate A100 preset check passed all 56 Table 3 configurations and Table 2 /
Figure 2 assembly, plus 494 CPU tests. These checks describe those revisions
and modes; they do not establish an end-to-end measurement of the new default.
That historical Table 5 result covered all four accuracy rows per size and
three finite-range implementations; cuFFTDx FP16 range was then uncollected.

On 2026-11-09, commit `a0f5755` passed 525 CPU tests and actual A100 fast
paper runs of Figures 4/5. Figure 4 took 1 minute 28 seconds and Figure 5
2 minutes 42 seconds, including incremental build/setup and plotting;
their measurement phases took 74.2 and 146.6 seconds. All 192 / 249 raw
trial rows passed the native correctness gates. Full-mode smoke runs
produced 9 / 12 rows, including cuDNN filter 53 in both figures, and offline
collection assembly succeeded for both modes.

The combined fast figure time was 4 minutes 10 seconds, compared with
11 minutes 18 seconds in the retained full-filter run (2 minutes 6 seconds /
9 minutes 12 seconds), about 63% shorter. Both used the same A100 GPU UUID.
The full-filter numbers come from the earlier run; full paper mode was not
remeasured alongside the new fast runs.

On 2026-11-09, commit `4344f24` passed 556 CPU tests and the new Table 5
paper boundary run on A100. It took 4 minutes 3 seconds: 3 minutes 30 seconds
in native measurements and 33.5 seconds in build/setup and summarization.
All 16 fresh accuracy values exactly matched the retained run, and all
12 previous range boundaries matched. The four new cuFFTDx FP16 boundaries
were 11/9/8/7 for N=64/256/1024/4096. The result contains 80 raw rows and
16 complete summary rows, with no window expansion. Full smoke produced
32 raw rows and four summary rows in 2.6 seconds; both offline assemblies
passed. Independent checks validated raw evidence, source/binary hashes,
GPU identity, preset provenance and the boundary/full-mode contract.

Table 5's 4m03s boundary run was about 78% shorter than the retained 18m26s
full scan of three implementations, while adding cuFFTDx range checks.
The new four-implementation full paper scan has not been timed; these modes
measure different exponent coverage.

Table 3 runs before the evaluations that reuse it. The workflow creates a
collection automatically; `--collection NAME` chooses its name. Final output
is under `result/collections/<name>/assembled/`. No manual import, status-file
editing or separate assembly command is needed.

Cloud stages require the evaluator account's Vast.ai credentials and incur
charges. The combined planned cost of all rentals, including an A100 fallback
and any retries, must fit the cloud configuration's budget. Preview the
workflow or current offers with:

```bash
./run.sh --dry-run                      # offline; no API calls or measurements
./run.sh --cloud-plan                   # required offers and combined cost; no rental
./run.sh --profile smoke --jobs 8        # all stages with smaller diagnostic workloads
```

The local GPU defaults to device 0. Use `--device N` or `CUDA_VISIBLE_DEVICES`
to select the A100 if needed. By default, `--a100-backend auto` waits up to
`--local-wait-seconds 120` for the local GPU, then rents an A100 if it is still
unavailable or busy. All A100 evaluations use that one selected GPU. Choose
`--a100-backend local` to require the local GPU or `--a100-backend vast` to
rent one directly. The workflow does not terminate other users' jobs. Paper
timings require exclusive use throughout the run.

The full workflow allows up to 1,200 seconds of build/run time for each H100
and B200 stage and 3,600 seconds for a rented A100 stage. Use
`--cloud-stage-seconds` and `--a100-stage-seconds` to change these limits;
the cloud configuration's `execution_seconds` remains an upper bound.
Startup, download, cleanup and delayed-billing reserves are accounted for
separately. If an affordable compatible offer is unavailable, the workflow
reports the stage and does not rent an incompatible GPU.

Every runnable evaluation produces its output. Table 5 labels its default
range coverage `boundary_recheck`; full mode uses `full_profile_scan`.
A boundary recheck does not establish global monotonicity or the maximum
finite exponent across the entire `0..127` grid. Default Figures 4/5 record
intentional cuDNN omissions in `policy_skipped` with `coverage_complete=false`;
their assembly can complete with no unexpected missing cells. Their arithmetic
means cover the measured sizes only: cuDNN uses `mean(12)` / `mean(5)`, while
the other libraries use `mean(26)` on the paper grid. Use `--full` for all
26 cuDNN sizes. Table 7 still has two unavailable historical paper stages;
its assembled table retains blank cells and coverage metadata for them.

## Run selected evaluations locally

```bash
./run.sh --list
./run.sh all --backend local --profile paper --jobs 8  # all A100 evaluations
./run.sh validation table2 table3 --jobs 8   # named evaluations default to local smoke
./run.sh table6 figure3 figure4 --profile paper --jobs 8
```

`smoke` checks paths with small workloads; `paper` uses the paper's settings.
Explicit `--backend local` runs default to `smoke`; add `--profile paper`
for paper measurements.
Other options include `--timeout SECONDS`, `--result-dir DIR` and
`--build-dir DIR`.

## Run on Vast.ai

GPUs the machine does not have are rented per command:

```bash
./run.sh table3 --backend vast --gpu H100 --profile paper --cloud-plan        # offer and cost only
./run.sh table3 --backend vast --gpu H100 --profile paper --collection ae --part h100
./run.sh table3 --assemble --collection ae --profile paper
```

The worker installs the dependencies and sets `MATHDX_ROOT`, `CUDNN_HOME` and
`THUNDERFFT_ARCHS` itself. `--gpu` selects the GPU for a single `--backend vast`
invocation; the complete workflow selects H100 and B200 automatically. See
[Vast.ai backend](cloud/README.md) for credentials and configuration.

## Collect per GPU and assemble

The complete workflow already collects and assembles all GPU stages. The
following commands are available when collecting selected parts independently.
Runs with `--collection NAME --part PART` are stored as separate parts;
`--assemble` combines them into tables and figures without GPU work. Table 3
and Table 4 put A100, H100 and B200 results into one table:

```bash
./run.sh table3 table4 --profile paper --collection ae --part a100
./run.sh table3 table4 --backend vast --gpu H100 --profile paper --collection ae --part h100
./run.sh table3 table4 --backend vast --gpu B200 --profile paper --collection ae --part b200
./run.sh table2 table3 table4 figure2 --assemble --collection ae --profile paper
```

One GPU's part must be collected in a single command, and all parts must come
from the same source. Use separate collections for smoke and paper runs.
Results from another account or machine are added with
`--import-result EVALUATION=DIR` (the complete result directory) and `--part`.

## Results

```text
result/<evaluation>/<run_id>/                  data.csv, table.csv or figure.pdf, logs, metadata.json
result/runs/<run_id>/summary.csv               status of each evaluation
result/pipelines/<run_id>/pipeline.json        complete-workflow stages and results
result/pipelines/<run_id>/summary.csv          complete-workflow stage summary
result/collections/<name>/assembled/<evaluation>/
result/collections/<name>/assembled/missing.csv
```

`run.sh` reports the stage results and assembly coverage. A complete workflow
requires the selected measurements and their expected output files. Intentional
fast-mode cuDNN omissions and unavailable historical Table 7 stages remain
explicit in the coverage report.
Named evaluation runs return 0 when every selected evaluation completed,
1 on a failure, 2 when something is pending or blocked, and 130 when interrupted.

## Troubleshooting

- `CUDA-capable device(s) is/are busy or unavailable`: the GPU is in
  exclusive-process mode and used by another process. The complete workflow's
  automatic mode waits for availability and then uses a rented A100. Select
  `--a100-backend local` when only the local GPU should be used.
- CMake older than 3.18 on `PATH`: set `THUNDERFFT_CMAKE=/path/to/cmake`.
- Toolkit not found: set `CUDACXX=/path/to/nvcc` and `CUDAToolkit_ROOT=/path/to/cuda`.

## Packaging

```bash
python3 artifact/package.py              # dist/thunderfft-<revision>.zip
```
