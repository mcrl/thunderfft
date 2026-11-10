ThunderFFT (xFFT) - Artifact Evaluation README
==============================================

Paper:    ThunderFFT: Accelerating FFT with Tensor Cores (PPoPP'27)
Artifact: thunderfft.zip

- Start with Section 2 to generate all tables and graphs with one command.
- Use Section 3 for individual experiments and additional options.
- xFFT in the paper is ThunderFFT in this artifact.
- This guide assumes the prepared evaluation account. For another machine,
  see artifact/README.md and artifact/cloud/README.md for setup requirements.


1. Access to the evaluation machine
-----------------------------------

- The evaluation host, SSH port, assigned account and password are supplied
  separately to reviewers.

  $ ssh -p <port> <assigned-account>@<host>

- Use only your assigned reviewer account.
- GPU 0: NVIDIA A100 80GB PCIe (the paper's A100). Use this GPU.
- GPU 1: Tesla V100 16GB. Not used by the artifact; do not select it.
- Software: CUDA 12.9, GCC 9.4, Python 3 with venv.
- All reviewer accounts share one A100 in Exclusive_Process mode.
  Check nvidia-smi before local experiments; retry later if it is busy.
  Timing measurements require exclusive use of the GPU.


2. Getting started
------------------

2.1 Setting (once per account, no root needed)

- For a Git checkout, start in the repository root and skip unzip/cd below.

  $ unzip thunderfft.zip
  $ cd thunderfft
  $ bash artifact/setup.sh
  $ source ~/.thunderfft_env
  $ export CUDA_VISIBLE_DEVICES=0
  $ bash artifact/check.sh

- Installs under $HOME: ~/venv (matplotlib, CMake, CPU PyTorch headers),
  NVIDIA MathDX 25.06.1 (cuFFTDx), and cuDNN 9.12.
- Requires Internet access to NVIDIA, PyPI and GitHub.
- check.sh runs CPU-only checks.
- Run the following commands from the thunderfft/ directory.


2.2 One-step generation: shorter version (default)

  $ ./run.sh

- Rough total-time estimate: 20-35 minutes on the prepared evaluation account.
  The new parallel workflow has not been measured end to end.
- Generates every Table 2-7 in paper-style PDF, PNG and HTML, plus
  Figure 2-5 PDFs. This shorter version checks that all tables/graphs render.
- Uses the paper workload settings and fresh measurements:
  - Table 3: remeasure saved GPU-specific cuFFTDx parameters.
  - Figures 4/5: skip the slow cuDNN filter tails; measure all other libraries.
  - Table 5: measure alpha=1 error for all four implementations, then
    recheck the known finite boundary and expand the window if needed.
- Checks all required GPU offers and their combined cost before starting.
  After acquiring and preparing every cloud server, runs A100 evaluations,
  H100 Tables 3/4 and B200 Tables 3/4 in parallel, then assembles the results.
  Measurements start only when every GPU stage is ready.
- H100/B200 use the supplied cloud credentials and incur rental charges.
  If the local A100 stays busy for 120 seconds, auto mode rents an A100.
- ./run.sh all is equivalent. No manual merging or plotting is required.
- The command prints the collection name (full-<run_id> by default).

Progress display (illustrative):
  A100 | 1/10 steps | Table3 36/56 configs
  H100 | 0/2 steps  | Table3 44/56 configs
  B200 | 0/2 steps  | Table3 40/56 configs

- On a terminal, each GPU keeps one updating line instead of accumulating
  progress messages. Named/local runs show one GPU line.
- Fractions show completed steps and work in the current experiment, not
  estimated remaining time. Batched native sweeps advance after validation;
  Table 5 counts phase/size groups, including any boundary rechecks.
- --progress auto is the default. Redirected output uses plain stage changes;
  --progress plain requests these explicitly, and --progress off hides the
  progress display. No additional tqdm installation is needed.
- Detailed logs remain in each experiment's run.log, pipeline a100.log and
  cloud-worker/jobs.log. Controller messages are saved to
  result/pipelines/<run_id>/pipeline.log, or result/runs/<run_id>/runner.log
  for a standalone local run. progress.json files retain current snapshots.
- Standalone --backend vast commands retain their existing text logs.

Final outputs:
  result/collections/<name>/assembled/table2/ ... table7/
  result/collections/<name>/assembled/figure2/figure.pdf
  result/collections/<name>/assembled/figure3/figure.pdf
  result/collections/<name>/assembled/figure5/figure.pdf
  result/collections/<name>/assembled/figure4/figure.pdf
  result/pipelines/<run_id>/summary.csv

- Each assembled table directory contains table.pdf, table.png, table.html
  and the original CSVs. Open table.html for the paper's row/column layout:
  each original numeric column expands to Paper / New / delta (%) columns,
  with one empty spacer column between groups.
  paper-comparison.csv contains signed differences and absolute relative errors;
  comparison.md links to the preview. Table 5 keeps four implementations per N.
- Latest manuscript numbering: Figure 4 = 2D, Figure 5 = 1D. Use a new
  collection after this update; old figure parts keep their original IDs.
  Figures 4 and 5 both measure A100, matching the supplied PDF.
- Expected exit status: 0.
- Default Figures 4/5 show gaps and annotations for skipped cuDNN filters.
  Table 5 reports local boundary coverage. Table 7 estimates its first row
  from Table 6 tcFFT while retaining utilization. Its second row measures a new
  stage-wise shared-memory radix-4 reconstruction, marked ‡. Coefficients are
  prepared once per launch outside its repeated timing interval; the original
  paper-stage identity remains unconfirmed (see Section 4).


2.3 One-step generation: full version

  $ ./run.sh --full

- Rough total-time estimate: 50-70 minutes on the prepared evaluation account.
  The new parallel full workflow has not been measured end to end.
- Generates the same set of table files and Figure 2-5 PDFs.
- Runs the complete searches and measurement grids:
  - Table 3: search all cuFFTDx parameter candidates on A100, H100 and B200.
  - Figures 4/5: measure every requested cuDNN filter.
  - Table 5: scan every exponent 0..127 for all four implementations.
- Equivalent options:
  --cufftdx-mode all --cudnn-mode full --table5-range-mode full
- Each invocation creates a new collection automatically.


2.4 Approximate time per step

- Estimates assume dependencies and local builds are already available.
- Cloud times include startup, builds, measurements, download and cleanup.
- GPU stages overlap after all cloud servers are prepared; total elapsed time
  depends on preparation and the slowest GPU stage, not the sum of all rows.
- Initial setup/builds, waiting at the readiness gate and waiting for exclusive
  GPU availability can add time.

| Step (A100 steps run in order; GPU stages run in parallel) | GPU | Default | --full |
| --- | --- | --- | --- |
| Validation | A100 | Under 1 minute | Under 1 minute |
| Table 3 | A100 | 4-6 minutes | 8-12 minutes |
| Table 2 | A100 | Under 1 minute | Under 1 minute |
| Table 4 | A100 | Under 1 minute | Under 1 minute |
| Table 5 | A100 | 3-5 minutes (rough estimate) | 22-28 minutes (rough estimate) |
| Table 6 | A100 | Under 1 minute | Under 1 minute |
| Table 7 | A100 | Under 1 minute | Under 1 minute |
| Figure 3 | A100 | Under 1 minute | Under 1 minute |
| Figure 4 | A100 | 2-4 minutes | 9-12 minutes |
| Figure 5 | A100 | 1-2 minutes | 2-3 minutes |
| Tables 3/4, complete cloud stage | H100 PCIe | 8-12 minutes | 10-15 minutes |
| Tables 3/4, complete cloud stage | B200 | 8-12 minutes | 10-15 minutes |
| Final tables and Figure 2-5 PDFs | CPU | Under 1 minute | Under 1 minute |
| Total elapsed time | | About 20-35 minutes (rough, unmeasured) | About 50-70 minutes (rough, unmeasured) |

- Figure 2 reuses Table 3 measurements during final assembly.
- Both totals, H100/B200 preset-stage times and the expanded full Table 5
  time are estimates. The new parallel default and expanded full workflows
  have not been measured end to end on all three GPUs.
- Table 5 time allowances are carried over from the earlier implementation.
  The 4m03s boundary measurement at 4344f24 and retained 18m26s full scan
  used batch 65536 and cuFFT FP32. The current batch-64/cuFFTDx-FP32
  variant has not been timed; these are not measurements of that variant.
- Measured checks of the individual changes are listed in Section 5.


3. Individual experiments and options
--------------------------------------

3.1 Small local checks (smoke profile)

  $ ./run.sh --list
  $ ./run.sh validation table2 table3 --jobs 8

- Named evaluations default to local execution with small smoke workloads.
- Expect exit status 0 and completed entries in result/runs/<run_id>/summary.csv.
- Validation: 88 FFT cases and 18 host API checks; relative L2 <= 5e-3.
- Smoke Table 3: all seven sizes, batch 1024.
- Smoke timings are not comparable to the paper.
- To check local plotting with small workloads:

  $ ./run.sh figure2 figure3 figure4 figure5 --jobs 8

- Figure 2 runs its Table 3 dependency automatically.
- ./run.sh --profile smoke (without evaluation names) instead runs the
  whole A100/H100/B200 workflow, including cloud rentals.


3.2 Selected paper experiments on the local A100

  $ ./run.sh table3 --profile paper
  $ ./run.sh table5 --profile paper
  $ ./run.sh figure4 figure5 --profile paper

- Specify one or more evaluation names; add --profile paper for paper settings.
- Table 3 paper settings include batch 65536, R=1000 and three trials.
- To reuse one Table 3 run for Table 2 and Figure 2:

  $ ./run.sh table3 table2 table4 figure2 --profile paper

- To run every local A100 experiment:

  $ ./run.sh all --backend local --profile paper --jobs 8

- The local-only command does not collect H100/B200 results.
- Per-experiment outputs: result/<evaluation>/<run_id>/.
- Detailed grids and environment options:
  artifact/<table|figure>/README.md


3.3 Choose the exhaustive steps individually

Table 3: saved parameters or full parameter search

  $ ./run.sh table3 --profile paper --cufftdx-mode preset
  $ ./run.sh table3 --profile paper --cufftdx-mode all

- Preset mode rechecks correctness and measures one saved candidate per
  size/precision/placement, with the same batch, repeats and trials.
- The 84 saved candidates were selected in the validated A100/H100/B200
  full search. Provenance: artifact/table3/cufftdx_presets.json.
- Compilation still builds all candidates. This option does not affect
  Figure 4's tiled cuFFTDx implementation.

Figures 4/5: short or complete cuDNN filter grids

  $ ./run.sh figure4 figure5 --profile paper --cudnn-mode fast
  $ ./run.sh figure4 figure5 --profile paper --cudnn-mode full

- Fast mode: cuDNN odd filters 3..25 for Figure 5, 3..11 for Figure 4.
- Other libraries retain the full requested filter grid.
- Full mode restores all filters and disables optional SLOW_RATIO cutoffs.
- THUNDERFFT_CUDNN_MODE accepts fast or full; CLI options take precedence.

Table 5: boundary recheck or full exponent scan

  $ ./run.sh table5 --profile paper --table5-range-mode boundary
  $ ./run.sh table5 --profile paper --table5-range-mode full

- Four rows per N: xFFT FP16, cuFFTDx FP16, xFFT TF32 and cuFFTDx FP32.
  cuFFTDx requires MathDX. The default paper batch is 64.
- Both modes freshly measure alpha=1 error against cuFFT FP64 for every row.
- Boundary mode measures alpha=2^e at e=k-1..k+2 around the saved last-finite
  exponent k; it expands if the boundary shifts or is not bracketed.
  Saved values only select the starting window; reported values are fresh.
- Default paper: 16 accuracy rows + 64 finite checks, unless expansion is needed.
- Full paper: 16 accuracy rows + 2048 finite checks (exponents 0..127).
- Presets/provenance: artifact/table5/boundary_presets.json.
- THUNDERFFT_TABLE5_RANGE_MODE accepts boundary or full; CLI takes precedence.

- The same options work on the whole workflow, e.g. ./run.sh --cufftdx-mode all.
- Use separate collections for different profiles and measurement modes;
  assembly rejects incompatible parts.


3.4 Cloud execution and workflow controls

  $ ./run.sh --dry-run                  # offline preview; no GPU/API work
  $ ./run.sh --cloud-plan               # current offers/cost; no rental
  $ ./run.sh --a100-backend local       # require local A100; still rent H100/B200

- Default cloud config: ~/cloud_key/thunderfft/config.json.
  On the prepared reviewer account, API and SSH keys are already provided
  in ~/cloud_key/thunderfft/. For another machine, follow artifact/cloud/README.md.
  --cloud-config FILE selects another config; keep credentials outside the artifact.
- The configured budget covers all concurrent rentals, including startup,
  readiness waiting, download, cleanup and billing-delay allowances.
- A100 options: auto (default), local, vast; --local-wait-seconds changes
  the 120-second local wait. Other users' processes are never stopped.
- The whole workflow checks every required offer and the combined cost,
  acquires all cloud servers, and prepares them in parallel. Only after every
  server is ready and local A100 availability is rechecked do the
  A100/H100/B200 measurements start together.
- If any acquisition or preparation fails, all acquired instances are cleaned
  up and no benchmark starts. The whole workflow does not retry startup
  automatically. Standalone --backend vast commands use one GPU at a time
  and retain up to two startup attempts by default.
- After measurements, results are downloaded and each cloud instance is
  destroyed; final assembly waits for all GPU stages to finish.
- Default build/run limits: 1200 seconds per H100/B200 stage, 3600 seconds
  for a rented A100. --cloud-stage-seconds / --a100-stage-seconds change
  them within the config's execution_seconds upper bound.
  Startup, download and cleanup have separate allowances.
- --collection NAME chooses a collection name; use a new name for a whole run.
- --result-dir DIR and --build-dir DIR select output and build locations.
- Independent GPU parts, importing results and offline assembly:
  artifact/README.md and artifact/cloud/README.md.
  These extra steps are already automatic with ./run.sh.


4. Outputs and coverage
------------------------

- Each evaluation: result/<evaluation>/<run_id>/
  (data.csv, table.csv or figure.pdf; metadata.json; status and logs).
- Per-evaluation status: result/runs/<run_id>/summary.csv.
- Final merged outputs: result/collections/<name>/assembled/.
- Tables 2-7: table.pdf (print), table.png (preview), table.html (browser);
  Each original numeric column becomes three columns: Paper, New and delta
  (%), with one blank spacer column between groups. Original row labels and
  grouped headers are retained. In HTML, select one view or show all three;
  scroll horizontally for wide tables. Delta = 100 * (New - Paper) / abs(Paper).
  paper-comparison.csv also reports absolute relative error and measurement
  status. Original CSVs/notes remain intact; table-render.json records hashes.
  Reference numbers are from the supplied PPoPP27_chaewon__FFT_ (4).pdf.
  Missing references, missing measurements and censored bounds have no exact
  delta. Table 5 includes references for all four implementations and uses
  stacked accuracy / finite-range panels, matching the supplied PDF.
- Whole-workflow status: result/pipelines/<run_id>/pipeline.json and summary.csv.

| Output | What to check |
| --- | --- |
| Validation | All 500 paper FFT cases and 18 host API checks pass; relative L2 <= 5e-3. |
| Table 2 | table.html: model rows and the measured cuFFTDx row; paper/new differences. |
| Table 3 | table.html: register/shared, precision and GPU groups; latency and speedup. |
| Table 4 | table.html: GPU columns; peak, ideal MMA and unit-kernel TFLOPS rows. |
| Table 5 | table.html: four implementation columns per metric; alpha=1 error and finite range. |
| Table 6 | table.html: library rows, time and speedup columns. |
| Table 7 | table.html: stage rows, time/utilization/speedup; estimates, reconstruction (‡) and unchecked timings (†) marked. |
| Figures 2-5 | figure.pdf and summary.csv: compare measured trends with the paper. |

- Successful assembly retains the following explicit coverage limits:
  - Figures 4/5 default: skipped cuDNN tails have no fabricated values;
    policy_skipped is recorded and coverage_complete=false.
    Figure 5 means cover 12 sizes for cuDNN and 26 for other libraries.
    Figure 4 summary.csv contains per-filter means over trials, with skipped
    cuDNN cells empty; its plot uses a linear time axis.
    --full restores all 26 cuDNN sizes.
  - Table 5 default: boundary_recheck confirms a local finite/non-finite
    bracket within the displayed window, not a global maximum over unmeasured
    exponents. Full mode records full_profile_scan.
  - Table 7, both modes: the first row is estimated from validated Table 6
    tcFFT at batch 65536 on A100, assuming retained Tensor Core utilization:
    N64 TF32 time = N256 FP16 time * (12288/65536) * (312/156)
                 = N256 FP16 time * 0.375.
    It uses measured utilization, not a fixed 70%. No extra GPU work is needed.
    The estimate is labeled and recorded in metadata/coverage; without compatible
    Table 6 tcFFT evidence it stays blank.
    The second row measures shared_stage_radix4, a new reconstruction using
    shared-memory loads/stores at each radix-4 stage without register-fragment
    permutation. Its optimized 8 KiB layout and vector accesses retain all
    48 MMAs per warp. It prepares 21 invariant coefficient pairs once per kernel
    launch, before repetition; (t_2R - t_R) / R excludes that preparation.
    The measured scope is repeated computation, not full FFT launch latency,
    and is recorded in metadata and the ‡ note. Source/binary hashes identify
    the reconstruction; the original paper source and its 0.0120 ms experiment
    have not been recovered.
    The reduced-communication row selects register_branch_cached, a complete
    register FFT with the 51e346c data mapping and three conditional swaps from
    checkpoint 652808b, before reduce divergence. Each condition covers eight
    disjoint quartets per stage, preserving lane-dependent branches, every
    permutation and all 48 MMAs. This is a derived implementation. Its 28
    coefficient pairs are prepared before repetition, matching the shared candidate's
    timing boundary. The § note identifies this reconstruction.
    Default collection measures four complete candidates in timing-only mode
    (unchecked, marked †). THUNDERFFT_TABLE7_TIMING_ONLY=0 validates all four
    at R=1. Partial historical kernels require explicit diagnostic selection and
    cannot fill current paper rows. A missing measurement for either selected
    reconstruction fails the complete pipeline.
- The pipeline can report completed_with_coverage_limits with exit status 0.
- Absolute timings depend on GPU clocks/load. Smoke timings are not paper results.

To regenerate tables from an existing paper collection, without GPU work:

  $ ./run.sh table2 table3 table4 table5 table6 table7 --assemble --collection NAME --profile paper

- Replace NAME with the collection name printed by the original run.
- Table 7-only assembly reuses any collected Table 6 tcFFT evidence without
  rerunning GPU measurements. Missing measurements stay blank; the projected
  first row is explicitly an estimate. An incomplete collection still renders
  and returns exit status 2.
- If rendering fails, the previous assembled output is retained.


5. Recorded validation
-----------------------

Checks below were run on 2026-11-09 at the stated source revisions.

| Revision / mode | Measured result |
| --- | --- |
| 6864641: former full workflow | A100/H100 PCIe/B200, 61m52s, exit 0; all tables and Figure 2-5 PDFs assembled. |
| 317593c: Table 3 presets, A100 | 56 configurations passed; Table 2/Figure 2 and offline assembly passed. Measurement 4m56s + build/setup 2m09s. |
| a0f5755: fast cuDNN, A100 paper | 1D (now Figure 5): 1m28s; 2D (now Figure 4): 2m42s, including build/setup and plotting. All 192 / 249 raw rows passed correctness. |
| 4344f24: Table 5 boundary, A100 paper | 4m03s; 80 raw rows, 16 summary rows. All 16 accuracy values and 12 prior boundaries matched retained results. |

- Historical full workflow: 500 FFT cases and 18 host API checks passed;
  all 168 Table 3 cells and all three Table 4 GPU rows were collected.
  An SSH startup failure was cleaned up and retried automatically.
  At that revision, Table 7 had two unfilled historical stages.
- Historical Table 5 measured four accuracy implementations but only three
  finite-range implementations. The new full mode also scans cuFFTDx FP16;
  its full paper runtime has not been measured.
- Table 3 presets reduced measured A100 measurement time by 43%.
  The exhaustive path also passed a small eight-candidate smoke run.
  H100/B200 preset provenance matches retained full-search results;
  the new preset path has not been rerun on those rented GPUs.
- Fast Figures 4/5: combined 4m10s versus retained full-filter 11m18s
  on the same A100 (about 63% shorter). Full smoke restored every filter,
  including cuDNN filter 53; both modes passed offline assembly.
- Table 5 boundary: 4m03s versus retained three-implementation full scan
  18m26s (about 78% shorter, with different exponent coverage).
  cuFFTDx FP16 boundaries: 11/9/8/7 for N=64/256/1024/4096.
  Full smoke: 32 raw rows, four summary rows, 2.6 seconds.
  Both modes passed offline assembly and independent evidence checks.
- Latest implementation checks passed 556 CPU tests.
- These checks validate the listed revisions/modes; they are not a measured
  end-to-end run of the combined new default or expanded full workflow.


6. Claims supported by the measurements
----------------------------------------

  C1  Correctness: FP16/TF32 forward/inverse C2C FFTs for N=64..4096
      match cuFFT FP64 within relative L2 5e-3 (validation).
  C2  Performance model: A100 theoretical model and measured cuFFTDx
      utilization (Table 2).
  C3  FFT latency on A100/H100/B200 and speedup over cuFFTDx for register
      and shared-memory input (Table 3); A100 register comparison (Figure 2).
  C4  Tensor Core efficiency: ideal TF32 MMA throughput and 64-point
      unit-kernel throughput on A100/H100/B200 (Table 4).
  C5  Numerical accuracy and finite dynamic range of xFFT FP16/TF32,
      compared with cuFFTDx FP32 and FP16 (Table 5).
      Default finite-range coverage is a local boundary recheck.
  C6  256-point FP16 FFT performance versus tcFFT, FlashFFTConv and cuFFTDx
      (Table 6).
  C7  End-to-end FFT performance versus a communication-only baseline,
      cuFFT and cuFFTDx (Figure 3).
  C8  1D/2D convolution performance versus the cuDNN, FlashFFTConv, cuFFT
      and cuFFTDx baselines (Figures 4/5). Each library is checked against
      a direct reference; default cuDNN comparisons use the selected filters.


7. Troubleshooting and further details
---------------------------------------

- GPU busy/unavailable: another account is using the A100. Retry local
  experiments later. The whole workflow waits, then rents an A100 in auto
  mode; use --a100-backend local to require the shared machine instead.
- Cloud GPU unavailable: we are sorry for the interruption. Provider GPU
  inventory is outside our control. The runner prints a brief explanation
  and retry/local-only commands. Check offers with ./run.sh --cloud-plan,
  then rerun ./run.sh when available, or run only A100:

  $ ./run.sh all --backend local --profile paper

  Add --full if you want the complete searches and measurement grids.
  This generates the local Figure 2-5 outputs; Tables 3/4 lack H100/B200
  measurements. No rentals or benchmarks start if the initial availability
  check fails. The workflow does not automatically wait for cloud inventory
  or skip a missing GPU.
- CMake older than 3.18: set THUNDERFFT_CMAKE=/path/to/cmake.
- CUDA not found: set CUDACXX=/path/to/nvcc and CUDAToolkit_ROOT=/path/to/cuda.
- Table 6 / Figure 5 need GitHub access for pinned tcFFT/FlashFFTConv sources.
- Library overview: README.md.
- Runner, collections and assembly: artifact/README.md.
- Per-experiment settings: artifact/<table|figure>/README.md.
- Cloud setup and controls: artifact/cloud/README.md.
