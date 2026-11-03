# Table 3: FFT kernel latency matrix

Owner: unassigned

Status: operational ThunderFFT-only adapter; cuFFTDx sweep/comparison and complete multi-GPU reproduction pending

Paper p. 8, Section 6.2: single A100 PCIe 80GB, H100 PCIe, and B200; FFT
sizes 64/128/256/512/1024/2048/4096; batch 65,536. Measure register-input and
shared-memory-input kernels separately. Compare matched FP16 paths and
ThunderFFT TF32 against cuFFTDx FP32. Report latency in microseconds,
`cuFFTDx_time / ThunderFFT_time` speedups, and geometric-mean speedups over
sizes. These are 84 matched comparison configurations across GPUs, precisions,
placements, and sizes. The paper selects the best cuFFTDx launch parameters
after an exhaustive sweep. [Figure 2](../figure2/README.md) consumes this data's
A100/register subset.

## Available ThunderFFT measurement

The shared source [`sample/bench/paper_fft.cu`](../../sample/bench/paper_fft.cu)
builds as `thff_paper_fft`. It selects all seven sizes, FP16 or TF32, and register
or shared input: **28 ThunderFFT configurations per GPU**. It uses the existing
benchmark kernels in `sample/bench/baseline/thunderfft_bench.h`. This driver
does not implement the cuFFTDx parameter sweep, matched speedups or complete
Table 3 reproduction. The unified adapter below completes the ThunderFFT-only
subset without requiring cuFFTDx/MathDX or other external FFT libraries. cuFFT
from the CUDA toolkit is used as an FP64 correctness reference, not as a timed
comparison implementation.

From the repository root:

```sh
# All seven N values, FP16/TF32, register/shared: 28 configurations.
./run.sh table3 --profile smoke --device 0

# Paper-sized ThunderFFT measurements for the selected GPU.
./run.sh table3 --profile paper --device 0
```

The adapter configures/builds `thff_paper_fft`, then saves its raw CSV and
stderr separately for every configuration. Smoke defaults to batch 1024,
R=256 and two trials; it is a short installation/numerical/timing check, not a
paper performance comparison. Paper defaults to batch 65536, R=1000 and three
trials. Results are written to `result/table3/<run_id>/`; neither profile runs
cuFFTDx or creates baseline speedups/geometric means. Success means the
declared 28-configuration ThunderFFT subset completed on one selected GPU.

Set `CUDACXX` and `THUNDERFFT_ARCHS` for the evaluation GPU/toolkit before
running, for example:

```sh
CUDACXX=/usr/local/cuda-12.9/bin/nvcc THUNDERFFT_ARCHS=80 \
  ./run.sh table3 --profile smoke
```

`--device` is a logical ordinal within any inherited `CUDA_VISIBLE_DEVICES`.
The adapter preserves scheduler-assigned ordinal/UUID masks, selecting just
that device; it never expands the visible GPU set. The native executable
reports the actual GPU name/compute capability. A100/H100/B200 names select
their corresponding paper target; unknown models retain their real name and
get no paper target rather than silently comparing against A100.

Environment overrides are `THUNDERFFT_TABLE3_BATCH`,
`THUNDERFFT_TABLE3_REPEATS`, `THUNDERFFT_TABLE3_TRIALS` and
`THUNDERFFT_TABLE3_TIMEOUT` (seconds per executable call, default 120). Batch
must be a positive multiple of 16; all other values must be positive integers.
The paper profile requires batch 65536 and R=1000. Any override is recorded in
metadata. A nonpositive/noisy differential measurement fails instead of being
discarded or replaced; repeat with a better-controlled GPU or a larger smoke
batch if needed. No failed candidate is silently retried or selected away.

Every CSV row must match its requested configuration and trial, have positive
finite raw/differential/e2e times, and pass the native single-FFT finite-output
and `relL2 <= 0.005` gate. The Python adapter independently checks those CSV
fields and consistency of `(t_2R_us-t_R_us)/R`. Missing/duplicate trials,
identity changes and subprocess failures/timeouts fail the evaluation while
preserving partial raw evidence. This bounded gate does not substitute for
the broader FFT validation adapter.

Build from the repository root, choosing the compiler and architecture for the
target GPU. The paper uses CUDA 12.9 / SM80 for A100 and CUDA 13.0 / SM90 or
SM100 for H100 or B200. A minimal A100 build is:

```sh
cmake -S . -B build -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
  -DTHUNDERFFT_ARCHS=80
cmake --build build --target thff_paper_fft -j2
```

The target compiles with `-O3 --use_fast_math -lineinfo`. Preserve those flags
and architecture settings in a before/after comparison. GPU commands below
are intended for the designated evaluation GPU; local CPU compilation alone
does not establish a measured result.

The CLI is:

```text
thff_paper_fft N fp16|tf32 register|shared [R=1000] [trials=3] [batch=65536]
```

For example, retain a separate raw CSV and metadata log for each configuration:

```sh
mkdir -p result/table3/manual-a100/raw
build/sample/bench/thff_paper_fft 4096 fp16 register 1000 3 65536 \
  > result/table3/manual-a100/raw/n4096-fp16-register.csv \
  2> result/table3/manual-a100/raw/n4096-fp16-register.log
python3 artifact/table3/compare_paper.py \
  'result/table3/manual-a100/raw/*.csv' --gpu A100 \
  --output result/table3/manual-a100/compared.csv
```

Repeat the executable call for N=64/128/256/512/1024/2048/4096, both precisions
and both placements. Use distinct result directories and `--gpu H100` or
`--gpu B200` for those GPUs. Do not mix GPU outputs in one comparison. The
comparison script requires batch 65536 and reports median/minimum/maximum
compute latency, signed deviation from the paper, and the separate e2e latency.
Its default 10% tolerance is a descriptive, adjustable comparison threshold;
the paper does not define a reproduction tolerance. All three GPUs require
their own measurements.

`paper-table3-xfft-targets.csv` transcribes the paper's 84 ThunderFFT timing
targets; `paper-table3-all-targets.csv` also includes its 84 cuFFTDx targets.
These are reference targets, not measurements produced by this repository run.

## Measured check: 2026-11-02

Vast.ai single-GPU rentals measured all 28 ThunderFFT forward configurations
per available GPU. Each configuration used batch 65536, R=1000 and two timing
trials per version, with the same harness and flags for the corrected library
and parent `2f0810c7d4adb2663702b0774baf832900665d4b`.

| GPU | Toolkit / architecture | Then-current latency versus paper | Then-current versus parent latency |
| --- | --- | --- | --- |
| A100 PCIe 80GB | CUDA 12.9 / SM80 | -2.78% to +6.61% | -0.61% to +0.19%; geometric mean -0.063% |
| H100 PCIe 80GB | CUDA 13.0 / SM90 | +4.38% to +15.98% | -0.49% to +0.51%; geometric mean +0.041% |
| B200 | CUDA 13.0 / native SM100 binaries built | Unmeasured: no rentable single-GPU offer | Unmeasured |

Positive percentages mean longer latency. The A100 values are close to the
paper; the H100 run remains slower, so full paper performance reproduction is
not established. With the descriptive 10% threshold, A100 has 28/28 and H100
20/28 configurations within range. That threshold is not specified by the
paper. The small before/after differences with two trials do not demonstrate
an optimization or a regression. At that revision, forward FFT bodies were
retained while inverse and API correctness were repaired.

Before timing, each GPU passed the 500-case FFT correctness matrix, 88 cases
under each of Compute Sanitizer memcheck, synccheck and racecheck, and eight
small convolution checks. The full FFT matrix's maximum relative L2 errors
were FP16 `0.0006623171` on both GPUs, TF32 `0.0026404948` on A100 and
`0.0026454357` on H100, against a `0.005` gate. These are bounded-input results.

Raw data and billing evidence are local generated outputs under
`result/implementation-completion-20261102/performance/measurement/gpu-run1/`
(A100) and `gpu-run2/` (H100). The same performance directory contains
`a100-paper-comparison.csv`, `h100-paper-comparison.csv` and before/after
summaries. The frozen source archive SHA256 is
`07d99326607c586edd5bf448f86bd40a95f27b4c23d47dc12d1d3c6b442aac49`.
Both instances were destroyed after measurement. The H100 active measurement
clocks and the paper's clock settings were not recorded together, so the
remaining H100 gap cannot be attributed to a clock difference from this run.
The cuFFTDx exhaustive sweep and full convolution comparison remain separate
artifact tasks.

## N=1024 SM80 follow-up: 2026-11-03

The float N=1024 core now reuses the twiddle table already prepared by
`ThunderFFTInitialize<float>(1024)` instead of evaluating sin/cos for each FFT.
This change applies to SM80 only, in both directions; the measured forward
register and shared-input paths use the same core. Native SM90 and SM100 builds
keep their previous paths; SM80 PTX JIT-compiled on a newer GPU would use the
cached path.

Matched local A100 PCIe 80GB runs used CUDA 12.9, native SM80,
`-O3 --use_fast_math -lineinfo`, batch 65536 and R=1000. Register runs interleaved
three invocations per version, each with three trials. Shared runs used two
invocations per version, each with three trials. The table reports medians;
percentages describe latency, so negative values mean faster.

| TF32 input placement | Before (us) | After (us) | Change | Paper (us) | After versus paper |
| --- | ---: | ---: | ---: | ---: | ---: |
| Register | 137.240570 | 126.527481 | -7.81% | 127.85 | -1.03% |
| Shared | 160.784378 | 155.632126 | -3.20% | 155.34 | +0.19% |

The register baseline used 255 registers/thread without spills. Shared load
and store conflict-counter/wavefront ratios were 0.065% and 0.070%; these are
memory-counter ratios, not percentages of kernel time. Static bank mapping
and ideal/actual shared wavefront counters did not identify excessive bank
serialization. Global loads and stores used eight 32-byte sectors per
256-byte full-warp float2 access. Register caps introduced spills and slowed
the tested candidates. The accepted change reuses the existing table without
changing launch geometry, synchronization or the timing harness.
In the accepted register benchmark's compiled repeat loop, 32 COS and 32 SIN
instructions disappear; the twiddle loads are hoisted outside that loop.
The cached version still uses 255 registers/thread and introduces a 64-byte
stack with 60 bytes each of spill loads/stores. Thus the observed improvement
comes with a spill tradeoff, rather than from lower register use. Native SM90
and SM100 builds were checked to retain identical SASS across all 116 compiled
functions compared with the preceding build.

The production build passed 500 full-matrix correctness cases and 112 actual
benchmark-entry checks, including a `5e-3` maximum per-transform relative L2
gate, finite outputs and output guards. Memcheck, initcheck and synccheck each
passed the 88-case quick matrix. A short repeated N=1024 register benchmark
also passed racecheck with no hazards. Initcheck here covers global-memory
initialization; it is not a proof of shared-memory initialization.

Raw matched data, profiler reports, build metadata and validation outputs are
under local generated directory `result/n1024-performance-fix-20261103/`, with
`local-matched/`, `local-shared-matched/`, `profile/` and `validation/`
subdirectories. Clocks and power settings were not changed; intermittent jobs
from another account were present on the GPU. The repeated interleaved A/B
observations support this local improvement, with those environmental limits.
The separate single-FFT end-to-end metric did not show an improvement.
Other Table 3 configurations were not retimed, and the cuFFTDx sweep remains
pending. This follow-up does not establish full paper reproduction.

## Timing and correctness definitions

The driver reports CUDA-event times in microseconds. For `R=1000`, it records
one R-repeat and one 2R-repeat kernel and computes
`compute_us = (t_2R_us - t_R_us) / R`. Input/output global memory traffic occurs
once per kernel; the FFT repeats in registers or with shared-memory loads and
stores according to the selected placement. The difference removes most fixed
load/store and launch cost.

Before timing, the driver runs three extra R-repeat warmup kernels. Each trial
then warms the R and 2R kernels once before recording their individual times.
Those warmups are outside reported intervals. The CSV retains both raw times;
nonpositive or nonfinite differential times fail the run.

Timed input follows the existing sample's real sinusoid
`sin(2*pi*(i % N)/N)/sqrt(N)`, with zero imaginary part. Input and output global
buffers are separate, so each kernel launch starts from the same input. Repeated
FFTs inside a timing kernel can grow to nonfinite values; their final values
are not used for correctness validation.

Accuracy is checked separately on a deterministic complex LCG input, batch 64,
rounded to the selected storage precision. A single forward FFT is compared
against cuFFT FP64 applied to that exact rounded input. Nonfinite outputs or
relative L2 error above `5e-3` fail before performance timing. This is a limited
measurement gate; the broader correctness suite covers additional inputs,
directions and batch boundaries.

`e2e_out_of_place_us` measures a separate single-FFT kernel, with one warmup and
ten timed launches. It includes global input/output movement and must be kept
separate from Table 3's differential compute latency. The metadata log records
GPU name, compute capability, driver/runtime/compiled CUDA and cuFFT versions,
and the extra warmup count. Also retain exact build flags, Git revision, dirty
state, GPU clocks/power settings and raw commands in the run manifest.

Raw CSV columns are:

```text
N,precision,placement,batch,inside_repeats,trial,t_R_us,t_2R_us,compute_us,e2e_out_of_place_us,single_fft_relL2,shared_bytes
```

## Remaining work to assign

- Implement cuFFTDx device-level register/shared paths and an exhaustive legal
  launch-parameter sweep. Retain every candidate and the winner-selection rule.
- Confirm the paper's cuFFTDx/MathDX version, swept parameter ranges, direction,
  timing statistic and other missing experimental settings. R=1000 comes from
  the existing repository helper; the PDF does not explicitly specify it or the
  three added warmups.
- Validate each implementation using matched input, reference and tolerances
  before reporting timings or speedups. Preserve separate correctness input and
  performance input; never validate the overflowed repeated-loop output.
- Run the unified adapter independently on the remaining target GPUs. Its
  metadata records one actual GPU; a successful single-GPU run does not
  establish multi-GPU reproduction.

## Results and completion

The adapter writes these outputs under `THUNDERFFT_RESULT_DIR`:

- `data.csv`: every validated timing trial, with canonical columns
  `gpu,N,batch,placement,library,precision,trial,time_us,e2e_out_of_place_us,rel_l2,passed,inside_repeats,t_R_us,t_2R_us,shared_bytes,raw_csv`.
  `library=ThunderFFT`; `time_us` is the differential compute latency.
- `table.csv`: per-configuration median/min/max latency, separate median e2e
  latency and maximum validation error. Paper targets/deviations appear only
  for paper-sized measurements on a recognized GPU. These targets are
  transcribed references, never replacement measurements.
- `metadata.json`: expected/completed coverage, explicit missing cuFFTDx
  comparison, exact executable argv, timeout/exit status, GPU/software metadata,
  source/binary SHA256, Git revision/dirty state and CMake build flags.
- `raw/*.csv`, `raw/*.log`, `commands.txt`, `configure.log`, `build.log`:
  auditable raw measurements, native metadata and build output.

[`Figure 2`](../figure2/README.md) consumes `data.csv` from the same runner's
Table 3 dependency, retaining its A100/register subset. A smoke-derived figure
is a smoke result; it is not relabeled as paper-sized data.

Complete paper Table 3 comparison still requires matched cuFFTDx tuning
candidates and winner selection across all three GPUs. Until supplied, the
metadata records `full_paper_comparison_complete=false` and the summary's
`cufftdx_comparison=not_measured`. See the [runner contract](../README.md).

Parser/validation contract checks run without a GPU:

```sh
python3 -m unittest discover -s artifact/table3 -p 'test_*.py' -v
```
