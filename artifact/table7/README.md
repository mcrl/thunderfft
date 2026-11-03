# Table 7: optimization ablation

The measurement sources are self-contained in this folder:

```text
ablation/benchmark.cu                 correctness and timing driver
ablation/kernels/recovered.cuh        two historical TF32 implementations
ablation/kernels/provenance.json      source commits, hashes and adaptations
variants.json                        available and missing implementations
build.sh                             compile without launching a GPU workload
measure.py                           selection, execution and table export
collect_profile.py                   optional Nsight Compute diagnostic
assemble.py                          combine collected parts without GPU work
run.sh                               root-runner adapter
```

The available candidates are `register_branch` and `register_uniform`, recovered
from the same historical commit. Their fragment and input/output layouts and
loop unrolling also differ. They are not established as the exact original
paper variants, or as a comparison that isolates only branch divergence.
The uniform wrapper's omitted second group of output FFTs and the branch
wrapper's missing output barrier have been repaired; provenance lists these
adaptations. Kernel arithmetic and original mappings are otherwise retained.

The original `radix16` and `radix4_compute` sources have not been recovered.
They remain explicit missing entries. With no selection, `./run.sh table7`
returns pending (78 from the adapter, 2 from the root runner) before building
or launching GPU kernels. It does not silently replace the paper's four rows
with the two available candidates. In a named collection, the default selection
is the two available candidates, while assembly retains all four rows and marks
the uncollected entries as missing. Ordinary default batches exclude this task.

## Build without measuring

```sh
CUDACXX=/usr/local/cuda-12.9/bin/nvcc THUNDERFFT_ARCHS=80 \
  bash artifact/table7/build.sh
./build_artifact/table7/thff_ablation --help
python3 artifact/table7/test_contract.py
python3 artifact/table7/test_profile.py
python3 artifact/table7/test_assemble.py
```

The build uses only CUDA Toolkit and a C++17 compiler. It needs no cuFFTDx
parameters, external FFT checkout or GPU execution. Native compilation and
CPU contracts have been checked; GPU numerical validation and timing have not
been run for these recovered candidates.

## Run an explicitly selected subset

These commands launch GPU work when the evaluator chooses to measure:

```sh
THUNDERFFT_TABLE7_VARIANTS=register_branch,register_uniform ./run.sh table7
THUNDERFFT_TABLE7_VARIANTS=register_branch,register_uniform \
  ./run.sh table7 --profile paper
```

The smoke protocol uses N=64, batch1024, R256 and two trials; the paper-sized
protocol uses batch65536, R1000 and three trials. `THUNDERFFT_TABLE7_BATCH`,
`THUNDERFFT_TABLE7_REPEATS` and `THUNDERFFT_TABLE7_TRIALS` permit explicit
smoke overrides. Paper-sized mode requires batch65536 and R1000. These are
artifact protocol choices: the supplied Table 7 does not specify its original
GPU, precision or repetitions. The utilization denominator is the A100 hardware
peak of 156 TFLOPS, as confirmed for this experiment. The recovered code
uses TF32 and requires compute capability 8.0 or newer. Choose a matching
native architecture build and the root runner's logical `--device` selection.

Before timing either variant, independent bounded complex Gaussian and tone
inputs at batch64 undergo R=1 comparison against a CPU FP64 DFT of the represented
FP32 input. Every transform must have relative L2 at most 0.005, every output
must be finite, and the 16-element guards on both sides must be intact.

Each timed launch receives the same immutable input through a device copy
outside the event interval. After three warmups for each repeat count, compute
time is `(time(2R)-time(R))/R`. Repeated outputs may overflow and are not used
as an accuracy check. Invalid numerical gates, guards or nonpositive timing
differences fail the evaluation. Results go under root `result/table7/<run_id>/`
and include raw CSV/logs, `data.csv`, `table.csv`, `coverage.json`, source/binary
hashes, compiler commands and the exact measurement commands. Speedup against
the missing radix-16 baseline is left blank.

## Collect separate parts and assemble

Each variant can be timed in a separate execution. Utilization is calculated
from those timing CSVs, so it requires no additional GPU execution or profiler
CSV. The following collection commands launch GPU work; the final assembly
command reads files only. Keep the same compiled binary, source files, GPU and
protocol settings for all parts of one table.

```sh
THUNDERFFT_TABLE7_VARIANTS=register_branch \
  ./run.sh table7 --collection a100-ablation --profile paper --part branch-timing
THUNDERFFT_TABLE7_VARIANTS=register_uniform \
  ./run.sh table7 --collection a100-ablation --profile paper --part uniform-timing

./run.sh table7 --collection a100-ablation --profile paper --assemble
```

The measurement phase is `timing`. The former `profile` and `both` phases and
`THUNDERFFT_TABLE7_NCU=1` are rejected before launching GPU work. Within a
collection, `THUNDERFFT_TABLE7_VARIANTS` may be omitted to collect the two
available implementations.

Collected parts are retained under
`result/collections/a100-ablation/parts/table7/`. Assembly writes
`result/collections/a100-ablation/assembled/table7/table.csv`, the combined
timing `data.csv`, and metadata listing collected and missing measurements.
Run assembly at any point: missing timing cells remain blank, and utilization
is derived for collected A100 timing rows. The result stays `partial` until
all required cells are present. The two
unrecovered implementations remain missing even after the available candidates
have been measured. Paper identity remains unconfirmed.

The assembler checks the numerical gates, raw timing CSVs and hashes.
It rejects mixed smoke/paper protocols, settings, source/binary hashes, GPU
identities or runtime/compiler versions. Identical
duplicates are harmless; overlapping measurements with different values require
choosing one result, rather than silently overwriting or averaging them.
Parts created before native GPU UUID logging was added cannot be combined
through this strict assembly contract.

Existing parts collected with the former `both` phase can contribute their
validated timing rows; their profiling counters are ignored. A profiling-only
part does not supply timing or Table 7 utilization.

For direct offline use with existing successful result directories:

```sh
python3 artifact/table7/assemble.py path/to/branch-part path/to/uniform-part \
  --profile paper --output result/table7-assembled
```

## Utilization from operation count and time

For each recovered TF32 implementation, one FFT issues 6144 Tensor Core MMA
FLOPs: three stages, 16 MMA instructions per warp per stage, 2048 FLOPs per
`m16n8k8` instruction, and 16 FFTs per warp. The count includes the real MMA
operations used to represent the complex FFT arithmetic.

The output uses the median of the valid per-trial compute times:

```text
mma_flops_per_fft = 3 * 16 * (2 * 16 * 8 * 8) / 16 = 6144
achieved_tflops = 6144 * batch / (median_compute_us * 1e6)
utilization_pct = 100 * achieved_tflops / 156
```

`compute_us = (time(2R) - time(R)) / R` already measures one batch iteration;
the utilization calculation does not multiply or divide by R again. The
denominator is the A100 hardware peak of **156 TFLOPS**, rather than the
147.22 TFLOPS ideal-MMA measurement in Table 4. This is achieved MMA throughput
as a percentage of hardware peak, not a hardware-counter percentage or the
useful arithmetic count of a conventional FFT.

The 156 TFLOPS denominator is applied only when the measured GPU is identified
as an A100. For another GPU, achieved throughput is still calculated but the
utilization reference remains pending. The 6144 FLOP count is established for
the two recovered candidates only; the missing `radix16` and `radix4_compute`
implementations require their own operation counts when recovered.

`collect_profile.py` remains a standalone optional Nsight Compute diagnostic.
Its counters are not inputs to Table 7 utilization or required collection
artifacts. For diagnostics, it retains profiler CSV/logs, tool version, metric,
unit, kernel identity and exact command. The native `--profile` mode runs two
R=1 correctness launches followed by a dedicated R32 launch; Nsight Compute
skips the correctness launches and profiles the R32 launch. No profiler was
run as part of the CPU-only preparation.

The complete paper table still requires the two missing implementations,
confirmation of original experimental settings and validated measurements.
The printed paper times and speedups are not substituted for fresh results.
