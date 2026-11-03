# Table 7: optimization ablation

The measurement sources are self-contained in this folder:

```text
ablation/benchmark.cu                 correctness and timing driver
ablation/kernels/recovered.cuh        two historical TF32 implementations
ablation/kernels/provenance.json      source commits, hashes and adaptations
variants.json                        available and missing implementations
build.sh                             compile without launching a GPU workload
measure.py                           selection, execution and table export
collect_profile.py                   optional Nsight Compute collection
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
GPU, precision, repetitions or exact utilization metric. The recovered code
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

Timing and utilization can be collected in different executions. The following
collection commands launch GPU work; the final assembly command reads files
only. Keep the same compiled binary, source files, GPU and protocol settings for
all parts of one table.

```sh
THUNDERFFT_TABLE7_VARIANTS=register_branch THUNDERFFT_TABLE7_PHASE=timing \
  ./run.sh table7 --collection a100-ablation --profile paper --part branch-timing
THUNDERFFT_TABLE7_VARIANTS=register_uniform THUNDERFFT_TABLE7_PHASE=timing \
  ./run.sh table7 --collection a100-ablation --profile paper --part uniform-timing

# This can be collected later, after Nsight Compute is available.
THUNDERFFT_TABLE7_PHASE=profile \
  ./run.sh table7 --collection a100-ablation --profile paper --part utilization

./run.sh table7 --collection a100-ablation --profile paper --assemble
```

`THUNDERFFT_TABLE7_PHASE=timing` collects ordinary timing trials; `profile`
collects only the utilization counter and its independent correctness checks;
`both` collects both kinds. The default is `timing`; the existing
`THUNDERFFT_TABLE7_NCU=1` option selects `both` unless an explicit phase is set.
Within a collection, `THUNDERFFT_TABLE7_VARIANTS` may be omitted to collect the
two available implementations.

Collected parts are retained under
`result/collections/a100-ablation/parts/table7/`. Assembly writes
`result/collections/a100-ablation/assembled/table7/table.csv`, the combined
timing `data.csv`, and metadata listing collected and missing measurements.
Run assembly at any point: missing timing or utilization cells remain blank,
and the result stays `partial` until all required cells are present. The two
unrecovered implementations remain missing even after the available candidates
have been measured. Paper identity remains unconfirmed.

The assembler checks the numerical gates, raw CSVs, profiler logs and hashes.
It rejects mixed smoke/paper protocols, settings, source/binary hashes, GPU
identities, runtime/compiler versions, or utilization counters. Identical
duplicates are harmless; overlapping measurements with different values require
choosing one result, rather than silently overwriting or averaging them.
Parts created before native GPU UUID logging was added cannot be combined
through this strict assembly contract.

For direct offline use with existing successful result directories:

```sh
python3 artifact/table7/assemble.py path/to/branch-part path/to/uniform-part \
  --profile paper --output result/table7-assembled
```

## Optional utilization counter

Set `THUNDERFFT_TABLE7_NCU=1` with the explicit subset command to collect
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active` separately from
ordinary timing. Set `THUNDERFFT_NCU` to the executable path when needed.
The native `--profile` mode performs its two R=1 numerical gates, then exactly
one R32 launch. Nsight Compute skips the two correctness launches and profiles
that third matching kernel. Clocks and cache controls are left unchanged.

Profiler CSV/logs, tool version, metric name, unit, kernel identity and exact
command are retained. The measured percentage is stored with its metric name;
it is not claimed to be the same counter used for the paper's utilization.
`THUNDERFFT_TABLE7_METRIC` may select another percentage metric explicitly.
Without profiling, utilization remains blank. No profiler was run as part of
the CPU-only preparation.

The complete paper table still requires the two missing implementations,
confirmation of original experimental settings and validated measurements.
The printed paper times and speedups are not substituted for fresh results.
