# Table 4: MMA and unit-kernel throughput

Table 4 combines three quantities: the paper's reported TF32 hardware peak,
an independently measured ideal MMA rate, and unit-kernel throughput calculated
from **ThunderFFT, N=64, TF32, register-input** Table 3 results. The unit-kernel
row needs only CSV processing; the ideal MMA row runs a separate CUDA benchmark.
A100, H100 and B200 results can be collected separately. Missing cells remain
blank when the table is assembled.

## Unit-kernel calculation

The N=64 TF32 path calls the existing unit kernel in
`include/thunderfft/detail/kernel/unit_kernel_fp32.cuh`. One warp handles 16
FFTs. Each FFT block issues three stages of 16 `mma.m16n8k8` instructions,
each accounting for `2 * 16 * 8 * 8 = 2048` FLOPs. Thus:

```text
issued MMA FLOPs per FFT = 3 * 16 * 2048 / 16 = 6144
unit TFLOPS = 6144 * batch / (median_time_us * 1e6)
```

This counts issued Tensor Core operations, rather than the conventional
`5*N*log2(N)` FFT operation count. The median uses Table 3 `data.csv`'s
`time_us = (t_2R_us - t_R_us) / R`, already normalized by the repeat count.
The separate end-to-end timing is not used.

For the paper's batch of 65,536, its Table 3 times reproduce the Table 4 unit
row after rounding:

| GPU | Paper Table 3 time (us) | Calculated unit TFLOPS |
| --- | ---: | ---: |
| A100 | 3.55 | 113.42 |
| H100 | 2.57 | 156.67 |
| B200 | 1.99 | 202.34 |

These are paper reference values. The adapter calculates measured throughput
from the supplied result CSVs.

## Measure ideal MMA and reuse Table 3

The default `measure` mode builds the standalone CUDA benchmark, measures
ideal MMA throughput on the selected GPU, and combines it with Table 3 when
a source is supplied. It needs CUDA Toolkit, CMake and Python, without external
FFT libraries. Set `THUNDERFFT_ARCHS` for the target architecture and use the
common `CUDACXX`/`CUDAToolkit_ROOT` build settings if needed.

Pass a completed Table 3 result directory containing `data.csv`,
`metadata.json` and `status.json`. Its profile must match the requested profile.
Use the same GPU for both measurements so their efficiency can be calculated.

```bash
THUNDERFFT_TABLE4_SOURCE_DIR=result/table3/<run-id> \
  ./run.sh table4 --profile paper --device 0

# Collect ideal MMA independently; Table 3 can be added later.
./run.sh table4 --profile paper --device 0
```

When Table 3 and Table 4 run in the same batch, the runner passes the Table 3
result automatically. Table 4 itself does not launch Table 3:

```bash
THUNDERFFT_TABLE3_N_LIST=64 THUNDERFFT_TABLE3_PRECISIONS=tf32 \
THUNDERFFT_TABLE3_PLACEMENTS=register \
  ./run.sh table3 table4 --profile paper --device 0
```

The H100 and B200 measurement paths are implemented but have not yet been
validated by running this benchmark on those GPUs.

## Ideal MMA protocol

`ideal_mma.cu` reuses the unit kernel's TF32-input, FP32-accumulator
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32` primitive. It measures
four candidates: 128 or 256 threads per block, with 4 or 8 independent
accumulator chains per warp. For each candidate, the grid contains
`SM count * active blocks per SM * waves` blocks; active blocks are obtained
from CUDA's occupancy API for that compiled kernel. The selected rate is the
highest median trial TFLOPS, with candidate-label order breaking ties.

| Setting | Smoke profile | Paper profile |
| --- | ---: | ---: |
| Outer iterations | 64 | 1,024 |
| Unrolled repetitions per iteration | 32 | 32 |
| Timed trials per candidate | 3 | 5 |
| Warmup launches per candidate | 2 | 5 |
| Grid waves | 2 | 2 |
| Timeout per candidate (seconds) | 120 | 120 |

Override outer iterations, trials, warmups, waves or timeout with
`THUNDERFFT_TABLE4_ITERATIONS`, `THUNDERFFT_TABLE4_TRIALS`,
`THUNDERFFT_TABLE4_WARMUPS`, `THUNDERFFT_TABLE4_WAVES` and
`THUNDERFFT_TABLE4_TIMEOUT`, respectively. Iterations are restricted to
1–8,192; the unroll factor and four candidate configurations are fixed.

```text
warp MMA instructions = blocks * (threads / 32) * chains * iterations * 32
ideal trial TFLOPS = warp MMA instructions * 2048 / (time_us * 1e6)
```

CUDA events time each complete kernel launch, including final output stores.
There is no subtraction of memory or launch overhead. Host copies and output
checks occur outside the timed interval. GPU clocks and power limits are
recorded before and after the measurement; the adapter does not change them.

A and B contain the exactly representable, nonzero value `1/32`. Each MMA
adds exactly `1/128` to each accumulator element, giving an expected final
value of `iterations * 32 / 128`. This stays finite and exact throughout the
supported iteration range. A short numerical check precedes warmup; every
output is checked after warmup and after each timed trial. A nonfinite or
incorrect value fails the benchmark. The output retains source and binary
hashes, build configuration, native CSVs/logs and SASS instruction evidence.

The paper specifies the MMA shape but does not specify these launch dimensions,
chain counts, iteration settings or selection rule. They are explicit artifact
parameters, so matching the paper's ideal rates is not assumed.

## Assemble existing CSVs without GPU work

Set `THUNDERFFT_TABLE4_MODE=derive` to calculate unit throughput without
building or running the ideal MMA benchmark:

```bash
THUNDERFFT_TABLE4_MODE=derive \
THUNDERFFT_TABLE4_SOURCE_DIR=result/table3/<run-id> \
  ./run.sh table4 --profile paper

# Reuse completed unit and ideal measurements; CPU only.
python3 artifact/table4/derive.py --profile paper \
  --source-dir result/table3/<a100-run-id> \
  --ideal-dir result/table4/<a100-ideal-run-id>/measurement \
  --output-dir result/table4/combined
```

Repeat `--source-dir` and `--ideal-dir` for independently collected GPUs.
An ideal source directory must contain `ideal.csv`, `ideal_metadata.json`
and the raw/build/disassembly evidence referenced by that metadata.

For a collection, import existing Table 3 and Table 4 bundles with distinct
part names, then assemble. Ideal-only Table 4 runs can be imported before the
corresponding Table 3 result is available.

```bash
./run.sh --collection ae-table4 --profile paper --part a100-unit \
  --import-result table3=result/table3/<a100-run-id>
./run.sh --collection ae-table4 --profile paper --part a100-ideal \
  --import-result table4=result/table4/<a100-ideal-run-id>
./run.sh table4 --collection ae-table4 --assemble
```

Table 4 reads the Table 3 parts directly, including bundles that also contain
cuFFTDx rows. The example uses a collection dedicated to Table 4: the common
runner refreshes previously assembled evaluations, and Table 3's current
assembler does not yet accept mixed-library bundles. Keeping this assembly
separate avoids invoking that adapter.

An incomplete assembly writes the available data and exits with the runner's
partial-result code (2). Add the remaining parts and assemble again. No GPU
query or workload runs during `--assemble`.

## Outputs and interpretation

- `data.csv`: selected Table 3 trials with calculated unit throughput.
- `ideal.csv`: collected ideal MMA trials for all four candidates.
- `table.csv` and `table.md`: per-GPU summary, with missing measurements blank.
- `metadata.json`: formulas, source hashes, coverage and missing items.
- `inputs/`: source snapshots that make the assembled bundle portable.
- `measurement/`: the original ideal MMA CSV, metadata, native outputs,
  build evidence and `disassembly.sass` from a measurement run.

The hardware peak row uses the supplied paper's reported TF32 chip peaks of
156, 378 and 1,100 TFLOPS, labeled `paper_reference`. These fixed comparison
values do not measure the selected GPU's current clock. The paper's ideal MMA
rates of 147.22, 243.82 and 275.80 TFLOPS are retained in separate reference
columns and never fill missing measurements.

Efficiency is `100 * unit_tflops / ideal_mma_tflops`. Both measurements must
have the same verified GPU UUID; a missing identity leaves efficiency blank,
and conflicting identities are rejected. Historical Table 3 metadata can
establish the UUID from an unambiguous GPU inventory. Matching UUIDs do not
guarantee matching clock state or measurement date; both source records are
retained. Table 7 utilization instead uses hardware peak as its denominator.

Unit throughput inherits Table 3's timing and accuracy scope: correctness is
checked separately with a single FFT, and its repeated FFT timing loop can
grow to nonfinite values. CSV conversion adds no new numerical validation or
finite-loop claim for that unit measurement. The adapter checks source status,
trial coverage, numerical-error gates and differential timing consistency.
The new ideal MMA loop is checked independently as described above.
Smoke-profile throughput is a pipeline check, not a paper-size comparison.
