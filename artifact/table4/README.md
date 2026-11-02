# Table 4: MMA and unit-kernel throughput

Owner: unassigned

Status: pending

Paper p. 8, Section 6.2: A100/H100/B200. Compare published chip-level peak
throughput, measured ideal warp-level `mma.sync.m16n8k8` throughput, and achieved
N=64 unit-kernel throughput in TFLOPS. The ideal microbenchmark repeatedly
executes only the target MMA instruction. The paper uses measured instruction
throughput as the relevant reference because newer architectures' published
peaks include other execution paths. Its unit-kernel efficiencies relative to
this reference are 77.0%/64.3%/73.4% for A100/H100/B200.

## Starting points and work to assign

- Inspect `include/thunderfft/detail/kernel/unit_kernel_fp16.cuh` and
  `unit_kernel_fp32.cuh` and the benchmark helpers under `sample/bench/`.
  A dedicated ideal-MMA microbenchmark and Table 4 driver are not present.
- Implement the ideal-MMA and unit-kernel drivers, throughput calculation,
  metadata export, table generation, and `run.sh`. Cite the exact published peak
  source and precision/sparsity assumption; published values are source data,
  not a fresh measurement.
- Confirm exact MMA datatype/accumulation mode, occupancy, independent
  accumulator count, iteration count, unit-kernel batch, FLOP counting rule,
  clock policy, timing scope, warmups/repetitions, and statistic. The draft does
  not fully specify these details.
- Check the microbenchmark's MMA outputs against expected arithmetic and the
  FFT unit outputs against the same-input reference before timing. Prevent
  compiler elimination and invalid/non-finite repeated arithmetic; retain code
  or disassembly evidence that the intended instruction path executes.
- Measure each GPU independently and keep measured instruction throughput and
  theoretical hardware peak distinctly labeled in the table.

## Results and completion

Write `data.csv` and `table.csv` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,kind,precision,mma_shape,N,batch,repeat,time_ms,flop_count,tflops,source`.
Retain raw timing data, arithmetic validation, instruction evidence, and the
efficiency calculation. Completion requires traceable peak specifications plus
validated, matched MMA/unit measurements on all three GPUs. See the
[runner contract](../README.md).
