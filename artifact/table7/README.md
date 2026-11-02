# Table 7: Optimization ablation

Owner: unassigned

Status: pending

Paper p. 9, Section 6.4: N=64 unit kernel, batch 65,536. Evaluate these
incremental configurations:

1. Radix-16 baseline following the prior Tensor Core execution pattern.
2. Radix-4 with reduced arithmetic and fused twiddle computation.
3. Add permutation-fused execution to reduce communication.
4. Add branch-uniform register remapping to reduce branch divergence.

Report runtime in milliseconds, Tensor Core utilization, and speedup relative
to the radix-16 baseline. The paper values are 0.0074/0.0120/0.0079/0.0035 ms,
70%/22%/33%/75% utilization, and 1.00/0.62/0.95/2.18 speedup; these are
historical paper targets, not fresh measurements from this branch.

## Starting points and work to assign

- Inspect `include/thunderfft/detail/kernel/unit_kernel_{fp16,fp32}.cuh` and
  `sample/bench/baseline/thunderfft_bench.h` for the current kernel. Standalone
  drivers and the first three historical ablation variants are not present.
- Recover or implement each incremental variant, with a clear source mapping to
  the stated change. Implement a matched driver, profiler collection, raw CSV,
  table generator, and `run.sh`.
- Confirm GPU, precision, input placement, FFT direction, exact baseline,
  utilization definition and metric, profiling tool/version, input/seed,
  timing/repetition method, and statistic. Section 6.4 does not fully specify
  these settings; do not infer them from nearby tables.
- Validate every variant against the same-input FFT reference before timings
  and counters. Use identical workload, normalization, and launch-policy rules;
  record necessary configuration differences.
- Collect timing and utilization under a documented method. Do not substitute
  computed speedup or theoretical utilization for a profiler measurement.

## Results and completion

Write `data.csv` and `table.csv` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,N,batch,precision,placement,variant,repeat,time_ms,utilization_pct,metric,rel_l2,passed`.
Retain raw profiler files, commands, validation evidence, and the speedup
derivation. Completion requires four identifiable validated variants with
matched fresh timing and utilization measurements. See the
[runner contract](../README.md).
