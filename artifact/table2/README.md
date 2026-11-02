# Table 2: A100 theoretical performance model

Owner: unassigned

Status: pending

Paper p. 3, Section 3.2: FP16 A100 model. Rows cover ideal CUDA Core FFT,
measured cuFFTDx, and Tensor Core radix-4/8/16 designs. Columns are engine peak
TFLOPS, arithmetic coefficient of `N log2 N`, MMA shape, utilization ceiling,
and relative time normalized to an ideal CUDA Core FFT. The simplified
load–MMA–store model keeps the twiddle operand in registers and uses shared
memory for data redistribution. Although mostly theoretical, cuFFTDx utilization
55.8% and relative time 1.79 are explicitly marked **measured**.

## Starting points and work to assign

- Use the paper's derivation as the model specification. There is no standalone
  model generator or measured cuFFTDx utilization driver in this tree.
- Implement a small reproducible model calculation and record the sources and
  assumptions for the paper's 78/312 TFLOPS peaks and 128 B/cycle/SM bandwidth.
  Keep calculated rows distinct from measured rows in raw data.
- Confirm the cuFFTDx measurement's FFT size, batch, launch configuration,
  memory placement, timing scope, and utilization definition/profiler metric.
  The table does not provide enough detail to recreate its two measured cells.
- Integrate the measured driver/profiler and `run.sh`; validate cuFFTDx against
  the same-input FFT reference before collecting time/counters. Record profiler
  version, full metric names, and repeat/aggregation policy.
- Document exactly how measured utilization and normalized time are derived.
  Do not substitute theoretical estimates for the dagger-marked measurements.

## Results and completion

Write `data.csv` and `table.csv` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `engine,design,origin,gpu,N,batch,precision,mma_shape,peak_tflops,load_coefficient,utilization_pct,relative_time`.
Retain model inputs, calculations, raw counter/timing files, and validation.
Completion requires reproducible calculated rows and independently traceable
validated measurements, with assumptions and units stated. See the
[runner contract](../README.md).
