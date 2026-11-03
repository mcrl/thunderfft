# Table 2: A100 theoretical performance model

Owner: unassigned

Status: runnable calculated subset; the measured cuFFTDx row remains excluded

Paper p. 3, Section 3.2: FP16 A100 model. Rows cover ideal CUDA Core FFT,
measured cuFFTDx, and Tensor Core radix-4/8/16 designs. Columns are engine peak
TFLOPS, arithmetic coefficient of `N log2 N`, MMA shape, utilization ceiling,
and relative time normalized to an ideal CUDA Core FFT. The simplified
load–MMA–store model keeps the twiddle operand in registers and uses shared
memory for data redistribution. Although mostly theoretical, cuFFTDx utilization
55.8% and relative time 1.79 are explicitly marked **measured**.

## Starting points and work to assign

Run `./run.sh table2` for the CPU-only calculated model. Both profiles produce
`data.csv`, `table.csv`, a readable `table.md` and `metadata.json`. No CUDA,
cuFFTDx parameters or external FFT library is needed. Four rows are generated:
ideal CUDA Core FFT and Tensor Core radix-4/8/16. The measured cuFFTDx row is
omitted, so successful execution covers the declared calculated subset.

[`assumptions.json`](assumptions.json) records the supplied paper's 78/312
TFLOPS peaks and 128 B/cycle/SM shared bandwidth as model inputs. The 2048
operations/SM/cycle assumption comes from the paper's one `m16n8k8` MMA/cycle
anchor; it is not freshly measured instruction throughput. For radix `r`,
`m=16,n=k=2r`, MMA work is `2*m*n*k`, and shared traffic is
`(m*k+m*n)*2` bytes because operand B stays in registers. Dividing work by
2048 gives issue cycles; traffic per cycle gives the bandwidth requirement.
Utilization is bounded by available/required shared bandwidth and capped at 1.

Arithmetic is `8*r/log2(r) * N*log2(N)`. Relative time is
`(coefficient/(peak*utilization))/(5/78)`. The generated values are
`1.00,3.20,2.1333,1.60`, matching the rounded paper values. Radix-8 retains
the exact `64/3` coefficient internally. Metadata retains the assumptions,
formulae, source PDF hash and generator hash; raw rows are labeled calculated.
Run `python3 artifact/table2/test_model.py` for CPU formula/unit checks.

The remaining measured row is assigned separately:

- Confirm the cuFFTDx measurement's FFT size, batch, launch configuration,
  memory placement, timing scope, and utilization definition/profiler metric.
  The table does not provide enough detail to recreate its two measured cells.
- Integrate the measured driver/profiler; validate cuFFTDx against
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
