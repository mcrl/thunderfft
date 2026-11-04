# Table 2: A100 theoretical performance model

Owner: unassigned

Status: calculated rows plus the measured cuFFTDx row from Table 3 when it runs in the same batch (A100 paper profile reproduces 1.79 / 55.9%)

Paper p. 3, Section 3.2: FP16 A100 model. Rows cover ideal CUDA Core FFT,
measured cuFFTDx, and Tensor Core radix-4/8/16 designs. Columns are engine peak
TFLOPS, arithmetic coefficient of `N log2 N`, MMA shape, utilization ceiling,
and relative time normalized to an ideal CUDA Core FFT. The simplified
load–MMA–store model keeps the twiddle operand in registers and uses shared
memory for data redistribution. Although mostly theoretical, cuFFTDx utilization
55.8% and relative time 1.79 are explicitly marked **measured**.

## Starting points and work to assign

Table 3 is an optional dependency: `./run.sh table2` alone needs no GPU and
produces the calculated rows, while `./run.sh table3 table2` (or the default
batch) runs Table 3 first and passes its completed result directory to
`model.py`. Both profiles produce
`data.csv`, `table.csv`, a readable `table.md` and `metadata.json`. Four rows
are calculated: ideal CUDA Core FFT and Tensor Core radix-4/8/16. The fifth,
measured cuFFTDx row (inserted after the ideal row) is present when that Table 3
run measured cuFFTDx (`MATHDX_ROOT` set); otherwise it is omitted and
`metadata.json` records why. To regenerate Table 2 without a GPU from an
existing Table 3 result:

```sh
python3 artifact/table2/model.py --profile paper \
  --source-dir result/table3/<run> --output /tmp/table2
```

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

## Measured cuFFTDx row

The measured row uses the Table 3 cuFFTDx FP16 register N=64 time at batch
65536, divided by the ideal time `5*N*log2(N)*batch / 78 TFLOPS`. No profiler
is involved. The configuration and paper values are recorded in
`assumptions.json` under `measured_reference`.

- Source rows: A100, cuFFTDx, FP16, register, N=64 from the Table 3 `data.csv`;
  every candidate must pass the single-FFT gate (relL2 <= 5e-3) with the same
  trial set, and the paper profile requires batch 65536 and R=1000.
- Time: the median `compute_us = (t_2R - t_R)/R` of the fastest candidate
  (Table 3's selection rule; cross-checked against `cufftdx_candidates.csv`).
- `relative_time = time / ideal_time`, `utilization = 1 / relative_time`
  (stored in the `utilization_ceiling` column for this row).
- A smoke-profile row (batch 1024) underfills the GPU and is flagged in
  `table.md` as not comparable to the paper.

Measured 2026-11-04 (A100 80GB PCIe, cuFFTDx 1.5.1, paper profile, Table 3 run
`20261104T025105Z-3c295330`): candidate `thread_t32`, 2.888 us vs ideal
1.613 us, relative time 1.79, utilization 55.9% (paper 1.79 / 55.8%; the
paper's 2.89 us rounds to 55.8%).

## Results and completion

`data.csv` and `table.csv` (identical) have the calculated columns plus
`N,batch,measured_time_us,ideal_time_us,candidate` for the measured row.
`metadata.json` keeps the Table 3 source directory and hashes, the candidate's
trial times, the cuFFTDx build and the measured-versus-paper values.
Completion requires reproducible calculated rows and independently traceable
validated measurements, with assumptions and units stated. See the
[runner contract](../README.md).
