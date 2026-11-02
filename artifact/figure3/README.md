# Figure 3: End-to-end FFT performance

Owner: unassigned

Status: pending

Paper p. 10, Section 6.5: N=64/256/1024/4096, communication-only baseline,
cuFFT, cuFFTDx, and ThunderFFT. The communication-only kernel performs
vectorized global loads and stores without FFT arithmetic. Plot normalized
performance, with the communication-only result equal to 1. The paper's
evaluation uses batch 65,536; Section 6.5 does not identify this figure's GPU or
precision.

## Starting points and work to assign

- Reuse the end-to-end kernels in
  `sample/bench/baseline/thunderfft_bench.h` and the cuFFT reference in
  `sample/bench/baseline/baseline_cufft.cu`. The sample is not a complete
  four-baseline driver or CSV exporter.
- Implement communication-only and cuFFTDx baselines, then a common driver and
  `run.sh` for smoke/paper profiles.
- Confirm GPU, precision, FFT direction, input distribution, memory layout,
  in-place policy, timing boundary, baseline versions, warmups/repetitions, and
  statistic. The draft does not fully specify these settings.
- Check FFT outputs against the same reference and check the communication-only
  kernel copies all elements exactly before timing. Match bytes transferred and
  input/output types; document whether plans/initialization are excluded.
- Compute normalized performance as `communication_time / implementation_time`,
  with documented matching conditions. Do not force values to the paper's claim
  that implementations exceed 95% of the communication baseline.

## Results and completion

Write `data.csv` and `figure.pdf` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,N,batch,library,precision,repeat,time_ms,bytes,rel_l2,passed`.
Retain raw repetitions and the normalization/plot command. Completion requires
all four validated paths and all four sizes, with explicit timing scope and
fresh measured normalization. See the [runner contract](../README.md).
