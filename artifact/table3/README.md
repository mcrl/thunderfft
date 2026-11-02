# Table 3: FFT kernel latency matrix

Owner: unassigned

Status: pending

Paper p. 8, Section 6.2: single A100 PCIe 80GB, H100 PCIe, and B200; FFT
sizes 64/128/256/512/1024/2048/4096; batch 65,536. Measure register-input and
shared-memory-input kernels separately. Compare matched FP16 paths and
ThunderFFT TF32 against cuFFTDx FP32. Report latency in microseconds,
`cuFFTDx_time / ThunderFFT_time` speedups, and geometric-mean speedups over
sizes. These are 84 matched comparison configurations across GPUs, precisions,
placements, and sizes. The paper selects the best cuFFTDx launch parameters
after an exhaustive sweep. [Figure 2](../figure2/README.md) consumes this data's
A100/register subset.

## Starting points and work to assign

- Reuse `sample/bench/main.cu` and
  `sample/bench/baseline/thunderfft_bench.h`. Existing ThunderFFT helpers expose
  register/shared paths, but the sample activates only N=128 and does not
  implement the cuFFTDx sweep or this matrix's raw CSV contract.
- Implement the cuFFTDx device-level paths, enumerated legal launch-parameter
  sweep, common timing driver, CSV exporter, table generator, and `run.sh`.
  Retain every candidate and the winner-selection rule rather than only minima.
- Confirm cuFFTDx/MathDx version, exact swept parameter ranges, FFT direction,
  input distribution/seed, repetition strategy and normalization, timer,
  warmups/repetitions, and reported statistic. The draft leaves these incomplete.
- Validate outputs for every reported implementation/configuration using identical
  input data and the agreed reference/tolerances before timing. Repeated FFTs
  used to amortize global access must not silently overflow or change the timing
  workload; document the method and correctness check separately.
- Match timing boundaries and layouts; distinguish register-resident and
  shared-input paths. Record GPU/software metadata for each independently run
  GPU; do not label one GPU's measurements as a three-GPU reproduction.

## Results and completion

Write `data.csv` and `table.csv` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,N,batch,placement,library,precision,configuration,repeat,time_us,rel_l2,passed,selected`.
Keep all tuning candidates, raw timings, validation reports, and derived
speedups/geometric means. Completion requires the complete confirmed matrix
across all three GPUs, correctness evidence, and an auditable sweep/selection
process. Partial GPU runs must identify missing configurations. See the
[runner contract](../README.md).
