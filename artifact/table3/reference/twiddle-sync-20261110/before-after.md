# ThunderFFT Table 3 configuration comparison

These are measured before/after results for ThunderFFT alone. They do not replace paper reference values or constitute a full paper Table 3 collection.

Baseline commit: `00bc845d0f3119fa405517cc601126bdbd9b3351`. Times are in **microseconds (us)** for the configured batch, not divided by the number of FFTs. Batch = **65,536**, inside repeats R = **1,000**, **3 timing trials**. Each cell is the median of the three native trials; speedup = before / after.

Compute time is the differential repeated-kernel estimate `(t_2R - t_R) / R`. Register/shared placement follows the Table 3 native benchmark. Single-pass out-of-place end-to-end CUDA-event latency is reported separately: each trial averages 10 single-pass launches, then the report takes the median of three trials. It excludes host initialization, twiddle setup and host/device transfers. Repeated outputs are not accuracy evidence.

Selected changes reuse the existing twiddle lookup tables for FP16 N128/512/1024 and TF32 N128/256, and use warp synchronization in benchmark wrappers for the complete single-warp N64/N512 cases and FP16 N128. TF32 N128 retains block synchronization because the measured compiler output is faster with that synchronization choice. The FFT stages, tensor-core operations and precision categories are unchanged. Public global wrapper barriers remain unchanged. Host lookup coefficients can differ in their last bits from device sin/cos, so bitwise-identical results are not assumed.

Lookup loads and complex multiplication remain in each FFT core at source level, including single-pass kernels. The compiler can hoist invariant coefficient work, as it can for the baseline; this report does not claim each lookup executes on every repeated iteration.

Configurations without a selected operation or synchronization change (FP16 N256, TF32 N1024, and all N2048/N4096 cases) serve as controls. Their observed movements are retained and are not evidence of optimization of those configurations.

Variation uses the observed minimum and maximum of only three trials; it is not a confidence interval. The ±5% latency-change count is a descriptive small-change band, not a statistical significance test. Overlapping trial ranges are reported without declaring a speedup robust. See `before-after.csv` for per-configuration spans and observed ratio bounds.

## A100: NVIDIA A100 80GB PCIe

GPU UUID: `GPU-f4bb2c48-b43c-0f5a-8f59-75d6373ea4e6`; compute capability 8.0; native target SM80. `compiled_CUDART=12090 runtime=12080 driver=12090 cuFFT=11303 extra_R_warmups=3`.

Before and after were built and measured with the same compiler and CUDA/cuFFT versions on this GPU. Different GPUs may use different toolkit versions; their absolute latencies are not a controlled cross-GPU comparison.

### Core compute median

| N | FP16 register before (us) | after (us) | speedup |  | FP16 shared before (us) | after (us) | speedup |  | TF32 register before (us) | after (us) | speedup |  | TF32 shared before (us) | after (us) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 2.2743 | 2.1975 | 1.035x |  | 2.7648 | 2.7034 | 1.023x |  | 3.5953 | 3.6229 | 0.992x |  | 4.7933 | 4.5507 | 1.053x |
| 128 | 7.4342 | 5.9177 | 1.256x |  | 8.4992 | 7.0001 | 1.214x |  | 10.1192 | 8.4746 | 1.194x |  | 13.1891 | 12.5430 | 1.052x |
| 256 | 15.7860 | 15.7911 | 1.000x |  | 22.3621 | 22.3662 | 1.000x |  | 26.9834 | 23.5745 | 1.145x |  | 44.5000 | 43.9235 | 1.013x |
| 512 | 47.8648 | 44.9956 | 1.064x |  | 60.8788 | 52.4155 | 1.161x |  | 64.6799 | 61.6202 | 1.050x |  | 123.6726 | 119.3769 | 1.036x |
| 1024 | 102.2802 | 84.7565 | 1.207x |  | 114.4228 | 100.5691 | 1.138x |  | 125.4513 | 126.3811 | 0.993x |  | 155.0879 | 155.0735 | 1.000x |
| 2048 | 235.9962 | 236.0801 | 1.000x |  | 317.3918 | 317.3673 | 1.000x |  | 305.5165 | 305.6978 | 0.999x |  | 418.6102 | 418.3542 | 1.001x |
| 4096 | 504.8576 | 507.2148 | 0.995x |  | 552.3231 | 552.2934 | 1.000x |  | 734.7947 | 734.7886 | 1.000x |  | 837.0175 | 838.4553 | 0.998x |

### Single-pass out-of-place end-to-end median

| N | FP16 register before (us) | after (us) | speedup |  | FP16 shared before (us) | after (us) | speedup |  | TF32 register before (us) | after (us) | speedup |  | TF32 shared before (us) | after (us) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 20.7872 | 20.6848 | 1.005x |  | 20.7872 | 20.6848 | 1.005x |  | 45.2608 | 45.4656 | 0.995x |  | 45.1584 | 45.4656 | 0.993x |
| 128 | 46.1824 | 45.9776 | 1.004x |  | 46.1824 | 46.1824 | 1.000x |  | 86.9376 | 89.1904 | 0.975x |  | 86.7328 | 88.9856 | 0.975x |
| 256 | 88.2688 | 87.8592 | 1.005x |  | 88.0640 | 87.8592 | 1.002x |  | 171.8272 | 171.9296 | 0.999x |  | 172.1344 | 171.7248 | 1.002x |
| 512 | 169.6768 | 171.9296 | 0.987x |  | 169.8816 | 171.4176 | 0.991x |  | 328.4992 | 328.1920 | 1.001x |  | 328.4992 | 328.4992 | 1.000x |
| 1024 | 325.9392 | 328.4992 | 0.992x |  | 325.7344 | 329.1136 | 0.990x |  | 645.5296 | 645.5296 | 1.000x |  | 646.0416 | 646.3488 | 1.000x |
| 2048 | 746.4960 | 745.9840 | 1.001x |  | 741.1712 | 741.3760 | 1.000x |  | 1296.3839 | 1296.8960 | 1.000x |  | 1296.8960 | 1297.6128 | 0.999x |
| 4096 | 1374.1056 | 1372.1600 | 1.001x |  | 1378.5088 | 1376.0512 | 1.002x |  | 2534.8097 | 2535.0143 | 1.000x |  | 2534.6048 | 2538.7009 | 0.998x |

### Aggregates and observed variation

| Group | Metric | Geomean speedup | Min–max speedup | Small-change cases | Overlapping trial ranges | Largest trial span / median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | compute | 1.055x | 0.992–1.256x | 19/28 | 11/28 | 6.84% |
| all | e2e | 0.997x | 0.975–1.005x | 28/28 | 21/28 | 1.48% |
| fp16 register | compute | 1.075x | 0.995–1.256x | 4/7 | 2/7 | 6.84% |
| fp16 register | e2e | 0.999x | 0.987–1.005x | 7/7 | 4/7 | 0.99% |
| fp16 shared | compute | 1.073x | 1.000–1.214x | 4/7 | 3/7 | 5.06% |
| fp16 shared | e2e | 0.998x | 0.990–1.005x | 7/7 | 5/7 | 1.48% |
| tf32 register | compute | 1.051x | 0.992–1.194x | 5/7 | 3/7 | 3.74% |
| tf32 register | e2e | 0.996x | 0.975–1.001x | 7/7 | 6/7 | 1.36% |
| tf32 shared | compute | 1.022x | 0.998–1.053x | 6/7 | 3/7 | 2.09% |
| tf32 shared | e2e | 0.995x | 0.975–1.002x | 7/7 | 6/7 | 0.91% |

### Slower single-pass medians

The following cases have a higher single-pass median after the change. These are observed regressions in this run; the three-trial ranges and small-change caveat still apply. Core-compute improvements do not imply a faster single-pass kernel for every case.

| N | Precision | Placement | Before (us) | After (us) | Latency change | Trial ranges overlap |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 128 | TF32 | shared | 86.7328 | 88.9856 | +2.60% | no |
| 128 | TF32 | register | 86.9376 | 89.1904 | +2.59% | no |
| 512 | FP16 | register | 169.6768 | 171.9296 | +1.33% | no |
| 1024 | FP16 | shared | 325.7344 | 329.1136 | +1.04% | no |
| 512 | FP16 | shared | 169.8816 | 171.4176 | +0.90% | no |
| 1024 | FP16 | register | 325.9392 | 328.4992 | +0.79% | no |
| 64 | TF32 | shared | 45.1584 | 45.4656 | +0.68% | yes |
| 64 | TF32 | register | 45.2608 | 45.4656 | +0.45% | yes |
| 4096 | TF32 | shared | 2534.6048 | 2538.7009 | +0.16% | yes |
| 256 | TF32 | register | 171.8272 | 171.9296 | +0.06% | yes |
| 2048 | TF32 | shared | 1296.8960 | 1297.6128 | +0.06% | yes |
| 1024 | TF32 | shared | 646.0416 | 646.3488 | +0.05% | yes |
| 2048 | TF32 | register | 1296.3839 | 1296.8960 | +0.04% | yes |
| 2048 | FP16 | shared | 741.1712 | 741.3760 | +0.03% | yes |
| 4096 | TF32 | register | 2534.8097 | 2535.0143 | +0.01% | yes |

### Correctness

Candidate validation: **500 rows passed**, including both directions and direct/public APIs at all seven sizes. The native single-FFT FP64 relative-L2 gates passed for all 56 before/after executions; the limit is 0.005.

| Precision | Before native maximum relative L2 | After native maximum relative L2 | After validation maximum relative L2 |
| --- | ---: | ---: | ---: |
| FP16 | 0.00061000603 | 0.00061000603 | 0.00066231711 |
| TF32 | 0.0026537534 | 0.0026537534 | 0.0026404948 |

Sanitizers: **14 required checks passed**: memcheck and synccheck each covered 88 quick validator cases, and racecheck covered 12 real benchmark wrappers (N64/128/512 × FP16/TF32 × register/shared, R32, one trial, batch32). Every sanitizer recorded an explicit zero-error summary; racecheck also required zero warnings. Instrumented timings are excluded from the performance tables.

For **1 racecheck case(s)**, the native return code remains **1**: `after-racecheck-n64-tf32-register`. Each stopped only at the exact differential-timing check after both R=1 FP64 gates and the repeated-wrapper launches, with zero hazards and header-only CSV output. Their instrumented timing is **unavailable**; no completed benchmark row is claimed. All other validation commands returned zero. Ordinary paired performance measurements still require return code zero and complete timing rows.

Full numerical inputs: Gaussian RMS1, offset impulse, complex tone and constant; both directions and direct/public paths are covered.

Source and binary hashes, input paths and file digests are retained in `report-inputs.json`. The validation maxima and native maxima cover different input/batch/path sets and should not be interpreted as identical experiments.

## H100: NVIDIA H100 PCIe

GPU UUID: `GPU-508ace01-a2b3-bec9-5fe1-855752e6b08c`; compute capability 9.0; native target SM90. `compiled_CUDART=13000 runtime=13000 driver=13030 cuFFT=12000 extra_R_warmups=3`.

Before and after were built and measured with the same compiler and CUDA/cuFFT versions on this GPU. Different GPUs may use different toolkit versions; their absolute latencies are not a controlled cross-GPU comparison.

### Core compute median

| N | FP16 register before (us) | after (us) | speedup |  | FP16 shared before (us) | after (us) | speedup |  | TF32 register before (us) | after (us) | speedup |  | TF32 shared before (us) | after (us) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 1.8011 | 1.7595 | 1.024x |  | 2.3681 | 2.1736 | 1.089x |  | 2.6279 | 2.6391 | 0.996x |  | 4.5852 | 3.4782 | 1.318x |
| 128 | 5.5976 | 4.8563 | 1.153x |  | 6.1625 | 5.4154 | 1.138x |  | 7.3058 | 6.2465 | 1.170x |  | 11.0917 | 10.0659 | 1.102x |
| 256 | 12.9098 | 13.0447 | 0.990x |  | 17.6739 | 16.4187 | 1.076x |  | 21.4647 | 16.7625 | 1.281x |  | 34.8817 | 35.5411 | 0.981x |
| 512 | 37.1033 | 33.4483 | 1.109x |  | 48.3812 | 41.7439 | 1.159x |  | 54.3740 | 51.1499 | 1.063x |  | 103.7462 | 98.5328 | 1.053x |
| 1024 | 80.9306 | 68.0100 | 1.190x |  | 106.7370 | 82.2178 | 1.298x |  | 107.3825 | 107.4080 | 1.000x |  | 135.6490 | 137.7950 | 0.984x |
| 2048 | 177.9423 | 178.3894 | 0.997x |  | 216.3613 | 216.7456 | 0.998x |  | 268.2090 | 269.2811 | 0.996x |  | 332.6552 | 334.9095 | 0.993x |
| 4096 | 394.8543 | 398.3828 | 0.991x |  | 412.4551 | 411.9022 | 1.001x |  | 563.2432 | 558.6803 | 1.008x |  | 643.4295 | 655.7134 | 0.981x |

### Single-pass out-of-place end-to-end median

| N | FP16 register before (us) | after (us) | speedup |  | FP16 shared before (us) | after (us) | speedup |  | TF32 register before (us) | after (us) | speedup |  | TF32 shared before (us) | after (us) | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 14.6144 | 12.0032 | 1.218x |  | 18.3392 | 12.2272 | 1.500x |  | 41.2128 | 40.0832 | 1.028x |  | 39.6928 | 40.0704 | 0.991x |
| 128 | 40.1664 | 39.2352 | 1.024x |  | 40.0992 | 38.8448 | 1.032x |  | 78.6176 | 79.8432 | 0.985x |  | 78.7456 | 80.2656 | 0.981x |
| 256 | 79.3696 | 81.6320 | 0.972x |  | 81.3216 | 80.9824 | 1.004x |  | 163.7504 | 163.7472 | 1.000x |  | 160.9024 | 161.0208 | 0.999x |
| 512 | 153.5840 | 156.8896 | 0.979x |  | 155.1968 | 156.9600 | 0.989x |  | 305.3408 | 302.5856 | 1.009x |  | 303.7120 | 300.8256 | 1.010x |
| 1024 | 301.4880 | 304.2272 | 0.991x |  | 301.0912 | 303.4432 | 0.992x |  | 592.8192 | 593.0112 | 1.000x |  | 592.8448 | 592.7808 | 1.000x |
| 2048 | 610.5536 | 614.0384 | 0.994x |  | 607.2128 | 611.7792 | 0.993x |  | 1214.9312 | 1223.8368 | 0.993x |  | 1226.9857 | 1221.6353 | 1.004x |
| 4096 | 1323.3312 | 1318.4160 | 1.004x |  | 1324.8608 | 1324.9568 | 1.000x |  | 2347.4913 | 2345.6673 | 1.001x |  | 2345.8496 | 2351.1871 | 0.998x |

### Aggregates and observed variation

| Group | Metric | Geomean speedup | Min–max speedup | Small-change cases | Overlapping trial ranges | Largest trial span / median |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| all | compute | 1.072x | 0.981–1.318x | 14/28 | 13/28 | 13.54% |
| all | e2e | 1.021x | 0.972–1.500x | 26/28 | 11/28 | 45.46% |
| fp16 register | compute | 1.062x | 0.990–1.190x | 4/7 | 3/7 | 7.93% |
| fp16 register | e2e | 1.023x | 0.972–1.218x | 6/7 | 2/7 | 45.46% |
| fp16 shared | compute | 1.105x | 0.998–1.298x | 2/7 | 3/7 | 13.54% |
| fp16 shared | e2e | 1.061x | 0.989–1.500x | 6/7 | 3/7 | 8.10% |
| tf32 register | compute | 1.069x | 0.996–1.281x | 4/7 | 4/7 | 7.88% |
| tf32 register | e2e | 1.002x | 0.985–1.028x | 7/7 | 3/7 | 1.67% |
| tf32 shared | compute | 1.053x | 0.981–1.318x | 4/7 | 3/7 | 5.52% |
| tf32 shared | e2e | 0.997x | 0.981–1.010x | 7/7 | 3/7 | 1.77% |

### Slower single-pass medians

The following cases have a higher single-pass median after the change. These are observed regressions in this run; the three-trial ranges and small-change caveat still apply. Core-compute improvements do not imply a faster single-pass kernel for every case.

| N | Precision | Placement | Before (us) | After (us) | Latency change | Trial ranges overlap |
| ---: | --- | --- | ---: | ---: | ---: | --- |
| 256 | FP16 | register | 79.3696 | 81.6320 | +2.85% | no |
| 512 | FP16 | register | 153.5840 | 156.8896 | +2.15% | no |
| 128 | TF32 | shared | 78.7456 | 80.2656 | +1.93% | no |
| 128 | TF32 | register | 78.6176 | 79.8432 | +1.56% | no |
| 512 | FP16 | shared | 155.1968 | 156.9600 | +1.14% | yes |
| 64 | TF32 | shared | 39.6928 | 40.0704 | +0.95% | no |
| 1024 | FP16 | register | 301.4880 | 304.2272 | +0.91% | no |
| 1024 | FP16 | shared | 301.0912 | 303.4432 | +0.78% | no |
| 2048 | FP16 | shared | 607.2128 | 611.7792 | +0.75% | yes |
| 2048 | TF32 | register | 1214.9312 | 1223.8368 | +0.73% | yes |
| 2048 | FP16 | register | 610.5536 | 614.0384 | +0.57% | yes |
| 4096 | TF32 | shared | 2345.8496 | 2351.1871 | +0.23% | yes |
| 256 | TF32 | shared | 160.9024 | 161.0208 | +0.07% | yes |
| 1024 | TF32 | register | 592.8192 | 593.0112 | +0.03% | yes |
| 4096 | FP16 | shared | 1324.8608 | 1324.9568 | +0.01% | yes |

### Correctness

Candidate validation: **500 rows passed**, including both directions and direct/public APIs at all seven sizes. The native single-FFT FP64 relative-L2 gates passed for all 56 before/after executions; the limit is 0.005.

| Precision | Before native maximum relative L2 | After native maximum relative L2 | After validation maximum relative L2 |
| --- | ---: | ---: | ---: |
| FP16 | 0.00061001645 | 0.00061001645 | 0.00066231711 |
| TF32 | 0.002663749 | 0.002663749 | 0.0026454357 |

Sanitizers: **14 required checks passed**: memcheck and synccheck each covered 88 quick validator cases, and racecheck covered 12 real benchmark wrappers (N64/128/512 × FP16/TF32 × register/shared, R32, one trial, batch32). Every sanitizer recorded an explicit zero-error summary; racecheck also required zero warnings. Instrumented timings are excluded from the performance tables.

For **2 racecheck case(s)**, the native return code remains **1**: `after-racecheck-n128-tf32-register`, `after-racecheck-n512-fp16-shared`. Each stopped only at the exact differential-timing check after both R=1 FP64 gates and the repeated-wrapper launches, with zero hazards and header-only CSV output. Their instrumented timing is **unavailable**; no completed benchmark row is claimed. All other validation commands returned zero. Ordinary paired performance measurements still require return code zero and complete timing rows.

Full numerical inputs: Gaussian RMS1, offset impulse, complex tone and constant; both directions and direct/public paths are covered.

Source and binary hashes, input paths and file digests are retained in `report-inputs.json`. The validation maxima and native maxima cover different input/batch/path sets and should not be interpreted as identical experiments.
