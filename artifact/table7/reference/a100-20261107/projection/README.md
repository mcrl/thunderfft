# Derived baseline and measured unit-kernel comparison

A100, batch 65,536. The first two rows project the rounded Table 6 N=256 FP16
resident-compute timings onto a theoretical N=64 TF32 work model, assuming
constant utilization of the respective hardware peaks (312 and 156 TFLOPS).
They are estimates; no N=64 radix-16 kernel was executed. The other four rows
are archived N=64 TF32 timings, one trial each, with accuracy unchecked.

| Source | Evidence | Time (us) | MMA utilization | Ratio to projected FlashFFTConv baseline |
| --- | --- | ---: | ---: | ---: |
| tcFFT | Estimated from rounded Table 6 timing | 8.17500 | 63.15% | 0.830 |
| FlashFFTConv | Estimated from rounded Table 6 timing | 6.78750 | 76.05% | 1.000 |
| 51e346c | Measured unit kernel; unchecked | 11.93677 | 21.62% | 0.569 |
| 5e37821 | Measured unit kernel; unchecked | 3.66797 | 70.37% | 1.850 |
| e4aa1b0 | Measured unit kernel; unchecked | 7.88890 | 32.72% | 0.860 |
| 652808b | Measured unit kernel; unchecked | 3.48160 | 74.14% | 1.950 |

The ratio divides the estimated FlashFFTConv baseline time by each row's time;
it is not a measured speedup between two complete N=64 FFT implementations.
`e4aa1b0` omits operand initialization, and `652808b` also omits stage
permutation. Source IDs are retained without asserting the paper's stage
identities. [comparison.csv](comparison.csv) retains row-level source hashes,
operation coverage and accuracy status; [baseline_estimates.csv](baseline_estimates.csv)
and [projection_metadata.json](projection_metadata.json) contain the formula
and input evidence. Original raw files are in the [parent bundle](../README.md).

Regenerate these derived files from the repository root into a new directory:

```sh
python3 artifact/table7/project_baseline.py --documented-table6 \
  --kernel-comparison artifact/table7/reference/a100-20261107/kernel_comparison.csv \
  --output result/table7/normalized-a100-20261107
```
