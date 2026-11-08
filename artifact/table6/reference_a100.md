# Table 6 A100 reference timing

256-point FP16 forward FFT, batch 65,536, on A100.

## A100 result (2026-11-06)

| Library | Time (ms) | Speedup | Paper (ms) | Paper speedup |
| --- | ---: | ---: | ---: | ---: |
| cuFFTDx | 0.0229 | 1.00 | 0.023 | 1.00 |
| tcFFT | 0.0218 | 1.05 | 0.021 | 1.11 |
| FlashFFTConv | 0.0181 | 1.27 | 0.018 | 1.28 |
| ThunderFFT | 0.0158 | 1.45 | 0.016 | 1.46 |
