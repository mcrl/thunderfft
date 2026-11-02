# Table 6: Prior Tensor Core FFT comparison

Owner: unassigned

Status: pending

Paper p. 9, Section 6.3: A100; N=256; batch 65,536; cuFFTDx, tcFFT,
FlashFFTConv, and ThunderFFT. Report runtime in milliseconds and speedup
relative to cuFFTDx. This size supports radix-16 implementations. The paper
acknowledges that tcFFT and FlashFFTConv do not expose the same device-level
kernel interface; execution-model differences must remain visible.

## Starting points and work to assign

- Use `sample/bench/` for ThunderFFT/cuFFT helpers and
  `artifact/flashfftconv/README.md` for external baseline notes. The existing
  Figure 4 driver is a convolution benchmark, not a standalone N=256 FFT
  measurement. No integrated Table 6 driver is present.
- Add version-pinned tcFFT/FlashFFTConv/cuFFTDx integration and document any
  wrapper changes or instrumentation needed to expose the FFT operation.
  Implement a common driver, raw exporter, table generator, and `run.sh`.
- Confirm precision, memory placement, direction, input distribution/seed,
  baseline revisions/configurations, timing scope, warmups/repetitions, and
  statistic. The draft's Table 6 does not fully specify them.
- Validate each path with identical logical inputs and an agreed FFT reference
  before timing. Record required layout transformations, normalization, padding,
  host/device boundaries, and fusion differences. Do not compare a full
  convolution call with a bare FFT and call them equivalent.
- Match the agreed timing scope where possible and explain unavoidable
  interface differences alongside the result. Keep raw timings and derivation
  of `cuFFTDx_time / implementation_time`.

## Results and completion

Write `data.csv` and `table.csv` under `THUNDERFFT_RESULT_DIR`. Proposed raw
columns: `gpu,N,batch,library,revision,precision,placement,execution_model,repeat,time_ms,rel_l2,passed`.
Retain baseline build commands/configuration and correctness evidence.
Completion requires four validated measured paths, confirmed semantics and
timing scope, and explicit execution-model limitations. See the
[runner contract](../README.md).
