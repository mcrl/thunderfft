# Figure 4 — 1D convolution

Status: ready subset. Measures ThunderFFT and cuDNN after a correctness gate;
FlashFFTConv is joined from a separately produced CSV. Not yet supported:
offline collection assembly.

Paper p. 10, Section 6.7: A100, `L=4096`, batch `64 x 768`, one full forward
FFT, element-wise multiplication and one full inverse FFT. The plot spans filter
sizes 3…51; the earlier A100 evaluation used odd lengths, so the paper profile
sweeps `K=3,5,…,53`.

## Run

```bash
./run.sh figure4 --profile smoke     # B=4, D=64, K=3,9,53
./run.sh figure4 --profile paper     # B=64, D=768, K=3..53 odd, 3 trials
```

cuDNN is measured when CMake finds it (`thff_conv1d_compare`). Overrides:
`THUNDERFFT_FIGURE4_FILTERS` (comma list), `_TRIALS`, `_VALIDATE_CHANNELS`
(default 1024) and `_BACKENDS` (comma list of backend CSVs).

## Operation (identical for every library)

Depthwise circular convolution `y[b][d][n] = sum_k x[b][d][(n-k) mod L] h[d][k]`
over all `L` outputs, filter not flipped, real FP16 input/filter/output. Inputs
come from `srand(42)` in the same order in both drivers: `x ~ U[0,1]`, then
`h ~ U[0,1]`.

| Library | Native | Notes |
| --- | --- | --- |
| ThunderFFT | `thff_conv1d_fullfft4096 4096 K 0 64 768 5 30 1 1 1 1 1024` | stores `h/L` to absorb the unnormalized inverse; filter FFT included in every timed iteration |
| cuDNN | `thff_conv1d_compare --figure4 4096 K 64 768 5 30 1024` | C++ cuDNN API, groups=D, reversed filter with `CUDNN_CROSS_CORRELATION` (the mode PyTorch `F.conv1d` uses), fastest `cudnnFind` algorithm; the circularly wrapped input (`L+K-1`) is built once before timing |

Both drivers check finite relative L2 ≤ 5e-3 against a direct double-precision
reference on 1025 `(b, d)` sequences spread over `b` and `d` (plus the last one)
before timing; the full reference would take minutes per K. Timing is the mean
of 30 launches after 5 warmups; the median of 3 trials is plotted.

## Backend CSVs (FlashFFTConv)

Columns `gpu,shape,filter,library,repeat,time_ms,rel_l2,passed,operation`, with
`shape=L=4096,B=64,D=768` and `operation` equal to the string in `measure.py`
`FIGURE4["operation"]`. Rows with another operation, GPU, shape, a failed gate
or a library measured here are rejected.

## Reference results (A100 80GB PCIe, CUDA 12.9, cuDNN 9.12, 2026-11-05)

ThunderFFT averages 1.810 ms over K=3…53 (paper: 1.86 ms, −2.7%; earlier
evaluation: 1.834–1.867 ms).

The earlier evaluation measured cuDNN through PyTorch `F.conv1d` (fp16,
`cudnn.benchmark=True`), which runs cuDNN's `conv2d_grouped_direct_kernel` in
cross-correlation mode; rerunning that script here reproduced its values
(K=3/25/53: 2.128/5.401/9.742 ms). With `CUDNN_CONVOLUTION` cuDNN selects a slower
variant of the same kernel (+30% at K=53), so this driver reverses the filter
and uses cross-correlation; it then measures 2.119/5.396/9.648 ms.

The legacy `benchmark_figure4_1dfft.py` (validation disabled) and its historical
`compare_fft4096_..._K3_63_step2.csv` are retained unchanged for provenance and
are not used by `run.sh`.
