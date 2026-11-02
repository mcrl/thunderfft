# Figure 5: 1D FFT convolution

Figure 5 (1D FFT-conv) reproduction lives here.

## Directory

- `benchmark_figure5_1dfft.py`: benchmark driver (ours + FlashFFTConv + flash_conv1d + cuDNN)
- `run_figure5.sh`: one-shot build + run wrapper
- `*.csv`: legacy local benchmark outputs copied during repository migration; they have not been revalidated against the migrated source.

## Prerequisites

- CUDA GPU environment
- `thunderfft` build dependencies
- External FlashFFTConv repo (not vendored in this repo)
- Python env with PyTorch

## Build + Run

From repo root:

```bash
bash artifact/figure5/run_figure5.sh /path/to/flashfftconv_repo
```

Default output:

`artifact/figure5/compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv`

Set `PYTHON=/path/to/python` to select a Python environment; otherwise the wrapper uses `python3`. Use `--csv /path/to/new.csv` to retain the existing output when rerunning.

## Manual run

```bash
cmake -S . -B build
cmake --build build --target thff_conv1d_fullfft4096 -j

python3 artifact/figure5/benchmark_figure5_1dfft.py \
  --flash-root /path/to/flashfftconv_repo \
  --ours-bin build/sample/conv1d/thff_conv1d_fullfft4096 \
  --L 4096 --B 64 --D 768 \
  --k-start 3 --k-end 63 --k-step 2 \
  --csv artifact/figure5/compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv
```

The driver uses the current binary's positional arguments (`N K device B D warmup runs include_filter_fft validate real_input real_filter`) and parses its timing output. The legacy `--ours-filter-fft-mode` option accepts only `1`; the current binary uses cuFFT for the filter FFT. This timing sweep disables the binary's correctness check and does not compare outputs across implementations.
