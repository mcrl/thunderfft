# Figure5_1dFFT

Figure 5 (1D FFT-conv) reproduction lives here.

## Directory
- `benchmark_figure5_1dfft.py`: benchmark driver (ours + FlashFFTConv + flash_conv1d + cuDNN)
- `run_figure5.sh`: one-shot build + run wrapper
- `compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv`: generated result CSV

## Prerequisites
- CUDA GPU environment
- `thunderfft` build dependencies
- External FlashFFTConv repo (not vendored in this repo)
- Python env with PyTorch

## Build + Run
From repo root:

```bash
bash Figure5_1dFFT/run_figure5.sh /path/to/flashfftconv_repo
```

Default output:

`Figure5_1dFFT/compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv`

## Manual run
```bash
cmake -S thunderfft -B thunderfft/build
cmake --build thunderfft/build --target thff_conv1d_fullfft4096 thff_conv1d_compare -j

python3 Figure5_1dFFT/benchmark_figure5_1dfft.py \
  --flash-root /path/to/flashfftconv_repo \
  --ours-bin thunderfft/build/sample/conv1d/thff_conv1d_fullfft4096 \
  --L 4096 --B 64 --D 768 \
  --k-start 3 --k-end 63 --k-step 2 \
  --csv Figure5_1dFFT/compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv
```
