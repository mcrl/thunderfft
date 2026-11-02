# Figure 4 — 1D convolution

**Owner:** unassigned. **Status:** native correctness gate implemented; paper-workload validation and equivalent baseline semantics remain pending.

The supplied PDF's Figure 4 compares 1D convolution at `L=4096`, `B=64`, and `D=768`. The existing implementation is depthwise. The plot suggests small-to-large filters around `K=3…53`, but the text does not specify every sampled filter length or fully define the operation; confirm those settings with the author. The retained driver uses a provisional odd-length `3…53` sweep. This directory was previously named `figure5`; its number now follows the supplied paper.

## Run contract

From the repository root:

```bash
./run.sh figure4
```

`run.sh` currently returns **78 (pending)** with a concrete reason. It creates no measured performance claims. The legacy driver is retained only as an implementation starting point:

- `benchmark_figure4_1dfft.py`: ThunderFFT, FlashFFTConv, flash_conv1d, and cuDNN timing driver.
- `compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv`: unchanged historical data. Its `K=3…63` sweep does not match the paper; these bytes are preserved for provenance and are never treated as fresh results.

The legacy driver defaults to `result/figure4/legacy_timings.csv`, or `$THUNDERFFT_RESULT_DIR/legacy_timings.csv` when set. New evaluation outputs belong in the runner-provided `$THUNDERFFT_RESULT_DIR`, never in this source directory.

## Work to assign

The 4096-point filter-frequency transpose has been corrected. Targeted A100 / CUDA 12.9 real-input checks with seed 42 reduced the `N=4096, K=9, B=D=1` output/`N` circular-convolution relative L2 error from `0.281035` to approximately `0.000964`, with the same FP16 arithmetic and an independent double direct reference. See the [sample validation notes](../../sample/README.md#full-length-1d-diagnostics-thff_conv1d_fullfft4096) for the checked filter lengths and remaining limits. This does not validate the paper workload or make the legacy driver correctness-gated.

1. Keep the native numerical gate enabled and validate the corrected 4096 FFT-convolution path on the paper workload before collecting comparison timings. With `validate=1`, the native diagnostic now requires finite output/`N` circular-convolution relative L2 error at most `5e-3` and returns nonzero on failure. The 2026-11-02 Vast.ai A100 and H100 checks each passed eight small convolution cases, including `N=4096, K=1,3,9,17,53, B=D=1` and `K=9, B=2, D=3`. These checks do not cover the paper's `B=64, D=768` workload; the legacy adapter still disables native validation.
2. Define the paper's operation precisely: linear versus circular convolution, padding/cropping, normalization, input/filter dtype, and filter reversal. The legacy cuDNN path currently performs cross-correlation with left circular padding, and the legacy paths do not compare shared outputs.
3. Generate identical seeded inputs and filters for every implementation. Require a numerical pass/fail gate with documented tolerances for each dtype before timing each setting. Export actual output tensors or use a common reference so the native executable can participate.
4. Pin CUDA, PyTorch, cuDNN, and external FlashFFTConv revisions; document installation. FlashFFTConv remains external (`--flash-root /path/to/flashfftconv`). Record setup and hardware in the output metadata.
5. Confirm the exact paper filter samples, then implement bounded `smoke` and paper profiles in `run.sh`. The retained driver's provisional `K=3…53`, step 2 setting is not proof of the original sampling. Use the shared runner environment for build, device, Python, and output directories.
6. Save `correctness.csv`, `timings.csv`, exact commands, environment metadata, and a plotting script/figure under `$THUNDERFFT_RESULT_DIR`. Record warmups, repetitions, whether filter FFT and transfers are included, and summary statistics.

Completion requires fresh correctness evidence for every compared path, matched timing boundaries, the paper's exact sweep, and a reproducible Figure 4 plot. A successful legacy subprocess or old CSV is insufficient.

## Legacy driver inspection

After the owner resolves the issues above, the existing CLI can be used to develop the adapter:

```bash
python3 artifact/figure4/benchmark_figure4_1dfft.py --flash-root /path/to/flashfftconv --ours-bin build/sample/conv1d/thff_conv1d_fullfft4096 --L 4096 --B 64 --D 768 --k-start 3 --k-end 53 --k-step 2 --csv result/figure4/legacy_timings.csv
```

This command still skips native correctness checks and is **not** an AE reproduction command. The runner intentionally does not invoke it.
