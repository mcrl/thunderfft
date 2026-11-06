# FlashFFTConv — external dependency

FlashFFTConv is not vendored here. The owners of Figure 4 and Table 6 must document an external clone, pin its commit, and record the PyTorch/CUDA build environment before claiming reproduction.

The current legacy Figure 4 driver accepts the external checkout with:

```bash
python3 artifact/figure4/benchmark_figure4_1dfft.py --flash-root /path/to/flashfftconv
```

That driver remains an implementation draft; [Figure 4's task list](../figure4/README.md) describes the correctness and comparison work still needed. `./run.sh figure4` reports pending until it is resolved. New adapters should take their external dependency location from an explicit environment variable such as `FLASHFFTCONV_ROOT` and write all results under `$THUNDERFFT_RESULT_DIR`.

[Table 6](../table6/README.md) clones a pinned commit and builds a patched copy of the N=256 conv kernel; the Python extension is not needed.
