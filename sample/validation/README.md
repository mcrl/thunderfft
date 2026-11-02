# FFT correctness validation

Build the optional `thff_validate` target with
`-DTHUNDERFFT_BUILD_VALIDATION=ON`, then run:

```sh
./build/sample/validation/thff_validate --device 0 --csv validation.csv
./build/sample/validation/thff_validate --quick
CUDA_VISIBLE_DEVICES="" ./build/sample/validation/thff_validate --host-only
```

This program checks FFT sizes 64, 128, 256, 512, 1024, 2048 and 4096 in both
directions and both FP16 and TF32 modes. It exercises the direct register core
and public global wrapper. The default full-tile batch is 64; `--batch` accepts
a positive multiple of 16. Public-wrapper tests also use the positive unique
batch sizes `1`, `BPB-1` and `BPB+1`, where BPB is the number of FFTs per block.

Inputs include complex Gaussian signals normalized to RMS 1, an offset impulse,
a complex tone, and a constant. The reference is an unnormalized cuFFT FP64
transform of the actual represented FP16 or FP32 input, so FP16 input conversion
is not charged to the FFT error. Both inverse and forward results use this same
normalization convention.

Each case passes only when every output is finite, relative L2 error is at most
`5e-3`, and all applicable memory guards are intact. Direct cores receive the
documented shared workspace followed by 64 sentinel complex values. Both paths
have 16 sentinel complex values after the requested output; input allocations
are exactly the requested size. CUDA Compute Sanitizer can detect invalid input
reads beyond that allocation. Public rows report zero shared-guard changes
because only direct-core calls expose shared workspace to this program.

`--quick` keeps every size, precision, direction and path while reducing inputs
to Gaussian, the full batch to 16, and public extra batches to 1. Both modes
also finalize and reinitialize the twiddle cache before a final N=64 public
forward/inverse pair for each precision.

Before touching CUDA, every run also checks negative batches, null or aliased
buffers, zero-batch no-op handling, and unsupported initialization lengths for
both precisions. `--host-only` stops after these 18 host API checks, so it can
run without an available GPU. It does not accept `--csv`.

CSV is written to stdout and, when requested, to `--csv PATH`. A numerical or
guard failure returns 1; a CLI, CUDA, cuFFT or file error returns 2. This is a
correctness check and performs no performance measurements. Passing these
bounded inputs does not establish accuracy for every input range or confirm
the paper's performance results.

The separate `thff_validate_cache` target checks twiddle cache behavior without
launching FFT or MMA kernels:

```sh
./build/sample/validation/thff_validate_cache
```

It checks same-length cache hits, replacement from N=64 to N=128, clearing of
device symbols on finalization, repeated finalization, reinitialization, and
FP16 unit coefficients against the host builder. It also keeps caches alive
on all supported visible GPUs and verifies pointer ownership and unchanged
per-device pointers after switching away and back. Two supported visible GPUs
are required to test cross-device isolation; a single GPU still exercises the
lifecycle checks. Results are CSV on stdout, with failure returning nonzero.

With CUDA 12 this target contains SM70 and SM80 code and can include a V100
in its cache checks. CUDA 13 builds SM80 code because offline SM70 compilation
is unavailable, and prints a reason when skipping older devices. The separate
translation unit includes only the twiddle implementation and CUDA runtime;
it has no cuFFT or MathDX dependency.
