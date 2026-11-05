# Samples

Start with `thff_fft` for a small, checked example of the host launch API, or
`thff_validate` for a correctness matrix. The convolution examples demonstrate
fusion inside a CUDA kernel; their validation scopes differ as described below.
A successful build or process exit does not establish correctness for every
FFT size, precision, direction, or layout.
See the root [README](../README.md) for prerequisites and the current validation
scope.

## Build

Run these commands from the repository root. Select the architecture for your
GPU; the default is SM 80.

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80
cmake --build build --target thff_fft thff_conv1d thff_conv1d_fullfft4096 benchmark -j
```

These targets require CUDA and cuFFT. Building `thff_conv` additionally requires
MathDX/CuTe; the optional cuDNN comparison has its own dependency described below.
Use explicit targets when those optional dependencies are unavailable.

| Target | Source | Purpose |
| --- | --- | --- |
| `thff_fft` | [fft1d/main.cu](fft1d/main.cu) | Forward FFT through the host launch API, checked against cuFFT |
| `thff_validate` | [validation/fft_validation.cu](validation/fft_validation.cu) | Optional FP16/TF32 forward/inverse correctness suite; configure with `THUNDERFFT_BUILD_VALIDATION=ON` |
| `thff_validate_cache` | [validation/cache_validation.cu](validation/cache_validation.cu) | Optional twiddle-cache lifecycle checks, with the same validation build option |
| `benchmark` | [bench/main.cu](bench/main.cu) | Compile-time FFT benchmark with register and shared-memory variants |
| `thff_paper_fft` | [bench/paper_fft.cu](bench/paper_fft.cu) | Configurable, correctness-gated Table 3 forward compute timing; separate end-to-end timing |
| `thff_conv1d` | [conv1d/main.cu](conv1d/main.cu) | Tiled, valid 1D operation using 64-point FFTs |
| `thff_conv1d_fullfft4096` | [conv1d/conv1d_fullfft_l4096_thunder.cu](conv1d/conv1d_fullfft_l4096_thunder.cu) | Full-length FFT convolution with an optional correctness gate |
| `thff_conv1d_compare` | [conv1d/conv1d_compare_cudnn_thunder.cu](conv1d/conv1d_compare_cudnn_thunder.cu) | Optional timing driver for CUDA, cuDNN, and ThunderFFT paths |
| `thff_conv` | [conv2d/main.cu](conv2d/main.cu) | Correctness-gated 2D convolution (cuDNN, cuFFT, tiled cuFFTDx) for Figure 5 |

## Basic FFT: thff_fft

```text
thff_fft [batch=16] [device=0]
```

```bash
./build/sample/fft1d/thff_fft 16 0
```

The example fixes `N=64`, `T=float`, and `forward=true` at compile time. Input and
output are separate, contiguous `float2[batch][64]` arrays; `.x` is the real part
and `.y` the imaginary part. Storage is FP32; ThunderFFT uses TF32 Tensor Core
arithmetic internally. It generates deterministic, bounded complex tones, calls
`ThunderFFTInitialize<float>(64)`, launches `ThunderFFT_global<float, 64, true>`,
and releases the internal tables with `ThunderFFTFinalize<float>()` after use.
The forward FFT is unnormalized and is compared with `CUFFT_FORWARD` in natural
frequency order.

`batch` must be positive and divisible by 16, the number of transforms processed
by one block for this size. The example rejects other values before launching;
the underlying host API also supports partial batches and a zero-batch no-op.
Choose the CUDA device with the second argument. Changing `N`, the type, or the
direction requires editing and rebuilding the source and validating that path.

The output reports the configuration, `Relative L2`, `Relative Linf`, and
`Tolerance` in scientific notation, followed by `PASS` or `FAIL`. Passing requires
finite outputs and both relative errors at most `5e-3`; invalid arguments, CUDA
errors, nonfinite values, and failed comparisons return a nonzero exit code.
This threshold is a smoke check for this deterministic input, rather than an
accuracy guarantee for other configurations. This example does not time the FFT.

An observed run on an NVIDIA A100 with a CUDA 12.9 build produced the following
for batch 16. Error values may vary; the tolerance applies to this example input.

```text
ThunderFFT vs cuFFT: float2, N=64, forward (unscaled), batch=16, device=0
Relative L2:   1.328438e-03
Relative Linf: 1.583866e-03
Tolerance:     5.000000e-03 (this example input)
PASS
```

## Correctness matrix: `thff_validate`

Enable the optional suite and build its target explicitly:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80 -DTHUNDERFFT_BUILD_VALIDATION=ON
cmake --build build --target thff_validate -j2
./build/sample/validation/thff_validate --quick
./build/sample/validation/thff_validate --device 0 --csv validation.csv
CUDA_VISIBLE_DEVICES="" ./build/sample/validation/thff_validate --host-only
```

Without `--quick`, the program runs its full matrix; there is no `--full` option.
Both GPU modes cover all seven sizes, FP16 and TF32, forward and inverse,
register cores and the public wrapper. The full matrix adds input patterns and
additional partial-batch cases. References use cuFFT FP64 on the represented
input values. Finite outputs, relative L2 error at most `5e-3`, and intact memory
guards are required. `--host-only` checks argument handling without a GPU and
does not accept `--csv`.

See the [validation guide](validation/README.md) for the exact workload and exit
codes. The remote A100 run passed all 500 full-matrix cases. Maximum observed
relative L2 errors were `6.6232e-4` for FP16 and `2.6405e-3` for TF32, within the
`5e-3` threshold. Compute Sanitizer memcheck, racecheck and synccheck each ran
88 quick-matrix cases with no reported errors. The H100 run passed the same
checks, with maximum FP16 `6.6232e-4` and TF32 `2.6454e-3` relative L2 error.
This suite measures correctness, not performance, and its bounded
inputs do not establish error limits for every input range. Native CUDA 13/SM 100
binaries were built for B200; that GPU remains unmeasured because no suitable
single-GPU rental was available.

## Current limitations

The library implements the following launch tuples for both `float` and `half`
and both transform directions. This table describes implementation coverage;
the validation guide defines the checks available for those paths:

| `N` | FFTs per block (`BPB`) | Warps per block (`WPB`) | Implementation notes |
| --- | --- | --- | --- |
| 64 | 16 | 1 | Used by `thff_fft` in float forward mode |
| 128 | 8 | 1 | Size-specific register kernels and fragment mappings |
| 256 | 16 | 4 | Size-specific register kernels and fragment mappings |
| 512 | 2 | 1 | Forward and inverse register kernels and fragment mappings |
| 1024 | 1 | 1 | Size-specific register kernels and fragment mappings |
| 2048 | 1 | 2 | Forward and inverse register kernels and fragment mappings |
| 4096 | 1 | 4 | Size-specific register kernels and fragment mappings |

Use these additional contracts when calling the APIs directly:

- `ThunderFFT_global` accepts any nonnegative global batch. A positive batch
  requires separate, non-overlapping input/output buffers of `batch*N` complex
  values. The final partial block zero-fills inactive transforms and masks their
  output writes. Zero batch returns before CUDA or pointer access. Negative
  batch, null pointers for a positive batch, and identical pointers throw
  `std::invalid_argument`; other input/output overlap is also prohibited.
- The global wrapper prepares the size- and direction-specific per-thread
  FP16 coefficients. Fused kernels calling the register API directly remain
  responsible for their own coefficient setup.
- Forward and inverse kernels are unnormalized. Apply `1/N` after inverse, or
  incorporate that scale into the operation being fused.
- The global wrapper synchronizes the supplied stream before returning,
  including earlier work in that stream. Attribute setup, launch and completion
  failures throw `std::runtime_error`. It is a synchronous host interface.
- Call `ThunderFFTInitialize<T>(N)` on the current device before use. There is
  one cached size per component type and CUDA device, shared by callers on that
  device; another size replaces that allocation. Unsupported sizes throw
  `std::invalid_argument` before CUDA access. Host initialization/finalization
  calls are serialized per type, but execution is not protected by that lock:
  finish all dependent kernels before replacement or `ThunderFFTFinalize<T>()`.
  Finalization affects the current device, clears its pointer/size symbols and
  initialization state, and checks CUDA errors. Reinitialize before reuse; other
  devices' caches remain intact. CUDA failures throw `std::runtime_error`.
  Resetting the CUDA device or replacing its context during a cache lifetime is
  unsupported.
- On SM80, the float N=1024 register kernel reads the global twiddle table
  prepared by `ThunderFFTInitialize<float>(1024)` in both directions. This also
  applies to fused callers; their per-thread `W` coefficients do not replace
  that initialization. Keep the matching cache alive until the kernel finishes.
- `preprocess_W`, `ThunderFFT_kernel_smem`, and `ThunderFFT_kernel_smem_pad` are
  declaration-only APIs in this release. Use `ThunderFFTInitialize/Finalize`,
  the implemented movement helpers, and `ThunderFFT_kernel_reg` as shown in the
  examples.

Convolution validation and cross-implementation equivalence have further limits
described with each executable below.

## FFT benchmark: `benchmark`

```bash
./build/sample/bench/benchmark
```

This executable currently has no command-line arguments. `main()` selects a
1024-point forward FFT, batch 65536, and FP32 complex storage by default. ThunderFFT uses
TF32 Tensor Core arithmetic; the printed `fp32` label describes storage rather
than full FP32 arithmetic throughout. The input is a real sinusoid stored
in contiguous complex `float2` arrays with zero imaginary components. The
`THUNDERFFT_PROFILE_N` compile-time macro selects the length;
`THUNDERFFT_PROFILE_BATCH`, `THUNDERFFT_PROFILE_FLOAT`, and
`THUNDERFFT_PROFILE_HALF` select the batch and enabled precisions.

The `FFT benchmark results` table contains `cufft`, `th_r(fp32)` (register), and
`th_s(fp32)` (shared-memory) rows. Interpret its columns as follows:

| Column | Meaning |
| --- | --- |
| `N`, `B` | FFT length and batch count |
| `max_err` | Maximum complex absolute error against the cuFFT output; the cuFFT row is zero by construction |
| `compute` | Estimate from the difference between two timings with different repeat counts inside a kernel; the cuFFT row uses a zero placeholder |
| `e2e` | GPU time for one FFT pass including its device data movement; excludes host transfers, allocation, and planning |

The benchmark prints error values without a pass/fail threshold. Inspect them
before interpreting timings. Repeated in-place transforms used for timing have
different numerical behavior from the separate single-pass validation. This
fixed workload is not a supported-configuration test suite or a complete AE
benchmark sweep.

## Configurable forward timing: `thff_paper_fft`

```bash
cmake --build build --target thff_paper_fft -j2
./build/sample/bench/thff_paper_fft 1024 fp16 register 1000 3 65536
```

The CLI is `N fp16|tf32 register|shared [R=1000] [trials=3] [batch=65536]`.
It accepts the seven supported lengths and a positive batch divisible by that
length's FFTs per block. Before timing it checks a separate single forward FFT
against cuFFT FP64 on deterministic complex input, with a relative L2 limit of
`5e-3`. CSV goes to stdout; GPU and CUDA/cuFFT versions go to stderr.

Table 3 compute latency is `(t_2R_us-t_R_us)/R`, using CUDA events after warmup.
The separate `e2e_out_of_place_us` column includes global input/output traffic
for a single FFT. Timed repeated-FFT outputs may overflow and are not used as
the correctness reference. This target measures ThunderFFT; it does not run
the paper's cuFFTDx parameter sweep or establish its speedup claims.

Configuring with `-DMATHDX_ROOT=/path/to/nvidia/mathdx/25.06` and a single
`THUNDERFFT_ARCHS` value adds `thff_paper_fft_cufftdx`, the same source built
with the cuFFTDx comparison (`--library cufftdx`). See
[Table 3](../artifact/table3/README.md#cufftdx-comparison-implemented-not-yet-measured)
for its CLI and candidate selection.

## Tiled 1D operation: `thff_conv1d`

```text
thff_conv1d [input_len=65536] [filter_len=9] [device=0] [validate=1] [warmup=5] [runs=50]
```

For a short run with validation enabled:

```bash
./build/sample/conv1d/thff_conv1d 1024 9 0 1 1 3
```

The program generates one complex input vector of length `input_len` and one
complex filter of length `filter_len`. The fused path stores complex values in
`half2`, with real and imaginary components in its two halves. It uses fixed
64-point FFT tiles and produces `input_len - filter_len + 1` complex outputs.
It requires `1 <= filter_len <= 64` and `input_len >= filter_len`.

The direct reference defines the operation explicitly:

```text
y[i] = sum(input[i + k] * filter[k], k=0..filter_len-1)
```

There is no conjugation in this complex valid correlation convention. The filter
is reversed when placed in the FFT buffer, and scaled by `1/64` to account for
the unnormalized inverse FFT. The filter FFT uses cuFFT and is prepared before
timing the fused kernel.

| Argument | Meaning |
| --- | --- |
| `input_len`, `filter_len` | Input and filter lengths |
| `device` | CUDA device index |
| `validate` | Nonzero computes the direct host reference; zero skips it |
| `warmup`, `runs` | Warmup launches and timed launches; use nonnegative warmup and positive runs |

The output lists the tile/launch configuration and `Kernel pipeline time` in
milliseconds per launch. With validation enabled it also prints
`Validation vs direct conv: L2_rel=... Linf_rel=...` and up to eight output/reference
pairs. The reference uses the original float inputs and filter, so these errors
include conversion to half precision. Validation is diagnostic: there is no
threshold or numerical failure exit code in this example.

`THFF_CONV1D_TILES_PER_WARP` is a CMake compile-time setting, defaulting to 1. It
changes the number of logical FFT blocks processed by a warp and requires a
rebuild; it is not another command-line argument.

## Full-length 1D convolution: `thff_conv1d_fullfft4096`

The executable retains its historical name but accepts three FFT lengths:

```text
thff_conv1d_fullfft4096 [N=4096] [K=9] [device=0] [B=1] [D=1] [warmup=5] [runs=30] [include_filter_fft=1] [validate=1] [real_input=1] [real_filter=1] [validate_channels=0]
```

For a small diagnostic run:

```bash
./build/sample/conv1d/thff_conv1d_fullfft4096 4096 9 0 1 1 1 3 1 1 1 1
```

| Argument | Meaning and constraints |
| --- | --- |
| `N` | FFT length: 256, 1024, or 4096 |
| `K` | Nonzero filter length, from 1 through `N`; padded to `N` |
| `device` | CUDA device index |
| `B`, `D` | Positive batch and channel counts; `B*D` must be divisible by 16 for `N=256`; the other two lengths process one transform per block |
| `warmup`, `runs` | Nonnegative warmup count and positive timed launch count |
| `include_filter_fft` | Nonzero times filter FFT preparation and permutation on every iteration; zero prepares them once before timing |
| `validate` | 1 enables direct-reference diagnostics and the correctness gate; 0 skips them |
| `real_input` | Must be 1; the complex-input path is unavailable |
| `real_filter` | Must be 1; complex filters are rejected because output is real-only |
| `validate_channels` | 0 validates every `(b, d)` sequence; a positive value validates that many sequences spread over `b` and `d`, plus the last one |

Input and output have contiguous real-half shape `[B][D][N]`, with one filter
per channel shared across batches. Filter storage has padded complex shape
`[D][N]`; its generated coefficients are divided by `N`. The 4096-point path
uses cuFFT for the filter FFT, followed by conversion to half precision. The
256- and 1024-point paths use ThunderFFT for that stage.

Validation prints three comparisons: raw output against direct circular
convolution, output divided by `N` against circular convolution, and output
divided by `N` against circular correlation. The corresponding references use
`input[(n-k) mod N]` and `input[(n+k) mod N]`, respectively, with no conjugation.
Each line reports relative L2 error and maximum absolute error. A final line
reports average pipeline milliseconds and the `include_filter_fft` setting.

The 4096-point filter-frequency permutation now matches the forward FFT register
layout. Previously, its row and column coordinates were transposed. Targeted
A100 / CUDA 12.9 checks with seed 42, real inputs and filters, `B=D=1`, and
`K=1,3,9,17,53` gave output/`N` circular-convolution relative L2 errors between
`5.20e-4` and `9.90e-4`. For `K=9`, correcting the indices reduced the error from
`0.281035` to approximately `0.000964` with the same FP16 arithmetic; an independent
double-precision direct reference confirmed the result.

With `validate=1`, output/`N` circular-convolution relative L2 error must be finite
and no greater than `5e-3`. The program prints `PASS` or `FAIL` and returns a
nonzero status on failure. `validate=0` disables this gate. Real input and real
filter flags must both equal 1; complex paths are rejected.

The targeted measurements above predate the latest general FFT fixes. Subsequent
remote A100 validation passed eight real convolution cases, including lengths
256, 1024 and 4096; the H100 campaign also completed successfully. These small
checks do not validate the full paper workload or establish a matched speedup.
Use a common reference and equivalent operation definitions before interpreting
comparison timings.
Keep `THFF_SKIP_MID_ROUNDTRIP=0`, the default CMake setting, when investigating
correctness; the alternative explicitly skips a reorder for a timing experiment.

## Optional native comparison: `thff_conv1d_compare`

CMake creates this target when it finds the cuDNN library. The cuDNN development
headers must also be available to the compiler. Build the tiled ThunderFFT
executable alongside it:

```bash
cmake --build build --target thff_conv1d thff_conv1d_compare -j
```

```text
thff_conv1d_compare [L=268435456] [Kb=3] [Ke=63] [Ks=2] [device=0] [B=1] [D=1] [warmup=3] [runs=20] [csv=compare_flash_cudnn_thunder.csv] [thff_bin=sibling thff_conv1d]
```

`Kb`, `Ke`, and `Ks` set the inclusive filter-length sweep. Use `1 <= Kb <= Ke <= 64`,
a positive step, positive `B`/`D`, and positive timed runs. Pass a short input
length explicitly; the default allocates a very large workload.

```bash
./build/sample/conv1d/thff_conv1d_compare 1024 3 9 2 0 1 1 1 3 compare.csv
```

The CUDA and cuDNN paths use real half inputs `[B][D][L]`, filters `[D][K]`, and
bias `[D]`, with output `[B][D][L-K+1]`. This driver initializes them to zero. The
CUDA row named `flash_conv1d_cuda` is the direct kernel in this source file,
rather than an invocation of the FlashFFTConv package. The ThunderFFT row
spawns `thff_conv1d` with validation disabled, which generates its own complex
data and always uses `B=D=1`.

The driver prints per-filter timings and writes CSV columns
`method,L,B,D,K,output_len,outputs_total,time_ms,ns_per_output`. It performs no
cross-implementation output comparison. The inputs, arithmetic, and batch/channel
coverage differ across paths; this output alone cannot establish equivalent
operations or a matched speedup.

## 2D convolution driver: `thff_conv`

Configure MathDX, including its cuFFTDx and bundled CUTLASS/CuTe headers:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80 -DMATHDX_ROOT=/path/to/nvidia/mathdx/25.06
cmake --build build --target thff_conv -j
./build/sample/conv2d/thff_conv 2048 33 cufftdx
```

The arguments are `N f cufft|cufftdx|cudnn [warmup=5] [runs=20] [device=0]`; `f`
is odd from 3 to 59 for `cufftdx`. One library runs per invocation on seeded
input, passes a sampled direct-reference relL2 ≤ 5e-3 gate (including tile
boundaries), and then prints its mean time. cuDNN is available when CMake finds
it. cuDNN uses the real parts only (real FP32), the FFT paths complex FP32; see
[Figure 5](../artifact/figure5/README.md) for the exact operation. `thunder.cu`
and the tile kernels remain experimental sources and are not called by this driver.

## Kernel composition

For a concrete fusion example, read `conv1d_direct_fft_kernel` and its launch in
[conv1d/main.cu](conv1d/main.cu). The full-length example also contains
`thunder_convN_block_kernel_realin_realout` in
[conv1d/conv1d_fullfft_l4096_thunder.cu](conv1d/conv1d_fullfft_l4096_thunder.cu),
whose numerical limitations are described above. These kernels keep the FFT
and pointwise multiplication in one launch.

The device API in [thunderfft.cuh](../include/thunderfft/thunderfft.cuh) requires
the caller to arrange the following:

- Select an implemented type/size/direction specialization. Use its
  `batch_per_block<N>` and `warp_per_block<N>` constants with a block of
  `dim3(32, warp_per_block<N>)`; handle complete logical batches and any tail
  outside the FFT primitives.
- Allocate each thread's register array with
  `N * batch_per_block<N> / (32 * warp_per_block<N>)` complex entries. Its ordering
  is internal to the specialization; it is not an ordinary sequential slice of
  the frequency vector.
- Supply the matching input/output `layout_t` and sufficient aligned dynamic
  shared memory. The examples use
  `sizeof(complex_type) * (N + pad_h(N)) * batch_per_block<N>` bytes. Shared memory
  also serves as the register FFT's workspace; follow the selected kernel's
  requirements before reusing or resizing it.
- Precompute the required direction-specific register twiddles. The fused half
  examples call `unit_fp16::make_reg_b_precompute<N, direction>` and pass their
  arrays to `ThunderFFT_kernel_reg`; host initialization does not replace these
  arguments.
- Preserve the block-wide synchronization around shared-memory movement and
  workspace use. Insert the size-specific register reorder needed between
  forward and inverse stages, and apply the intended normalization.

For `layout_t`, a shared-memory logical element has index
`j * batch_stride + i * elem_stride`; periodic padding changes this to
`idx + (idx / pad_period) * pad`; `pad_period` must remain positive even when
`pad=0`. The global buffers used by the layout helpers
remain contiguous `[batch][N]`. A `reversed` input layout requests the kernel's
bit-group permutation, rather than simply reversing the array. Matching the
layout, register ordering, filter permutation, and direction is necessary before
applying frequency-domain operations.

Use the complete source examples to understand these contracts. A declaration
in the public header alone does not mean that a generic implementation or every
specialization exists.

## Paper artifact

Paper reproduction instructions and experiment data are maintained separately:

```bash
git switch artifact
```

On that branch, see `artifact/README.md` and use the root `./run.sh --list` to
find the per-figure/table workflows. The sample smoke check and timing programs
above do not replace those reproduction workflows.
