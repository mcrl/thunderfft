# ThunderFFT

ThunderFFT is a header-only CUDA FFT implementation using NVIDIA Tensor Cores.
It provides a host launch wrapper and device routines for composing FFTs with
other work inside a CUDA kernel.

This distribution includes the library, samples and the paper's evaluation
scripts in `artifact/`.

For artifact evaluation, start with [AE_README.txt](AE_README.txt): one-command
table and figure generation, default/full modes, estimated times and individual runs.

## Quick start

Use a CUDA toolkit, a C++17 host compiler and CMake 3.18 or newer. The documented
starting configuration is an NVIDIA A100 (SM 80), CUDA 12.9, GCC 9.4 and CMake 4.1.
From the repository root:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80
cmake --build build --target thff_fft -j2
./build/sample/fft1d/thff_fft
```

This example computes 16 complex, length-64 forward transforms and compares them
with cuFFT. It prints relative L2 and Linf errors, then `PASS` or `FAIL`. Success
requires finite errors no greater than `5e-3`; failure returns a nonzero exit
code. The threshold applies to this example's bounded input. See the
[sample guide](sample/README.md#basic-fft-thff_fft) for arguments and validation
details.

If `nvcc` is not on `PATH`, add
`-DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc` and
`-DCUDAToolkit_ROOT=/path/to/cuda` to the configure command. `THUNDERFFT_ARCHS`
controls compilation; changing it does not establish correctness on another GPU.

## Implemented configurations and validation

The host wrapper and register core implement these configurations:

| Component type | Storage and arithmetic | Transform | Host-wrapper batch |
| --- | --- | --- | --- |
| `float` | `float2` complex storage; TF32 Tensor Core arithmetic | Length 64, 128, 256, 512, 1024, 2048 or 4096; forward/inverse C2C, unnormalized | Any nonnegative batch; partial final blocks are masked |
| `half` | `half2` complex storage; FP16 Tensor Core arithmetic | Same lengths and directions | Same; required per-thread coefficients are prepared internally |

`float` storage does not imply full-precision FP32 multiplication. Implementation
coverage does not guarantee accuracy for every signal or input range. The
quick-start example remains a bounded, length-64 forward check and deliberately
accepts only positive multiples of 16.

Build the optional correctness suite to exercise both types, both directions,
all seven sizes, public-wrapper partial batches and guarded register workspaces:

```bash
cmake -S . -B build -DTHUNDERFFT_ARCHS=80 -DTHUNDERFFT_BUILD_VALIDATION=ON
cmake --build build --target thff_validate -j2
./build/sample/validation/thff_validate --quick
./build/sample/validation/thff_validate --csv result-validation.csv
CUDA_VISIBLE_DEVICES="" ./build/sample/validation/thff_validate --host-only
```

The second invocation runs the full matrix; no `--full` flag is needed. See the
[validation guide](sample/validation/README.md) for inputs, references, tolerances
and exit codes. On 2026-11-02, a remote A100 campaign passed all 500 full-matrix
cases at the `5e-3` relative L2 threshold. Its largest observed FFT relative L2 errors were
`6.6232e-4` for FP16 and `2.6405e-3` for TF32. Compute Sanitizer memcheck,
racecheck and synccheck each completed 88 quick-matrix cases with no reported
errors, and eight real 1D convolution cases passed their correctness gate.
The remote H100 campaign passed the same checks; its largest TF32 relative L2
error was `2.6454e-3`. At that revision, forward kernel latency over the 28
Table 3 configurations was -2.78% to +6.61% versus the paper on A100, and
+4.38% to +15.98% on H100.
The H100 gap remains unresolved; full paper performance reproduction is not
established. These results apply to the tested inputs and configurations.
On 2026-11-09, paper-profile Table 3 and Table 4 completed on a rented
H100 80GB PCIe and B200, including all 28 ThunderFFT/cuFFTDx comparison
configurations per GPU and the benchmarks' correctness gates. Measured
cuFFTDx/ThunderFFT speedups were 1.20--3.34x on H100 and 1.10--3.54x on B200.
ThunderFFT latency differed from the stored paper references by up to about
19% and 12%, respectively. These measurements establish those benchmark
paths, not full validation-matrix coverage or exact paper timing reproduction
on either GPU.

On 2026-11-03, reusing the initialized twiddle table for the SM80 float N=1024
path reduced matched A100 TF32 register-input latency from `137.24` to
`126.53` microseconds (-7.81%; paper: `127.85`), and shared-input latency from
`160.78` to `155.63` microseconds (-3.20%; paper: `155.34`). These are median
R=1000 differential compute times at batch 65536, with nine register and six
shared observations per version. The corrected build passed all 500 full-matrix
cases and 112 checks of the actual benchmark entry points, including per-transform
error checks. Memcheck, initcheck and synccheck passed 88 quick cases each;
racecheck reported no hazards in a short repeated N=1024 benchmark. This local
run used unmodified clocks on a GPU with intermittent background activity.
It does not establish a single-FFT end-to-end speedup or update the other
Table 3 configurations. Native SM90 and SM100 builds retain their previous
code paths; SM80 PTX JIT-compiled on a newer GPU would use the cached path.

The host wrapper synchronizes its supplied stream before returning and throws
`std::runtime_error` for CUDA failures. A negative batch, null buffers for a
positive batch or identical input/output pointers throws `std::invalid_argument`.
Zero batch is a no-op. Positive batches require separate, non-overlapping buffers.
Twiddle caches are keyed by CUDA device and component type, holding one FFT
length per pair. Finish all dependent kernels before changing that device's
cache or calling finalization. See the [API contracts](sample/README.md#current-limitations).

## Basic usage

Include the public header in a CUDA translation unit:

```cpp
#include <thunderfft/thunderfft.cuh>

// Select the CUDA device and allocate/upload d_input before calling.
// Both pointers must refer to separate buffers of 16 * 64 float2 values,
// with element n of signal b at offset b * 64 + n; x/y are real/imaginary.
void forward_fft64_batch16(float2* d_input, float2* d_output,
                          cudaStream_t stream = 0) {
    thunderfft::ThunderFFTInitialize<float>(64);
    thunderfft::ThunderFFT_global<float, 64, true>(
        d_input, d_output, 16, stream);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream));
    thunderfft::ThunderFFTFinalize<float>();
}
```

`true` selects the forward transform. Inverse routines use the
positive exponent without a `1/N` normalization; apply scaling explicitly when
using an inverse path. Keep initialization before the first transform
and finalization after the last use of its twiddle tables. The complete
[FFT example](sample/fft1d/main.cu) includes allocation, deterministic input,
reference computation, error checks and cleanup.

For a CMake consumer, add this repository with
`add_subdirectory(... EXCLUDE_FROM_ALL)` and link the header-only `thunderfft`
target. This keeps the bundled samples out of the consumer's default build. The
target supplies the include path and CUDA runtime dependency; the example links
cuFFT separately for its reference calculation.

## FFTs inside a CUDA kernel

`ThunderFFT_kernel_reg<T, N, BPB, forward>` operates on each thread's FFT fragment.
The caller supplies the matching block shape, register fragment, shared-memory
workspace and any required twiddle operands. Layout and synchronization are part
of the call contract; fragments cannot be treated as a contiguous natural-order
signal. Data-movement helpers connect global/shared memory and registers.

See [kernel composition](sample/README.md#kernel-composition) and the
[public header comments](include/thunderfft/thunderfft.cuh) for these conditions.
`ThunderFFT_kernel_smem`, `ThunderFFT_kernel_smem_pad` and `preprocess_W<T>` are
currently declarations without definitions and are not usable entry points.

## Examples and optional dependencies

The [sample guide](sample/README.md) lists each program's build target, arguments,
output and validation behavior. Additional targets can be built explicitly:

```bash
cmake --build build --target benchmark thff_conv1d thff_conv1d_fullfft4096 -j2
```

`benchmark` is an FFT timing experiment. Full-length 1D convolution enforces a
relative L2 correctness threshold when validation is enabled. The 2D `thff_conv`
driver checks a direct reference before timing; the earlier tiled 1D sample
prints diagnostic errors. See the sample guide for each program's validation scope.

The 2D target `thff_conv` builds cuFFT and tiled ThunderFFT paths with the CUDA
Toolkit. Its optional cuFFTDx path needs NVIDIA MathDX and bundled CUTLASS/CuTe
headers; configure with `-DMATHDX_ROOT=/path/to/nvidia/mathdx/25.06`. The optional native
`thff_conv1d_compare` target needs cuDNN's library and development headers.

## Repository layout

- `include/thunderfft/`: public header and implementation details.
- `sample/`: runnable examples, the active benchmark and their usage guide.
- `bench/`: earlier benchmark sources, outside the current CMake build.
- `artifact/figure2/` through `artifact/figure4/`: figure-specific evaluation tasks.
- `artifact/table2/` through `artifact/table7/`: table-specific evaluation tasks.
- `run.sh`: unified evaluation launcher on this branch.
- `result/`: generated outputs, separated by evaluation and run ID; ignored by Git.

## Artifact experiments

See the [artifact guide](artifact/README.md) for setup, running the paper's
evaluations locally or on Vast.ai, and assembling results:

```bash
bash artifact/setup.sh
./run.sh
```

After setup, `run.sh` loads `~/.thunderfft_env` and runs the complete
paper-profile workflow: validation and every runnable table/figure on an
A100, Table 3 and Table 4 on rented H100 80GB PCIe and B200 GPUs, then
automatic assembly of the results. `./run.sh all` does the same. The cloud
steps use the evaluator account's credentials in `~/cloud_key/thunderfft/`
and incur rental charges. By default, the workflow waits up to 120 seconds
for the local A100, then rents an A100 if it remains unavailable. Use
`--a100-backend local` to require the local A100, or `--a100-backend vast`
to choose a rented A100 directly. Final tables and Figure 2--5 PDFs are written to
`result/collections/<collection>/assembled/`; the generated collection name
is printed during the run.

Use `./run.sh all --cloud-plan` to check the required offers and combined
cost without starting measurements or renting a GPU.
`./run.sh all --backend local --profile paper` runs all A100 evaluations only.
Named evaluations retain their local smoke defaults, for example
`./run.sh validation table3`.
Table 7 records its unavailable historical stages as missing; generating
all outputs does not establish those missing paper measurements.
