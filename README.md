# ThunderFFT

ThunderFFT is a header-only CUDA FFT implementation using NVIDIA Tensor Cores.
It provides a host launch wrapper and device routines for composing FFTs with
other work inside a CUDA kernel.

This is the `artifact` branch. It includes the library, samples, paper
reproduction scripts and historical local experiment outputs. The `main` branch
contains the library and samples without the `artifact/` directory.

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
B200 has a native CUDA 13/SM 100 build but remains
unmeasured because no suitable single-GPU rental was available.

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
relative L2 correctness threshold when validation is enabled; the tiled 1D and
2D programs still print diagnostic errors without a numerical failure gate.
See the sample guide before using their timing output.

The 2D target `thff_conv` additionally needs NVIDIA MathDX with cuFFTDx and bundled
CUTLASS/CuTe headers; configure with `-DMATHDX_ROOT=/path/to/nvidia/mathdx/25.06`.
A build of all targets includes this dependency. The optional native
`thff_conv1d_compare` target needs cuDNN's library and development headers.

## Repository layout

- `include/thunderfft/`: public header and implementation details.
- `sample/`: runnable examples, the active benchmark and their usage guide.
- `bench/`: earlier benchmark sources, outside the current CMake build.
- `artifact/figure2/` through `artifact/figure5/`: figure-specific evaluation tasks.
- `artifact/table2/` through `artifact/table7/`: table-specific evaluation tasks.
- `artifact/flashfftconv/`: notes on the external dependency.
- `run.sh`: unified evaluation launcher on this branch.
- `result/`: generated outputs, separated by evaluation and run ID; ignored by Git.

## Artifact experiments

See the [artifact task guide](artifact/README.md) for the ten paper assignments,
the correctness task, paper settings and implementation status. The supplied
paper calls 1D convolution Figure 4 and 2D convolution Figure 5.

```bash
./run.sh --list                 # inspect assignments and dependencies
./run.sh --dry-run              # preview available tasks without GPU work
./run.sh                        # validation, Table 2/3/4/5/7 and Figure 2 smoke checks
./run.sh all                    # include unfinished assignments as pending
./run.sh table5                 # run one evaluation
./run.sh table3 --profile paper # request the paper workload
```

New outputs go to `result/<evaluation>/<run_id>/`; batch manifests and summaries
go to `result/runs/<run_id>/`. Task folders include owner placeholders, required
work and completion criteria. The default batch uses CUDA Toolkit and Python;
install the plotting dependency with `python3 -m pip install -r artifact/requirements.txt`.
The scripts require Python 3.9 or newer; set `PYTHON=/path/to/python3` if `python3` is older.
It produces correctness results, Table 2's calculated rows, all 28 ThunderFFT
Table 3 configurations, Table 4's unit throughput derived from those CSVs and
independently measured ideal MMA rate, the corresponding A100 Figure 2 subset,
and Table 5 accuracy/range measurements. The excluded cuFFTDx comparisons and external FFT
libraries remain explicit in the output metadata. FP32 in Table 5 is cuFFT.
`./run.sh all` also visits unfinished tasks, which return 78; the batch returns
2 for pending-only work and 1 for actual failures. See each task's README for
profile settings and exact scope. `bash artifact/check.sh` runs CPU checks.

Existing checked-in CSVs remain historical local outputs. They have not been
revalidated as paper reference results. Numerical correctness and equivalent
baseline definitions remain explicit completion requirements.
