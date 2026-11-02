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

## Supported configurations and current limits

The quick-start validation covers the following entry point:

| API | Storage and arithmetic | Transform | Batch requirement |
| --- | --- | --- | --- |
| `ThunderFFT_global<float, 64, true>` | `float2` complex storage; TF32 Tensor Core arithmetic | Length-64 forward C2C, unnormalized, natural-order input/output | Positive multiple of 16 |

`float` storage does not imply full-precision FP32 multiplication. The source
also contains `half`/`half2` routines and specializations for lengths 128, 256,
512, 1024, 2048 and 4096; these remain experimental outside the documented
example. The 512-point inverse has no specialization, the 2048-point inverse is
incomplete, and several FP16 host-wrapper paths require twiddle data that the
wrapper currently does not pass. See [current limitations](sample/README.md#current-limitations)
before using another configuration.

The host wrapper requires `batch % batch_per_block<N> == 0`; its kernel does not
guard a partial final block. It calls `cudaStreamSynchronize(stream)` internally.
The twiddle cache is shared per scalar type and holds one FFT length at a time;
use one CUDA device and serialize initialization, execution and finalization.

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

`true` selects the forward transform. Implemented inverse routines use the
positive exponent without a `1/N` normalization; apply scaling explicitly when
using a validated inverse path. Keep initialization before the first transform
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

`benchmark` is an FFT timing experiment. The convolution programs are experimental
and print diagnostic errors without enforcing a correctness threshold.

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

See the [artifact task guide](artifact/README.md) for the ten assignment units,
paper settings, adapter contract and current implementation status. The supplied
paper calls 1D convolution Figure 4 and 2D convolution Figure 5.

```bash
./run.sh --list                 # inspect assignments and dependencies
./run.sh --dry-run              # plan all evaluations without GPU work
./run.sh                        # run all available smoke tasks; record pending work
./run.sh table5                 # run one evaluation
./run.sh table3 --profile paper # request the paper workload
```

New outputs go to `result/<evaluation>/<run_id>/`; batch manifests and summaries
go to `result/runs/<run_id>/`. Task folders include owner placeholders, required
work and completion criteria. Most adapters are currently unfinished and return
78; the launcher continues and returns 2 for a pending-only batch. Table 5 has a
bounded measurement smoke path; other task READMEs describe the missing work.

Existing checked-in CSVs remain historical local outputs. They have not been
revalidated as paper reference results. Numerical correctness and equivalent
baseline definitions remain explicit completion requirements.
