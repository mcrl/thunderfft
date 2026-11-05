# Table 5 — numerical accuracy and dynamic range

**Status:** runnable numerical evaluation with validated CSVs and exported tables. Conditions the paper does not state are recorded in every result's `metadata.json` (`paper_conditions`).

The Table 5 caption scales Gaussian inputs so that `||x||₂ / sqrt(N) = alpha` and reports relative L2 error against a double-precision FFT at `alpha=1` and the largest finite `alpha=2^k`, for `N=64,256,1024,4096`.

## Implementation

| CSV precision | Implementation | Input |
|---|---|---|
| `fp16` | ThunderFFT FP16 | FP32 input rounded to FP16 |
| `tf32` | ThunderFFT TF32 | FP32 input |
| `fp32` | cuFFT C2C | FP32 input |
| `fp16_cufftdx` | cuFFTDx FP16 (`ElementsPerThread 8`, `FFTsPerBlock 2`, no fast-math) | FP32 input rounded to FP16 |

Every accuracy row uses cuFFT Z2Z FP64 of the FP32 input as reference, so FP16 error includes input rounding. Relative L2 is taken over all outputs of the batch. Each FFT `b` has independent Gaussian real/imaginary parts seeded by `seed+b` and is normalized on its own. `fp16_cufftdx` is a separate executable, built only when the build directory is configured with `MATHDX_ROOT` and one `THUNDERFFT_ARCHS` entry.

The paper fixes batch 65536 (Section 6.1); the `paper` profile uses it and `smoke` uses 64. The seed (default 1234), real/imaginary construction, normalization scope, reference input, aggregation and scan window are not stated in the paper. The cuFFTDx comparison follows revised paper text that is not in the 2026-10-13 PDF.

At `N=4096`, batch 65536 has 268,435,456 complex elements and substantial host/GPU allocation requirements (multiple GB per buffer and several live buffers); use a deliberate smaller batch for a bounded development run.

## Run

From the repository root:

```bash
./run.sh table5
```

The default `smoke` profile uses `N=64`, batch `64`, seed `1234`, Gaussian inputs, FP16/TF32/FP32 accuracy at `alpha=1`, FP16 finite exponents `8…13`, and TF32/FP32 finite exponents `120…127`. These narrow windows exercise both sides of the expected boundary; they are not the complete paper experiment.

The paper profile checks all four paper sizes and each integer exponent `0…127` for all three precision paths:

```bash
./run.sh --profile paper table5
```

For all four sizes and the complete exponent scan with a bounded population:

```bash
THUNDERFFT_TABLE5_BATCH=64 ./run.sh --profile paper table5
```

This measures the full size/exponent grid at batch 64; it does not establish the published population. `THUNDERFFT_TABLE5_N_LIST=64,1024` selects a documented subset, including for a `paper` profile; metadata records the exact list and whether it contains all four paper sizes. Supported sizes are `64,128,256,512,1024,2048,4096`; duplicate or invalid sizes are rejected. Batch must be a positive multiple of 16; `THUNDERFFT_TABLE5_SEED` defaults to `1234`. Every override is stored in metadata. Each native invocation (one per selected phase) has a bounded `THUNDERFFT_TABLE5_TIMEOUT` (default 600 seconds). Increase that explicit limit for a large population if needed. No result from a timed-out invocation is accepted as a complete evaluation.

The wrapper uses the shared runner's build directory, job count, Python, and result directory. `--device` selects a logical ordinal within the inherited visible GPU set. The native executable has no device option, so the wrapper narrows its subprocess mask to that selected member and executes on local CUDA device 0. For example, `CUDA_VISIBLE_DEVICES=2,5` with `--device 1` selects physical GPU 5; UUID and MIG identifiers are preserved. With no inherited mask, the requested ordinal becomes the subprocess mask. Empty/disabled masks and out-of-range selections return pending before configuration. Metadata records whether the original mask was set, its exact value, the requested ordinal, and the effective mask.

Direct invocation is also supported and defaults to a result directory under the repository root:

```bash
bash artifact/table5/run.sh
```

## Collect separate CSV parts and assemble later

The six phases can be collected independently with `THUNDERFFT_TABLE5_PHASES`:
`accuracy_reduced`, `accuracy_fp32_cufft`, `range_fp16`, `range_tf32`,
`range_fp32_cufft`, and `accuracy_fp16_cufftdx`. The default selects all five
CUDA-only phases, plus `accuracy_fp16_cufftdx` when cuFFTDx is configured. Combine this with
`THUNDERFFT_TABLE5_N_LIST` to split work by FFT size. A selected phase must
finish its entire declared size/exponent grid and pass the same checks as a
full run. Each invocation stores an immutable result bundle; use a unique
`--part` name when adding it to a collection.

For example, collect the accuracy CSVs and the three range CSVs in separate
executions, then assemble them without using the GPU:

```bash
THUNDERFFT_TABLE5_PHASES=accuracy_reduced,accuracy_fp32_cufft \
  ./run.sh table5 --collection ae --part table5-accuracy

THUNDERFFT_TABLE5_PHASES=range_fp16 \
  ./run.sh table5 --collection ae --part table5-range-fp16

THUNDERFFT_TABLE5_PHASES=range_tf32,range_fp32_cufft \
  ./run.sh table5 --collection ae --part table5-range-wide

MATHDX_ROOT=/path/to/nvidia/mathdx/25.06 THUNDERFFT_TABLE5_PHASES=accuracy_fp16_cufftdx \
  ./run.sh table5 --collection ae --part table5-accuracy-cufftdx

./run.sh --assemble --collection ae
```

Use `--profile paper` consistently for a paper-profile collection. To collect
one size at a time, add `THUNDERFFT_TABLE5_N_LIST=64`, then collect the other
sizes with different part names. Keep batch and seed fixed across the parts;
any batch override must be repeated in each invocation. No new measurements
are launched by assembly.

Assembly can run after any part is collected. Missing accuracy or range
cells remain blank in CSV and show `pending` in the readable table. The
collection reports the exact missing phase/size combinations, including sizes
beyond the profile list. An incomplete or nonmonotonic scan is rejected. A
complete scan is `observed` only if it contains both finite and non-finite
exponents; otherwise it is `right_censored` or `left_censored` and gives no
overflow boundary. Identical duplicate components are deduplicated;
conflicting duplicates are rejected.

Parts must match in profile, batch, seed, input/reference definitions,
implementation labels, scan windows, source hashes, binary hash, build/compiler
settings, and the measured GPU UUID/CUDA/cuFFT environment; the cuFFTDx
executable is compared only among cuFFTDx parts. This prevents combining
measurements taken under incompatible conditions. Assembly revalidates raw
CSVs, records source-file hashes, and preserves the original bundles.
A legacy bundle without a GPU UUID can be assembled on its own or deduplicated
against an identical bundle; it cannot be combined with independently collected
parts. PCI bus IDs and GPU names alone do not establish device identity across hosts.
Completing the expected CSV grid establishes collection completeness; it does
not establish the Table 5 conditions the paper leaves unstated.

## Generated outputs

All new files go under `$THUNDERFFT_RESULT_DIR`, a runner-generated directory inside repository-root `result/`:

- `accuracy_reduced.csv` / `.log`: TF32 and FP16 error against FP64 at `alpha=1`.
- `accuracy_fp32_cufft.csv` / `.log`: cuFFT FP32 error against FP64 at `alpha=1`.
- `accuracy_fp16_cufftdx.csv` / `.log`: cuFFTDx FP16 error against FP64 at `alpha=1`; its range cells are `not_measured`.
- `range_fp16.csv`, `range_tf32.csv`, `range_fp32_cufft.csv` and their `.log` files: raw exponent scans with finite-output flags.
- `data.csv`: canonical combined long-form raw rows, labeled by phase, source CSV, implementation, input, and reference. Finite-only error values remain `nan`.
- `table.csv` and identical `table5_summary.csv`: per-size and per-precision measured errors, largest finite and first non-finite exponents observed, scan bounds, `boundary_status`, actual implementation labels, and separate accuracy/range completion flags. Uncollected cells remain blank.
- `table.md`: readable exported table.
- `comparison.csv`: informational differences and error ratios against the retained source reference CSV. Non-paper sizes and cuFFTDx rows have blank reference columns; this comparison has no invented paper-acceptance threshold.
- `metadata.json`, `commands.txt`: population and overrides, seed, input definition, `paper_conditions`, precision provenance, device selection, source commit/status and SHA-256 hashes, binary hash, CMake settings, compiler version, measured GPU/CUDA/cuFFT versions, build and exact native commands.
- `*.environment.json`: selected GPU UUID, PCI bus identity, total global memory, compute capability, driver API/runtime/compile versions and cuFFT version from each native invocation.
- `validation.json`: row completeness, finite alpha=1 outputs, broad numerical sanity limits (`1e-2` for FP16/TF32/cuFFTDx FP16, `1e-5` for cuFFT FP32), monotonicity of the scanned finite boundary, and consistent measured environment across invocations.

`run.sh` returns nonzero if those numerical sanity checks fail. These are broad implementation checks, not author-approved paper acceptance thresholds. Neither process completion nor these checks establish the exact published numbers. Metadata explicitly leaves scientific validation unassessed.

## Interpretation

The evaluator is ready to execute and export its measurements without cuFFTDx tuning parameters or downloaded baseline repositories. CUDA Toolkit's bundled cuFFT supplies the FP32 path and FP64 reference with default planning; no cuFFT tuning data is required here. The runner does not adjust GPU clocks or power limits.

The conditions the paper does not state need to be reconciled with the author before measured values are read as a reproduction of the published table. The paper lists approximate FP16 errors `4.34e-4…6.54e-4`, TF32 errors `1.35e-3…2.64e-3`, FP32 errors `1.24e-7…1.85e-7`, and finite exponents FP16 `11,10,9,8` and TF32/FP32 `123,122,121,120` for the four increasing sizes. The unchanged `table5_accuracy_range_summary_with_fp32.csv` retains the source reference values used by `comparison.csv`; it is not a fresh result. Fresh outputs always live in root `result/`.

A scan's largest finite exponent is an observation over its recorded population. `right_censored=true` means all samples were still finite at the scan maximum, and `left_censored=true` means none were finite within the selected window. Neither condition establishes an unobserved boundary. A finite-output range does not assert small numerical error at every scale.

CPU-only checks are available with `python3 artifact/table5/check_summary.py` and `python3 artifact/table5/test_assemble.py`; they exercise complete data, missing/duplicate rows, excessive alpha=1 error, nonmonotonic overflow boundaries and inconsistent environments. No CUDA math is changed by this adapter.

The retained `sweep_layout_conv1d.py` and `sweep_lmid_conv1d.py` are legacy convolution tuning helpers, not Table 5 numerical evaluations. They can mutate layout source or CMake settings and are excluded from the unified runner. Their default CSV locations now use root `result/table5/` (or `$THUNDERFFT_RESULT_DIR`). Any historical CSVs must remain immutable reference data.
