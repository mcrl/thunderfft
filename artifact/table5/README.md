# Table 5 — numerical accuracy and dynamic range

**Owner:** unassigned. **Status:** measurement adapter available; paper provenance and exact settings need author confirmation.

The supplied PDF reports complex Gaussian inputs normalized per FFT so that `||x||₂ / sqrt(N) = alpha`. It measures relative L2 error against a double-precision FFT at `alpha=1`, and the largest input scale `alpha=2^k` for which every output remains finite, for `N=64,256,1024,4096`.

## Existing implementation and limits

`main.cu` implements the accuracy and finite-output scans. Its paths are:

| CSV precision | Actual implementation | Accuracy reference |
|---|---|---|
| `fp16` | ThunderFFT with FP16 input/output | cuFFT Z2Z FP64 |
| `tf32` | ThunderFFT float/TF32 path | cuFFT Z2Z FP64 |
| `fp32` | **cuFFT C2C, not ThunderFFT** | cuFFT Z2Z FP64 |

Inputs are generated in FP32; the double reference transforms those FP32 values, and FP16 error includes input conversion. A reduced finite-output scan is not an accuracy check. `--fast-finite-only` intentionally stores `nan` in error columns; `all_finite` is the quantity used for the dynamic-range result. Largest finite scale depends on the input population and seed.

The C++ executable's original batch default is **65536**. At `N=4096` this has 268,435,456 complex elements and substantial host/GPU allocation requirements. The wrapper requires the paper profile's population to be explicitly selected rather than silently allocating it or presenting a small smoke check as an exact reproduction.

## Run

From the repository root:

```bash
./run.sh table5
```

The default `smoke` profile uses `N=64`, batch `64`, seed `1234`, Gaussian inputs, FP16/TF32/FP32 accuracy at `alpha=1`, FP16 finite exponents `8…13`, and TF32/FP32 finite exponents `120…127`. These narrow windows exercise both sides of the expected boundary; they are not the complete paper experiment.

The paper profile checks all four paper sizes and each integer exponent `0…127` for all three precision paths. Set a batch explicitly after confirming the author's population and available memory, for example:

```bash
THUNDERFFT_TABLE5_BATCH=65536 ./run.sh --profile paper table5
```

Without `THUNDERFFT_TABLE5_BATCH`, that profile returns **78 (pending)** with an explanation before building or allocating GPU memory. A deliberate smaller population can be used for development, but it must be reported as such. `THUNDERFFT_TABLE5_BATCH` must be a positive multiple of 16; `THUNDERFFT_TABLE5_SEED` defaults to `1234`. Each native invocation has a bounded `THUNDERFFT_TABLE5_TIMEOUT` (default 600 seconds).

The wrapper uses the shared runner's build directory, job count, Python, and result directory. `--device` selects a logical ordinal within the inherited visible GPU set. The native executable has no device option, so the wrapper narrows its subprocess mask to that selected member and executes on local CUDA device 0. For example, `CUDA_VISIBLE_DEVICES=2,5` with `--device 1` selects physical GPU 5; UUID and MIG identifiers are preserved. With no inherited mask, the requested ordinal becomes the subprocess mask. Empty/disabled masks and out-of-range selections return pending before configuration. Metadata records whether the original mask was set, its exact value, the requested ordinal, and the effective mask.

Direct invocation is also supported and defaults to a result directory under the repository root:

```bash
bash artifact/table5/run.sh
```

## Generated outputs

All new files go under `$THUNDERFFT_RESULT_DIR`, a runner-generated directory inside repository-root `result/`:

- `accuracy_reduced.csv` / `.log`: TF32 and FP16 error against FP64 at `alpha=1`.
- `accuracy_fp32_cufft.csv` / `.log`: cuFFT FP32 error against FP64 at `alpha=1`.
- `range_fp16.csv`, `range_tf32.csv`, `range_fp32_cufft.csv` and their `.log` files: raw exponent scans with finite-output flags.
- `data.csv`: canonical combined long-form raw rows, labeled by phase, source CSV, actual implementation, and reference. Finite-only error values remain `nan`.
- `table.csv` and identical `table5_summary.csv`: per-size and per-precision measured errors and largest finite exponent observed, with scan bounds, overflow/censoring flags, and actual implementation labels.
- `metadata.json`, `commands.txt`: population, seed, input definition, precision provenance, device selection, and exact native commands.
- `validation.json`: row completeness, finite alpha=1 outputs, broad numerical sanity limits (`1e-2` for FP16/TF32, `1e-5` for cuFFT FP32), and monotonicity of the scanned finite boundary.

`run.sh` returns nonzero if those numerical sanity checks fail. These are broad implementation checks, not author-approved paper acceptance thresholds. Neither process completion nor these checks establish the exact published numbers or FP32 provenance. Metadata explicitly leaves scientific validation unassessed.

## Work to assign

1. Confirm whether the paper's FP32 column is intended to measure cuFFT or a separate ThunderFFT FP32 implementation; reconcile source and labels before publication.
2. Confirm batch, seed, GPU/CUDA versions, and input normalization/rounding against the submitted Table 5 experiment. Select the paper population explicitly and document peak memory and duration.
3. Set author-approved error tolerances and acceptable numerical variation. The paper lists approximate FP16 errors `4.34e-4…6.54e-4`, TF32 errors `1.35e-3…2.64e-3`, FP32 errors `1.24e-7…1.85e-7`, and finite exponents FP16 `11,10,9,8` and TF32/FP32 `123,122,121,120` for the four increasing sizes; these are reference values, not fresh results.
4. Run the paper profile on the agreed population, inspect all raw failures and overflow boundaries, compare to the PDF, and add a deterministic formatted-table generator plus a concise comparison note under the result directory.
5. Record a validated environment and expected runtime. Add boundary/reference tests if source changes become necessary.

Completion requires provenance confirmation, fresh measurements at the agreed population, a documented comparison to Table 5, and an exported table. No CUDA math has been modified by this scaffold.

The retained `sweep_layout_conv1d.py` and `sweep_lmid_conv1d.py` are legacy convolution tuning helpers, not Table 5 numerical evaluations. They can mutate layout source or CMake settings and are excluded from the unified runner. Their default CSV locations now use root `result/table5/` (or `$THUNDERFFT_RESULT_DIR`). Any historical CSVs must remain immutable reference data.
