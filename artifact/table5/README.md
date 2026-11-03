# Table 5 — numerical accuracy and dynamic range

**Status:** runnable numerical evaluation with validated CSVs and exported tables. Exact published input-population and FP32 provenance remain unestablished; those limits are recorded in every result.

The supplied PDF reports complex Gaussian inputs normalized per FFT so that `||x||₂ / sqrt(N) = alpha`. It measures relative L2 error against a double-precision FFT at `alpha=1`, and the largest input scale `alpha=2^k` for which every output remains finite, for `N=64,256,1024,4096`.

## Existing implementation and limits

`main.cu` implements the accuracy and finite-output scans. Its paths are:

| CSV precision | Actual implementation | Accuracy reference |
|---|---|---|
| `fp16` | ThunderFFT with FP16 input/output | cuFFT Z2Z FP64 |
| `tf32` | ThunderFFT float/TF32 path | cuFFT Z2Z FP64 |
| `fp32` | **cuFFT C2C, not ThunderFFT** | cuFFT Z2Z FP64 |

Inputs are generated in FP32; the double reference transforms those FP32 values, and FP16 error includes input conversion. A reduced finite-output scan is not an accuracy check. `--fast-finite-only` intentionally stores `nan` in error columns; `all_finite` is the quantity used for the dynamic-range result. Largest finite scale depends on the input population and seed.

The retained evaluator uses operational defaults **batch=65536, seed=1234**. The wrapper preserves those defaults for the `paper` profile, while metadata states that the PDF does not specify its exact sample population or seed. At `N=4096`, batch 65536 has 268,435,456 complex elements and substantial host/GPU allocation requirements (multiple GB per buffer and several live buffers); use a deliberate smaller batch for a bounded development run.

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

This measures the full size/exponent grid at batch 64; it does not establish the published population. `THUNDERFFT_TABLE5_N_LIST=64,1024` selects a documented subset, including for a `paper` profile; metadata records the exact list and whether it contains all four paper sizes. Supported sizes are `64,128,256,512,1024,2048,4096`; duplicate or invalid sizes are rejected. Batch must be a positive multiple of 16; `THUNDERFFT_TABLE5_SEED` defaults to `1234`. Every override is stored in metadata. Each of the five native invocations has a bounded `THUNDERFFT_TABLE5_TIMEOUT` (default 600 seconds). Increase that explicit limit for a large population if needed. No result from a timed-out invocation is accepted as a complete evaluation.

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
- `table.md`: readable exported table.
- `comparison.csv`: informational differences and error ratios against the retained source reference CSV. Non-paper sizes have blank reference columns; this comparison has no invented paper-acceptance threshold.
- `metadata.json`, `commands.txt`: population and overrides, seed, input definition, precision provenance, device selection, source commit/status and SHA-256 hashes, binary hash, CMake settings, compiler version, measured GPU/CUDA/cuFFT versions, build and exact native commands.
- `*.environment.json`: selected GPU, PCI bus identity, total global memory, compute capability, driver API/runtime/compile versions and cuFFT version from each native invocation.
- `validation.json`: row completeness, finite alpha=1 outputs, broad numerical sanity limits (`1e-2` for FP16/TF32, `1e-5` for cuFFT FP32), monotonicity of the scanned finite boundary, and consistent measured environment across invocations.

`run.sh` returns nonzero if those numerical sanity checks fail. These are broad implementation checks, not author-approved paper acceptance thresholds. Neither process completion nor these checks establish the exact published numbers or FP32 provenance. Metadata explicitly leaves scientific validation unassessed.

## Interpretation and remaining provenance

The evaluator is ready to execute and export its measurements without cuFFTDx tuning parameters or downloaded baseline repositories. CUDA Toolkit's bundled cuFFT supplies the FP32 path and FP64 reference with default planning; no cuFFT tuning data is required here. The runner does not adjust GPU clocks or power limits.

For interpreting the published table, the exact original batch, seed, hardware/runtime versions and FP32 implementation need to be reconciled with the author. The paper lists approximate FP16 errors `4.34e-4…6.54e-4`, TF32 errors `1.35e-3…2.64e-3`, FP32 errors `1.24e-7…1.85e-7`, and finite exponents FP16 `11,10,9,8` and TF32/FP32 `123,122,121,120` for the four increasing sizes. The unchanged `table5_accuracy_range_summary_with_fp32.csv` retains the source reference values used by `comparison.csv`; it is not a fresh result. Fresh outputs always live in root `result/`.

A scan's largest finite exponent is an observation over its recorded population. `right_censored=true` means all samples were still finite at the scan maximum, and `left_censored=true` means none were finite within the selected window. Neither condition establishes an unobserved boundary. A finite-output range does not assert small numerical error at every scale.

CPU-only summarizer failure checks are available with `python3 artifact/table5/check_summary.py`; they exercise complete data, missing/duplicate rows, excessive alpha=1 error, nonmonotonic overflow boundaries and inconsistent environments. No CUDA math is changed by this adapter.

The retained `sweep_layout_conv1d.py` and `sweep_lmid_conv1d.py` are legacy convolution tuning helpers, not Table 5 numerical evaluations. They can mutate layout source or CMake settings and are excluded from the unified runner. Their default CSV locations now use root `result/table5/` (or `$THUNDERFFT_RESULT_DIR`). Any historical CSVs must remain immutable reference data.
