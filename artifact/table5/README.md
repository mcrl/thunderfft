# Table 5: Numerical accuracy and finite dynamic range

Measures relative L2 error against a cuFFT FP64 reference at `alpha=1` and scans
input scale `alpha=2^k` for the largest exponent with finite output (paper p. 9),
for ThunderFFT FP16/TF32, cuFFT FP32 and cuFFTDx FP16, on an A100.
`smoke` uses N=64, batch 64 and narrow exponent windows (FP16 `8..13`,
TF32/FP32 `120..127`); `paper` uses N=64/256/1024/4096, batch 65536 and
exponents `0..127`.

## Run

```bash
./run.sh table5                    # smoke
./run.sh table5 --profile paper    # paper

# Without a local A100
./run.sh table5 --backend vast --gpu A100
./run.sh table5 --profile paper --backend vast --gpu A100
```

The paper profile can be collected in parts (by phase or size) and assembled
without a GPU:

```bash
THUNDERFFT_TABLE5_PHASES=accuracy_reduced,accuracy_fp32_cufft,accuracy_fp16_cufftdx \
  ./run.sh table5 --profile paper --collection t5 --part accuracy
THUNDERFFT_TABLE5_PHASES=range_fp16,range_tf32,range_fp32_cufft \
  ./run.sh table5 --profile paper --collection t5 --part range
./run.sh table5 --assemble --collection t5 --profile paper
```

Use the same batch and seed in every part. Cells that were not collected stay
blank.

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE5_PHASES` | comma-separated subset of `accuracy_reduced`, `accuracy_fp32_cufft`, `range_fp16`, `range_tf32`, `range_fp32_cufft`, `accuracy_fp16_cufftdx` | all six (`accuracy_fp16_cufftdx` only when MathDX is configured) |
| `THUNDERFFT_TABLE5_N_LIST` | comma-separated subset of `64,128,256,512,1024,2048,4096` | `64` / `64,256,1024,4096` |
| `THUNDERFFT_TABLE5_BATCH` | positive multiple of 16 | `64` / `65536` |
| `THUNDERFFT_TABLE5_SEED` | nonnegative integer | `1234` |
| `THUNDERFFT_TABLE5_TIMEOUT` | seconds per phase | `600` |

## Output

Written to `result/table5/<run_id>/`:

- `table.csv` (same content as `table5_summary.csv`) and `table.md`: per-size, per-precision error and largest finite / first non-finite exponent
- `data.csv`: all raw rows combined
- `accuracy_reduced.csv`, `accuracy_fp32_cufft.csv`, `accuracy_fp16_cufftdx.csv`: error at `alpha=1` (FP16+TF32, cuFFT FP32, cuFFTDx FP16)
- `range_fp16.csv`, `range_tf32.csv`, `range_fp32_cufft.csv`: per-exponent finite-output scans
- `<phase>.log`, `<phase>.environment.json`: native output and GPU/CUDA/cuFFT identity per phase
- `comparison.csv`: differences from the reference values in `artifact/table5/table5_accuracy_range_summary_with_fp32.csv`
- `validation.json`: completeness and numerical sanity checks
- `metadata.json`, `commands.txt`: settings, build and native commands
- `status.json`, `run.log`: runner status and console log

Assembly writes `table.csv`, `table5_summary.csv`, `table.md`, `data.csv`, `comparison.csv`,
`validation.json` and `metadata.json` to `result/collections/<name>/assembled/table5/`;
missing cells are listed in `result/collections/<name>/assembled/missing.csv`.
