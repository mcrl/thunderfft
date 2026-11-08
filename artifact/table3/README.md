# Table 3: FFT kernel latency

FFT kernel latency of ThunderFFT and cuFFTDx on paper p. 8: sizes 64 to 4096,
batch 65536, register and shared-memory input, ThunderFFT FP16 and TF32 against
cuFFTDx FP16 and FP32, with speedups. The paper reports A100 PCIe 80GB, H100
PCIe and B200 columns; one run measures one GPU.

## Run

```sh
# Local GPU
./run.sh table3                    # smoke: batch 1024, R=256, 2 trials
./run.sh table3 --profile paper    # paper: batch 65536, R=1000, 3 trials

# H100 or B200 on Vast.ai (add --cloud-plan to check offer and cost without renting)
./run.sh table3 --backend vast --gpu H100
./run.sh table3 --backend vast --gpu H100 --profile paper
./run.sh table3 --backend vast --gpu B200 --profile paper
```

cuFFTDx is measured when the build has MathDX, which `artifact/setup.sh` and
the Vast.ai worker both provide.

## Collect per GPU and assemble

Collect each GPU column in one command, each GPU as a different part of the
same collection, then assemble without a GPU:

```sh
./run.sh table3 --profile paper --collection ae --part a100    # local A100
./run.sh table3 --backend vast --gpu H100 --profile paper --collection ae --part h100
./run.sh table3 --backend vast --gpu B200 --profile paper --collection ae --part b200
./run.sh table3 table2 figure2 --assemble --collection ae --profile paper
```

Use `table3 table4` instead of `table3` in the three collection commands to
also collect the [Table 4](../table4/README.md) column from the same GPU.
Configurations that were not collected stay blank; assembly exits with code 2
while the table is partial.

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_TABLE3_N_LIST` | comma-separated subset of `64,128,256,512,1024,2048,4096` | all |
| `THUNDERFFT_TABLE3_PRECISIONS` | `fp16`, `tf32` (`tf32` selects cuFFTDx FP32) | both |
| `THUNDERFFT_TABLE3_PLACEMENTS` | `register`, `shared` | both |
| `THUNDERFFT_TABLE3_LIBRARIES` | `ThunderFFT`, `cuFFTDx` | both if the build has MathDX, else `ThunderFFT` |
| `THUNDERFFT_TABLE3_BATCH` | positive multiple of 16 (512 with cuFFTDx); paper requires 65536 | 1024 / 65536 |
| `THUNDERFFT_TABLE3_REPEATS` | FFT repeats R per kernel; paper requires 1000 | 256 / 1000 |
| `THUNDERFFT_TABLE3_TRIALS` | positive integer | 2 / 3 |
| `THUNDERFFT_TABLE3_TIMEOUT` | seconds per ThunderFFT executable call | 120 |
| `THUNDERFFT_TABLE3_CUFFTDX_TIMEOUT` | seconds per cuFFTDx executable call | 1800 |

These variables are forwarded to Vast.ai when exported in the shell.

## Output

Written to `result/table3/<run_id>/`:

- `data.csv`: every validated timing trial (ThunderFFT and every cuFFTDx candidate)
- `table.csv`: median/min/max latency per configuration; paper targets on paper-profile runs
- `cufftdx_candidates.csv`: every cuFFTDx candidate, selected one marked (cuFFTDx runs only)
- `speedup.csv`: cuFFTDx/ThunderFFT speedups and geometric means (cuFFTDx runs only)
- `metadata.json`: settings, coverage, GPU and software versions, hashes
- `raw/*.csv`, `raw/*.log`: native output of each executable call
- `commands.txt`, `configure.log`, `build.log`: exact commands and build logs
- `status.json`, `run.log`: runner status and full task log

A Vast.ai run downloads the same directory to
`result/cloud/thunderfft-ae-<session>/download/result/table3/<run_id>/`. With
`--collection`, the result is also stored as
`result/collections/ae/parts/table3/<part>/`.

Assembly writes `result/collections/ae/assembled/table3/` (`data.csv`,
`table.csv`, `cufftdx_candidates.csv`, `speedup.csv`, `metadata.json`,
`status.json`) and `result/collections/ae/assembled/missing.csv`, listing
uncollected configurations.
