# Figure 3: FFT performance normalized by communication-only baseline

Measures the end-to-end time (global load, FFT, global store) of ThunderFFT,
cuFFT and cuFFTDx for N = 64, 256, 1024 and 4096 at batch 65,536, and divides
it into the time of a communication-only kernel that does the same global loads
and stores without FFT arithmetic (paper p. 10). Runs on the local A100 with
FP32 complex storage (ThunderFFT TF32, cuFFT and cuFFTDx FP32).

## Run

```bash
./run.sh figure3 --profile smoke
./run.sh figure3 --profile paper
```

To collect libraries or sizes in separate runs, add `--collection NAME --part LABEL`
(a new label per run) and then assemble offline:

```bash
THUNDERFFT_FIGURE3_LIBRARIES=copy,thunderfft ./run.sh figure3 --profile paper --collection fig3 --part copy-thunderfft
THUNDERFFT_FIGURE3_LIBRARIES=cufft,cufftdx ./run.sh figure3 --profile paper --collection fig3 --part cufft-cufftdx
./run.sh figure3 --assemble --collection fig3 --profile paper
```

## Options

| Environment variable | Values | Default (smoke / paper) |
| --- | --- | --- |
| `THUNDERFFT_FIGURE3_LIBRARIES` | `all` or comma list of `copy`, `copy-scalar`, `thunderfft`, `cufft`, `cufftdx` | all (cuFFTDx when MathDX is configured) |
| `THUNDERFFT_FIGURE3_N_LIST` | comma list of `64`, `256`, `1024`, `4096` | all four |
| `THUNDERFFT_FIGURE3_PRECISIONS` | comma list of `tf32`, `fp16` (`fp16` adds FP16 storage; cuFFT is FP32 only) | `tf32` |
| `THUNDERFFT_FIGURE3_BATCH` | positive multiple of 16; paper accepts only 65536 | 1024 / 65536 |
| `THUNDERFFT_FIGURE3_TRIALS` | positive integer (median over trials) | 2 / 3 |
| `THUNDERFFT_FIGURE3_REPEATS` | positive integer (in-kernel repeat count passed to the benchmark) | 16 / 16 |
| `THUNDERFFT_FIGURE3_TIMEOUT` | seconds per native benchmark call | 300 / 300 |

`copy` is the communication-only baseline; `copy-scalar` is an extra diagnostic
series and is not needed for the figure.

## Output

Each run writes to `result/figure3/<run_id>/`:

- `data.csv`: per-trial time, bytes and relative L2 error for each library, N and precision
- `summary.csv`: median time and normalized performance (communication-only median / library median)
- `baselines.csv`: the cuFFT and cuFFTDx rows of `data.csv`
- `figure.pdf`: the plot
- `metadata.json`: settings, GPU, source hashes, cuFFTDx candidate selection and missing series
- `commands.txt` and `raw/`: native commands and their stdout/stderr
- `configure.log`, `build.log`, `run.log`, `status.json`: build and runner logs

Assembly writes `data.csv`, `baselines.csv`, `summary.csv`, `figure.pdf` and
`metadata.json` to `result/collections/NAME/assembled/figure3/`.
