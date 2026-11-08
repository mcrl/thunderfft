# Table 6: Comparison with prior Tensor Core FFT libraries

Measures 256-point FP16 forward FFT time (batch 65536) for cuFFTDx, tcFFT,
FlashFFTConv and ThunderFFT, and the speedup over cuFFTDx (paper p. 9), on an A100.
Each time is `(t_2R - t_R) / R`, median over trials, after a check against an
FP64 DFT.

tcFFT and FlashFFTConv are not included in the artifact. When selected, they are
cloned from GitHub at the commits below into `build_artifact/artifact_table6/`
during the run, so network access is required:

| Library | Repository | Commit |
|---|---|---|
| tcFFT | https://github.com/rox906/tcFFT | `3e87a7083577741815e21b2a3c2bed22520786ce` |
| FlashFFTConv | https://github.com/HazyResearch/flash-fft-conv | `b8771028717f46d5b22cbb8e12833f35033d621b` |

## Run

Run locally on an A100 80GB PCIe GPU.

Table 6 is not in the default `./run.sh` set; name it explicitly.

```bash
./run.sh table6                    # smoke
./run.sh table6 --profile paper    # paper
```

Libraries can be collected in separate parts and assembled without a GPU:

```bash
THUNDERFFT_TABLE6_LIBRARIES=thunderfft,cufftdx \
  ./run.sh table6 --profile paper --collection t6 --part native
THUNDERFFT_TABLE6_LIBRARIES=tcfft,flashfftconv \
  ./run.sh table6 --profile paper --collection t6 --part external
./run.sh table6 --assemble --collection t6 --profile paper
```

Parts must come from the same GPU and use the same settings. Libraries not yet
collected stay blank; speedups stay blank until cuFFTDx is collected.

## Options

| Environment variable | Values | Default (smoke / paper) |
|---|---|---|
| `THUNDERFFT_TABLE6_LIBRARIES` | `all`, or comma-separated subset of `cufftdx`, `tcfft`, `flashfftconv`, `thunderfft` | `all` |
| `THUNDERFFT_TABLE6_BATCH` | positive multiple of 512 (paper requires `65536`) | `4096` / `65536` |
| `THUNDERFFT_TABLE6_REPEATS` | R, positive integer (paper requires `1000`) | `64` / `1000` |
| `THUNDERFFT_TABLE6_TRIALS` | positive integer | `2` / `3` |
| `THUNDERFFT_TABLE6_TIMEOUT` | seconds per library | `600` |
| `FLASHFFTCONV_PYTHON` | optional: Python with PyTorch headers for the FlashFFTConv build | runner Python (set up by `setup.sh`) |

## Output

Written to `result/table6/<run_id>/`:

- `table.csv`: per-library median time, speedup over cuFFTDx, and paper reference values
- `data.csv`: every trial
- `raw/<library>.csv`, `raw/<library>.log`: native driver output and device log
- `metadata.json`: settings, pinned sources and commands
- `status.json`, `run.log`: runner status and console log (including build output)

Assembly writes `table.csv`, `data.csv` and `metadata.json` to
`result/collections/<name>/assembled/table6/`.
