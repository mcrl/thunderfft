# ThunderFFT artifact

Evaluations for *xFFT: Accelerating FFT with Tensor Cores* (PPoPP'27). xFFT in
the paper is ThunderFFT in this repository.

| Evaluation | Paper | GPU | On a machine with an A100 |
| --- | --- | --- | --- |
| [validation](validation/README.md) | correctness | any | local |
| [table2](table2/README.md) | Table 2, p. 3 | CPU, A100 for the measured row | local |
| [table3](table3/README.md) | Table 3, p. 8 | A100, H100, B200 | local (A100), vast (H100, B200) |
| [table4](table4/README.md) | Table 4, p. 8 | A100, H100, B200 | local (A100), vast (H100, B200) |
| [table5](table5/README.md) | Table 5, p. 9 | A100 | local |
| [table6](table6/README.md) | Table 6, p. 9 | A100 | local |
| [table7](table7/README.md) | Table 7, p. 9 | A100 | local |
| [figure2](figure2/README.md) | Figure 2, p. 9 | A100 | local |
| [figure3](figure3/README.md) | Figure 3, p. 10 | A100 | local |
| [figure4](figure4/README.md) | Figure 4, p. 10 | A100 | local |
| [figure5](figure5/README.md) | Figure 5, p. 10 | H100 | vast |

Figure 1 and Table 1 have no evaluation.

## Requirements

- Linux x86_64 with an NVIDIA GPU and driver
- CUDA Toolkit 12.9 for A100 (`/usr/local/cuda-12.9`, `/usr/local/cuda` or `CUDA_HOME`)
- C++17 host compiler, Python 3.9 or newer with `venv`
- Internet access to NVIDIA, PyPI and GitHub
- For `--backend vast`: credentials in `~/cloud_key/thunderfft/` ([Vast.ai backend](cloud/README.md))

## Setup

Once per account, no root access needed:

```bash
unzip thunderfft-<revision>.zip
cd thunderfft
bash artifact/setup.sh
source ~/.thunderfft_env
bash artifact/check.sh
```

`setup.sh` installs under `$HOME`: a venv at `~/venv` (matplotlib, CMake, CPU
PyTorch headers), MathDX 25.06.1 and cuDNN 9.12. It writes `~/.thunderfft_env`,
which activates the venv and sets `CUDA_HOME`, `PATH`, `MATHDX_ROOT`,
`CUDNN_HOME`, `LD_LIBRARY_PATH` and `THUNDERFFT_ARCHS` (GPU 0's architecture),
and sources it from `~/.bashrc`. `CUDA_HOME`, `THUNDERFFT_ARCHS`,
`THUNDERFFT_VENV`, `THUNDERFFT_MATHDX_ARCHIVE` and `THUNDERFFT_CUDNN_ARCHIVE`
(local archive or directory) override the defaults. `check.sh` runs CPU-only
tests.

## Run locally

```bash
export CUDA_VISIBLE_DEVICES=0            # the GPU THUNDERFFT_ARCHS was detected for
./run.sh --list
./run.sh --jobs 8                        # default batch, smoke profile
./run.sh --profile paper --jobs 8        # default batch, paper profile
./run.sh table6 figure3 figure4 --profile paper --jobs 8
```

The default batch is validation, table2, table3, table4, table5, table7 and
figure2; `all` selects every evaluation. `smoke` runs small workloads that
check each path; `paper` uses the paper's settings. Other options: `--device N`,
`--timeout SECONDS`, `--result-dir DIR`, `--build-dir DIR`, `--dry-run`.

## Run on Vast.ai

GPUs the machine does not have are rented per command:

```bash
./run.sh figure5 --backend vast --gpu H100 --profile paper --cloud-plan        # offer and cost only
./run.sh figure5 --backend vast --gpu H100 --profile paper --collection fig5 --part h100
./run.sh figure5 --assemble --collection fig5 --profile paper
```

The worker installs the dependencies and sets `MATHDX_ROOT`, `CUDNN_HOME` and
`THUNDERFFT_ARCHS` itself. `--gpu` has no effect without `--backend vast`. See
[Vast.ai backend](cloud/README.md) for credentials and configuration.

## Collect per GPU and assemble

Runs with `--collection NAME --part PART` are stored as separate parts; `--assemble`
combines them into tables and figures without GPU work. Table 3 and Table 4 put
A100, H100 and B200 results into one table:

```bash
./run.sh table3 table4 --profile paper --collection ae --part a100
./run.sh table3 table4 --backend vast --gpu H100 --profile paper --collection ae --part h100
./run.sh table3 table4 --backend vast --gpu B200 --profile paper --collection ae --part b200
./run.sh table2 table3 table4 figure2 --assemble --collection ae --profile paper
```

One GPU's part must be collected in a single command, and all parts must come
from the same source. Use separate collections for smoke and paper runs.
Results from another account or machine are added with
`--import-result EVALUATION=DIR` (the complete result directory) and `--part`.

## Results

```text
result/<evaluation>/<run_id>/                  data.csv, table.csv or figure.pdf, logs, metadata.json
result/runs/<run_id>/summary.csv               status of each evaluation
result/collections/<name>/assembled/<evaluation>/
result/collections/<name>/assembled/missing.csv
```

`run.sh` returns 0 when every selected evaluation completed, 1 on a failure,
2 when something is pending or blocked, and 130 when interrupted.

## Troubleshooting

- `CUDA-capable device(s) is/are busy or unavailable`: the GPU is in
  exclusive-process mode and used by another process.
- CMake older than 3.18 on `PATH`: set `THUNDERFFT_CMAKE=/path/to/cmake`.
- Toolkit not found: set `CUDACXX=/path/to/nvcc` and `CUDAToolkit_ROOT=/path/to/cuda`.

## Packaging

```bash
python3 artifact/package.py              # dist/thunderfft-<revision>.zip
```
