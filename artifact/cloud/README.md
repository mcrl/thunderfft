# Vast.ai backend

`./run.sh ... --backend vast --gpu A100|H100|B200` rents one GPU on Vast.ai,
uploads the source, installs the dependencies the selected evaluations need
(MathDX, cuDNN 9.12, CPU PyTorch headers), runs the same adapters as a local
run, downloads the results and destroys the instance.

## Credentials

Each evaluator account holds its own credentials under `~/cloud_key/thunderfft/`.
They are never part of the repository or the distribution zip.

```text
~/cloud_key/                  mode 0700
  thunderfft/                 mode 0700
    vast_api_key              mode 0600, API key on one line
    id_ed25519                mode 0600, SSH private key without passphrase
    id_ed25519.pub
    config.json               optional, see Configuration
```

```bash
umask 077
install -d -m 700 ~/cloud_key ~/cloud_key/thunderfft
ssh-keygen -t ed25519 -N '' -C thunderfft-ae -f ~/cloud_key/thunderfft/id_ed25519
vi ~/cloud_key/thunderfft/vast_api_key
chmod 600 ~/cloud_key/thunderfft/vast_api_key
```

The Vast API key needs the permission categories `misc`, `user_read`,
`instance_read`, `instance_write` and `billing_read`.

## Usage

```bash
./run.sh figure5 --backend vast --gpu H100 --profile smoke --dry-run       # offline preview
./run.sh figure5 --backend vast --gpu H100 --profile smoke --cloud-plan    # offer and cost, no rental
./run.sh figure5 --backend vast --gpu H100 --profile smoke                 # rent, run, download, destroy
./run.sh figure5 --backend vast --gpu H100 --profile paper --collection fig5 --part h100
```

A100 and H100 offers are restricted to the PCIe 80GB models. Add
`--collection NAME --part PART` to import the downloaded result into the local
collection, then assemble it locally with `./run.sh --collection NAME --assemble`.

The uploaded source is the tracked file list of a Git checkout, or the packaged
file list of an extracted distribution zip. Edited files are rejected unless
`--cloud-allow-dirty` is given; extra files outside `result/`, `build/` and
`build_*/` are rejected.

## Configuration

The default configuration is `~/cloud_key/thunderfft/config.json`; `--cloud-config`
selects another file. A file lists only the fields it changes
([`config.example.json`](config.example.json) shows all of them).

| Field | Default | Meaning |
| --- | --- | --- |
| `execution_seconds` | 3600 | time allowed on the instance for build and run |
| `budget_usd` | 10 | maximum planned cost of one invocation |
| `disk_gb` | 32 | instance disk |
| `images` | CUDA 12.9.1 (A100), 13.0.2 (H100, B200) devel images | container image per GPU |
| `environment` | `{}` | measurement overrides forwarded to the job |
| `mathdx` | `true` | install MathDX for cuFFTDx paths |

The worker sets `MATHDX_ROOT`, `CUDNN_HOME` and `THUNDERFFT_ARCHS` for the rented
GPU and uses device 0. The `THUNDERFFT_*` measurement variables listed in
`ENVIRONMENT` in [`config.py`](config.py) are forwarded from the shell and from
`environment` (which takes precedence). Local paths such as `*_BACKENDS` and
`THUNDERFFT_TABLE4_SOURCE_DIR` are not forwarded.

## Several parts in one rental

```json
[
  {"evaluations": ["table3"], "part": "h100-fp16", "environment": {"THUNDERFFT_TABLE3_PRECISIONS": "fp16"}},
  {"evaluations": ["table3"], "part": "h100-tf32", "environment": {"THUNDERFFT_TABLE3_PRECISIONS": "tf32"}}
]
```

```bash
./run.sh --backend vast --gpu H100 --profile paper --collection ae --cloud-batch ~/h100-parts.json
```

All jobs share one instance, source upload and build. Keep the batch file
outside the source directory.

## Results

```text
result/cloud/thunderfft-ae-<session>/
  plan.json request.json source.json state.json status.json billing.json
  download/result/<evaluation>/<run_id>/     CSVs, logs, metadata
  download/result/cloud-worker/              setup and job logs
```

If the coordinator is interrupted, destroy the instance of a saved session with:

```bash
./run.sh --backend vast --cloud-recover result/cloud/thunderfft-ae-<session>
```
