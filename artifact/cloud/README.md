# Vast.ai backend

`./run.sh ... --backend vast --gpu A100|H100|B200` rents one GPU on Vast.ai,
uploads the source, installs the dependencies the selected evaluations need
(MathDX, cuDNN 9.12, CPU PyTorch headers), runs the same adapters as a local
run, downloads the results and destroys the instance.

The default `./run.sh` (also `./run.sh all`) coordinates the whole paper
workflow: all A100 evaluations and H100/B200 Tables 3 and 4 run in parallel,
followed by final assembly. It uses the local A100 when available and rents one if it remains
unavailable after `--local-wait-seconds 120`. Set `--a100-backend local` to
require the local A100, or `--a100-backend vast` to rent one directly. It rents
all required cloud GPUs before starting measurements and manages one total
cloud budget across them. First it checks every required offer and their
combined conservative cost. It then acquires all cloud servers, prepares them
in parallel, and waits for every server to be ready. The local A100 is rechecked
at this gate; only then do all GPU measurement stages start together.
Each cloud result is downloaded and its instance destroyed after its stage;
final assembly waits for all GPU stages to finish.

Use `./run.sh --cloud-plan` to inspect the required offers and combined cost
before running; this does not rent an instance or run local measurements.
The plan includes startup, readiness waiting, execution, download, cleanup and
billing-delay allowances across all concurrent rentals.

If any acquisition or preparation fails, the whole workflow cleans up every
acquired instance and no benchmarks start. There is no automatic startup retry
for this parallel group. Individual `--backend vast` commands still use one
GPU at a time, with up to two startup attempts by default.

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
./run.sh table3 --backend vast --gpu H100 --profile smoke --dry-run       # offline preview
./run.sh table3 --backend vast --gpu H100 --profile smoke --cloud-plan    # offer and cost, no rental
./run.sh table3 --backend vast --gpu H100 --profile smoke                 # rent, run, download, destroy
./run.sh table3 --backend vast --gpu H100 --profile paper --collection ae --part h100
```

A100 and H100 offers are restricted to the PCIe 80GB models. Add
`--collection NAME --part PART` to import the downloaded result into the local
collection, then assemble it locally with `./run.sh --collection NAME --assemble`.
The complete workflow creates these parts and assembles them automatically.

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
| `budget_usd` | 50 | maximum planned cost of one invocation; total of all rentals for the complete workflow |
| `disk_gb` | 32 | instance disk |
| `images` | CUDA 12.9.1 (A100), 13.0.2 (H100, B200) devel images | container image per GPU |
| `environment` | `{}` | measurement overrides forwarded to the job |
| `mathdx` | `true` | install MathDX for cuFFTDx paths |
| `exclude_machine_ids` | `[]` | Vast host machine IDs to skip |
| `max_startup_attempts` | 2 | maximum startup attempts for standalone single-GPU runs, within the same invocation budget |
| `ssh_auth_failure_seconds` | 90 | duration of repeated SSH authentication rejection before abandoning that host |

For the complete workflow, H100 and B200 build/run stages default to 1,200
seconds each and a rented A100 stage to 3,600 seconds. Override them with
`--cloud-stage-seconds` or `--a100-stage-seconds`; `execution_seconds` in the
configuration caps both. These limits do not consume the separate startup,
download, cleanup and delayed-billing allowances in the cost plan.

When a rental cannot reach authenticated SSH, the coordinator destroys it
and confirms its absence before considering another host. The failed machine
is excluded from the retry. All attempts retain their cost and cleanup
evidence, and their conservative costs share the same budget. No retry occurs
if deletion cannot be confirmed or the remaining budget is insufficient.
This retry behavior applies to individual `--backend vast` invocations; the
complete parallel workflow instead cleans up the whole group on startup failure.

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

The complete parallel workflow records its group plan, readiness gate and
per-instance evidence under:

```text
result/cloud/group-<session>/
  preflight.json plan.json source.json ready.json execution.json
  thunderfft-ae-<instance-session>/
    plan.json request.json source.json state.json status.json billing.json
    download/result/<evaluation>/<run_id>/
    download/result/cloud-worker/
```

`ready.json` is written only after every cloud environment is ready. The
pipeline summary at `result/pipelines/<run_id>/pipeline.json` records the group
audit directory. Standalone `--backend vast` invocations use:

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

For a parallel group, use the individual session paths printed by the runner;
recovery applies to one instance at a time:

```bash
./run.sh --backend vast --cloud-recover result/cloud/group-<session>/thunderfft-ae-<instance-session>
```

Repeat for each group member whose cleanup remains unconfirmed. Recovery also
refreshes billing evidence; deletion alone does not establish settled charges.

## Unavailable cloud GPUs

We are sorry for the interruption when no compatible offer is available.
Provider GPU inventory is outside our control. The initial check reports the
unavailable GPU before any rental or benchmark starts and prints brief commands
to retry or proceed locally. It does not automatically wait for inventory or
skip the missing GPU.

```bash
./run.sh --cloud-plan                            # check offers and combined cost; no rental
./run.sh                                        # retry the complete workflow when available
./run.sh all --backend local --profile paper     # alternative: run only the local A100
```

Add `--full` if the original run used the exhaustive settings. Local-only
execution generates Figure 2–5 outputs; Tables 3/4 lack H100/B200 measurements.
For independent GPU parts and offline assembly, see
[Collect per GPU and assemble](../README.md#collect-per-gpu-and-assemble).
