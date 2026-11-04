# Run evaluations on Vast.ai

Run `./run.sh --backend vast` on the coordinator server. The coordinator searches
for a suitable GPU, checks the cost estimate, uploads a source snapshot, executes
the selected evaluations, retrieves their result bundles, and destroys the
instance. Results return to the coordinator's `result/` directory. Local execution
remains available with `--backend local`, which is the default.

The cloud backend changes where an evaluation runs. It does not complete pending
evaluation adapters or supply missing baseline libraries and tuning parameters.
See the [evaluation status](../README.md) for each task's implemented scope.

## Prepare the coordinator

Use a Linux coordinator with Python 3.9 or newer, Git, OpenSSH client tools and
outbound HTTPS/SSH access. Keep the coordinator running until result retrieval and
cleanup finish. Its OS account must be dedicated to the intended evaluators; users
who can read that account's files can also use its credentials.

Store the review's dedicated credentials outside the repository:

```text
~/cloud_key/                         # mode 0700
  thunderfft/                       # mode 0700
    config.json                     # mode 0600; paths and settings only
    vast_api_key                    # mode 0600; API token, one line
    id_ed25519                      # mode 0600; dedicated SSH private key
    id_ed25519.pub                   # matching SSH public key
```

The operator prepares this once before giving evaluators access:

```bash
umask 077
install -d -m 700 "$HOME/cloud_key" "$HOME/cloud_key/thunderfft"
install -m 600 artifact/cloud/config.example.json \
  "$HOME/cloud_key/thunderfft/config.json"
ssh-keygen -t ed25519 -N '' -C thunderfft-ae \
  -f "$HOME/cloud_key/thunderfft/id_ed25519"
# Paste the dedicated token as one line in the editor, then save and close it.
vi "$HOME/cloud_key/thunderfft/vast_api_key"
chmod 600 "$HOME/cloud_key/thunderfft/vast_api_key" \
  "$HOME/cloud_key/thunderfft/id_ed25519"
```

Do not overwrite existing review credentials when preparing another coordinator.
An operator-managed secret provisioning tool can replace the editor step. Do not
put the token's value in a shell command, repository, or result log.
The SSH key must support unattended authentication. The runner attaches its public
key to the rented instance; the private key and account API key stay on the
coordinator and are excluded from uploads. Credential paths must be outside the
repository, with private files readable only by their owner and a private parent
directory.

Create a separate, revocable Vast API key for this review. Required permission
categories are `misc` (offer search), `user_read` (account checks), `instance_read`,
`instance_write` (create, attach public key, destroy), and `billing_read`. Do not
grant key administration, credit transfers or payment administration to this key.
An operator can use this permission object when creating it:

```json
{
  "api": {
    "misc": {},
    "user_read": {},
    "instance_read": {},
    "instance_write": {},
    "billing_read": {}
  }
}
```

These categories grant account-level access within their scope; this is not a key
restricted to one future instance. A dedicated review account or operator-managed
isolation is preferable when the same account holds unrelated resources. Vast
supports scoped keys and parameter constraints, but a revocable review key is not
automatically single-use. Revoke it and remove the dedicated SSH key after the
review. The runner does not create credentials or alter account/payment settings.
See [Vast's permission reference](https://docs.vast.ai/api-reference/permissions).

## Preview and execute

Run commands from the repository root:

```bash
# Offline preview: no credentials, API calls, GPU work, or rental.
./run.sh validation --backend vast --gpu A100 --dry-run

# Read-only account/offer query and cost estimate; no rental.
./run.sh validation --backend vast --gpu A100 --cloud-plan

# Execute a bounded smoke job and retrieve its outputs.
./run.sh validation --backend vast --gpu A100 --profile smoke

# Use a different operator-prepared configuration.
./run.sh table3 --backend vast --gpu A100 --profile paper \
  --cloud-config "$HOME/cloud_key/thunderfft/config.json"
```

`--gpu` accepts `A100`, `H100` or `B200`. A100/H100 searches require the paper's
PCIe 80 GB models; SXM and H100 NVL offers are not interchangeable substitutes.
Availability and accepted cost determine whether a run can start. A plan does not
reserve its displayed offer. Execution checks current account and offer
information again. Each invocation rents at most
one GPU, then releases it. Choose the task selection and execution time together;
the full paper workload is not guaranteed to fit the example's ten-minute
execution allowance.

The default configuration path is `~/cloud_key/thunderfft/config.json`. The
[example configuration](config.example.json) supplies credential paths, timing
allowances, transfer estimates, image choices and an `environment` object for
supported measurement overrides. The runner forwards explicitly supported
evaluation settings, not arbitrary coordinator environment variables. Put
measurement overrides such as `THUNDERFFT_TABLE3_N_LIST` in `environment`; never
put credentials there.

Cloud runs require a clean tracked source tree by default. Commit intended source
changes before execution, or explicitly use `--cloud-allow-dirty` to include
modified tracked files and staged new files. Nonignored untracked files cause the
source check to fail even with that flag; track new evaluation code and store
local batch files outside the checkout or explicitly ignore them. The source
archive records the original Git revision and hashes of the actual uploaded
files. It excludes Git history, credential folders, build directories and
previous results.

## Collect several parts on the same GPU

Use `--cloud-batch` to run multiple collections inside one rental with one source
snapshot and shared build. For example, save the following outside the checkout
as `~/cloud_key/thunderfft/a100-parts.json`:

```json
[
  {
    "evaluations": ["table3"],
    "part": "a100-fp16",
    "environment": {"THUNDERFFT_TABLE3_PRECISIONS": "fp16"}
  },
  {
    "evaluations": ["table3"],
    "part": "a100-tf32",
    "environment": {"THUNDERFFT_TABLE3_PRECISIONS": "tf32"}
  }
]
```

```bash
./run.sh --backend vast --gpu A100 --profile paper \
  --collection ae-a100 \
  --cloud-batch "$HOME/cloud_key/thunderfft/a100-parts.json" --cloud-plan
./run.sh --backend vast --gpu A100 --profile paper \
  --collection ae-a100 \
  --cloud-batch "$HOME/cloud_key/thunderfft/a100-parts.json"

# Assemble returned CSVs locally; no rental or GPU execution.
./run.sh --collection ae-a100 --assemble
```

The jobs inherit the command's profile, collection, build parallelism and timeout.
Each job's environment overrides the configuration's measurement settings. The
execution allowance applies to the entire batch, including build work. Choose
unique part labels and nonoverlapping measurements. Missing inputs remain pending
when assembling; the assembler does not fill gaps with paper values.

The same GPU model does not establish compatible evidence: existing assemblers
check the physical GPU UUID, compiler/runtime, source and binary hashes, and
measurement settings. Collect parts requiring the same physical GPU during one
rental. A later rental may use a different A100 and therefore require a separate
collection. Table 3 can contain different GPU families in separate rows when their
source provenance agrees. Do not remove these checks to combine unrelated runs.

## Returned evidence

Each rental execution retains its own cloud session directory. The downloaded
`result/` tree preserves the remote runner's evaluation directories and evidence:

```text
result/
  cloud/thunderfft-ae-<session_id>/
    plan.json                       # selected offer and conservative cost plan
    request.json                    # selected jobs and measurement overrides
    source.json                     # Git revision and uploaded file hashes
    state.json                      # instance ID, deadline and cleanup state
    cleanup.json
    billing.json                    # observed charges; settlement unconfirmed
    status.json
    download/result/
      <evaluation>/<run_id>/        # CSVs, metadata, correctness and task logs
      runs/<run_id>/                # batch manifest and summary
      cloud-worker/                 # setup/job logs and measured environment
      collections/<name>/           # remote immutable parts, when requested
  collections/<name>/parts/         # automatically imported local parts
```

When `--collection` is supplied, the coordinator imports validated remote parts
into the local collection after its instance cleanup step. Existing parts cannot
be overwritten. It retains downloaded evidence if import fails, so a duplicate
label or incompatible collection does not erase the CSVs. Tables and figures are
then assembled explicitly with local `./run.sh --collection NAME --assemble`.
Without `--collection`, results remain under the cloud session's
`download/result/` tree and can be imported later using the existing
`--import-result EVALUATION=DIR` option.

After a failed or interrupted job, the coordinator attempts to retrieve available
logs and terminal checkpoints before cleanup. Such retrieval can fail if the
remote server or connection is already unavailable; partial data does not become
a successful measurement just because the transfer succeeded.

## Images and reproduction conditions

The example uses CUDA 12.9.1 on Ubuntu 22.04 for A100 and CUDA 13.0.2 on Ubuntu
24.04 for H100/B200. The A100 image's Ubuntu version differs from the paper's
Ubuntu 20.04: Ubuntu 22.04 supplies the Python and CMake versions needed by this
artifact runner. This environment difference must remain in measurement metadata.
An operator can supply an Ubuntu 20.04 image with compatible Python/CMake already
installed when reproducing that OS setting.

Setup installs build tools and the repository's Python requirements. Image tags
and distribution package repositories can change. For a fixed review environment,
prepare and validate an image with the required dependencies, pin its image digest
in `images`, and retain the compiler, package and GPU information in the returned
evidence. Installing a missing external baseline remains the responsibility of
that evaluation's adapter; a CUDA development image alone does not supply it.

## Cost, cleanup and recovery

The example plans for at most **US$1.50**, with a **US$0.25** uncertainty reserve.
The runner rejects budgets above **US$5 per invocation**, including startup,
execution, downloads, cleanup, retries, storage and network traffic. It adds a
360-second billing-delay allowance. `upload_gb` is an inbound traffic estimate
including image/package downloads, not just the source archive; `download_gb`
reserves result-transfer traffic. Adjust estimates conservatively for your image
and task. A job that cannot fit its estimate is rejected before renting.

An independent coordinator watchdog is armed before instance creation. Normal
cleanup destroys the instance and checks that it disappeared. An instance-side
deadline fallback uses Vast's own per-instance credential when available; the
coordinator's account API token is never sent to the GPU server. These safeguards
reduce orphaned rentals but are not a provider-enforced spending cap: API/network
outages can delay destruction. The runner does not change autobilling or payment
settings.

Vast charges storage while an instance is stopped, so stopping is insufficient.
After destruction, billing/account evidence is recorded separately from the
absence check. A stable observed debit does not prove that delayed charges have
finally settled. See [Vast's billing documentation](https://docs.vast.ai/guides/reference/billing)
and [per-instance credentials](https://docs.vast.ai/guides/instances/docker-environment).

Cloud session state and logs are retained under
`result/cloud/thunderfft-ae-<session_id>/`.
After a coordinator interruption, use the saved session directory:

```bash
./run.sh --backend vast --cloud-recover result/cloud/thunderfft-ae-SESSION_ID
```

Recovery rechecks and destroys the instance identified by that session and
records billing evidence. It does not create a new rental, resume jobs or retrieve
additional results. Destruction removes any data still held only on that
instance. Preserve the session record until cleanup and billing checks have been
reviewed.

CPU contract and simulated failure-path checks cover the backend. A real Table 2
CPU model run also passes through the worker, result archive extraction,
collection import and offline assembly; its GPU/version probes are simulated.
Live read-only account, offer, instance-list and billing API queries were checked
on 2026-11-04.

No rented-GPU end-to-end run has been performed for this backend. Provisioning,
remote CUDA execution and result retrieval still require a small live smoke run
to establish integration evidence separately from these CPU/API checks and the
existing local GPU validation.
