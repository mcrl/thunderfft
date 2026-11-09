"""Budgeted AE jobs with bounded startup retries and immutable evidence."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid

from .config import (ARCHS, MATHDX_ARCHIVES, CUDNN_ARCHIVES, TORCH_HEADERS_VERSION,
                     CUFFTDX_MODE_ENV, cufftdx_mode, CUDNN_MODE_ENV, cudnn_mode,
                     TABLE5_RANGE_MODE_ENV, table5_range_mode,
                     inherited_environment, load_config, measurement_environment, required_dependencies)
from .provider import VastClient, ProviderError, CreateRejected, plan_cost
from .transport import SSH, build_source_bundle, safe_extract_result, validate_private_key
from .watchdog import destroy_owned, write_json
from ..progress import validate_snapshot


def add_arguments(parser):
    parser.add_argument("--backend", choices=("local", "vast", "all"), default=None,
                        help="execution backend; no evaluations or 'all' defaults to the complete local/cloud pipeline")
    parser.add_argument("--gpu", choices=tuple(ARCHS), default="A100", help="Vast GPU SKU: PCIe 80GB A100/H100 or B200")
    parser.add_argument("--cloud-config", type=Path, help="non-secret JSON settings; default ~/cloud_key/thunderfft/config.json")
    parser.add_argument("--cloud-plan", action="store_true", help="read-only account/offer/cost check; never create an instance")
    parser.add_argument("--cloud-batch", type=Path, help="JSON job list, sharing one rental/build; use --collection for named parts")
    parser.add_argument("--cloud-allow-dirty", action="store_true", help="include and hash tracked local edits; untracked source is rejected")
    parser.add_argument("--cloud-recover", type=Path, metavar="SESSION_DIR", help="destroy this saved session's rental and refresh billing; never rent again")


def make_jobs(args, entries, registry, config, select_tasks, parser):
    environment = inherited_environment(config)

    def job(evaluations, part, overrides=None):
        effective = dict(environment, **measurement_environment({} if overrides is None else overrides))
        mode = cufftdx_mode(getattr(args, "cufftdx_mode", None), effective)
        convolution_mode = cudnn_mode(getattr(args, "cudnn_mode", None), effective)
        range_mode = table5_range_mode(getattr(args, "table5_range_mode", None), effective)
        effective[CUFFTDX_MODE_ENV] = mode
        effective[CUDNN_MODE_ENV] = convolution_mode
        effective[TABLE5_RANGE_MODE_ENV] = range_mode
        return {"evaluations": evaluations, "part": part, "environment": effective,
                "cufftdx_mode": mode, "cudnn_mode": convolution_mode, "table5_range_mode": range_mode}

    if not args.cloud_batch:
        return [job([e["id"] for e in entries], args.part)]
    if args.evaluations or args.part:
        raise ValueError("--cloud-batch supplies evaluations and part names; omit positional evaluations and --part")
    supplied = json.loads(args.cloud_batch.read_text())
    if not isinstance(supplied, list) or not 1 <= len(supplied) <= 100:
        raise ValueError("cloud batch must contain 1–100 jobs")
    jobs, labels = [], set()
    for item in supplied:
        if not isinstance(item, dict) or set(item) - {"evaluations", "part", "environment"}:
            raise ValueError("batch jobs support evaluations, part and environment only")
        requested = item.get("evaluations")
        if not isinstance(requested, list) or not requested or any(not isinstance(e, str) for e in requested):
            raise ValueError("each batch job requires a nonempty evaluation list")
        selected = select_tasks(requested, registry["evaluations"], parser)
        if args.collection:
            selected = [e for e in selected if e["id"] != "figure2"]
        label = item.get("part")
        from .. import collection
        collection.name(label)
        if label in labels:
            raise ValueError("batch part labels must be unique")
        labels.add(label)
        jobs.append(job([e["id"] for e in selected], label, item.get("environment", {})))
    return jobs


class OfferSelectionUnavailable(ProviderError):
    """No eligible offer, with separate availability, price and balance evidence."""

    def __init__(self, gpu, diagnostics):
        self.diagnostics = diagnostics
        count = diagnostics["matching_offers"]
        if not count:
            self.reason = "no_matching_offers"
            detail = "no matching hardware offers were returned"
        elif diagnostics["excluded_host_offers"] == count:
            self.reason = "excluded_hosts_only"
            detail = "all matching offers are on excluded/unavailable hosts"
        else:
            self.reason = "no_eligible_offers"
            detail = "no matching offer passed the eligibility checks"
        counts = (f"matching={count}, excluded hosts={diagnostics['excluded_host_offers']}, "
                  f"invalid host IDs={diagnostics['invalid_host_ids']}, "
                  f"rejected cost plans={diagnostics['rejected_cost_plans']}, "
                  f"above available balance={diagnostics['above_available_balance']}")
        message = (f"No {gpu} offer is currently selectable: {detail} ({counts}; "
                   f"allocation USD {diagnostics['budget_usd']:.2f}, "
                   f"available balance USD {diagnostics['balance_usd']:.2f})")
        if diagnostics["cost_plan_errors"]:
            message += "; cost-plan details: " + "; ".join(diagnostics["cost_plan_errors"])
        if diagnostics["lowest_over_balance_plan_usd"] is not None:
            message += (f"; lowest otherwise eligible all-in plan USD "
                        f"{diagnostics['lowest_over_balance_plan_usd']:.4f}")
        super().__init__(message)


def plan(client, config, gpu):
    account = client.account_summary()
    client.billing_preflight()
    candidates = []
    diagnostics = {"matching_offers": 0, "excluded_host_offers": 0,
                   "invalid_host_ids": 0, "rejected_cost_plans": 0,
                   "above_available_balance": 0, "cost_plan_errors": [],
                   "lowest_over_balance_plan_usd": None,
                   "budget_usd": config["budget_usd"], "balance_usd": account["balance_usd"]}
    keys = ("disk_gb", "startup_seconds", "execution_seconds", "download_seconds", "cleanup_seconds",
            "billing_delay_seconds", "upload_gb", "download_gb", "budget_usd", "reserve_usd")
    for offer in client.search_offers(gpu, config["disk_gb"]):
        diagnostics["matching_offers"] += 1
        # A known machine ID is required to exclude failed hosts on retry.
        if (not isinstance(offer.get("machine_id"), int) or isinstance(offer["machine_id"], bool)
                or offer["machine_id"] <= 0):
            diagnostics["invalid_host_ids"] += 1
            continue
        if offer["machine_id"] in config["exclude_machine_ids"]:
            diagnostics["excluded_host_offers"] += 1
            continue
        try:
            cost = plan_cost(offer, **{k: config[k] for k in keys})
        except ProviderError as error:
            diagnostics["rejected_cost_plans"] += 1
            # plan_cost emits only locally generated price/configuration errors,
            # never a provider response body or credentials.
            detail = str(error)[:256]
            if detail not in diagnostics["cost_plan_errors"] and len(diagnostics["cost_plan_errors"]) < 3:
                diagnostics["cost_plan_errors"].append(detail)
            continue
        if account["balance_usd"] >= cost["planned_total_usd"]:
            candidates.append((cost["planned_total_usd"], offer, cost))
        else:
            diagnostics["above_available_balance"] += 1
            lowest = diagnostics["lowest_over_balance_plan_usd"]
            diagnostics["lowest_over_balance_plan_usd"] = min(
                cost["planned_total_usd"], lowest if lowest is not None else cost["planned_total_usd"])
    if not candidates:
        raise OfferSelectionUnavailable(gpu, diagnostics)
    _, offer, cost = min(candidates, key=lambda value: value[0])
    # Retain prices and identifying hardware fields, not account PII/provider
    # credentials or the potentially extensive raw offer response.
    selected = {k: offer.get(k) for k in ("id", "machine_id", "gpu_name", "gpu_ram", "num_gpus",
                "dph_total", "storage_cost", "inet_down_cost", "inet_up_cost", "cpu_name", "cpu_arch",
                "cuda_max_good", "reliability", "geolocation")}
    return {"offer": selected, "cost": cost, "account_before": account,
            "image": config["images"][gpu], "gpu": gpu, "planned_at": time.time()}


def onstart_script(label, deadline):
    # Vast's container-scoped credential stays inside the container and is only
    # read by this shell. Neither API nor private SSH coordinator keys are sent.
    # curl reads authorization from stdin, never from a command-line argument.
    return r'''umask 077
mkdir -p /tmp/thunderfft-watchdog
if command -v curl >/dev/null && [ -n "${CONTAINER_API_KEY:-}" ]; then
  thff_container_id="${CONTAINER_ID:-${VAST_CONTAINERLABEL#C.}}"
  case "$thff_container_id" in ''|*[!0-9]*) echo unsupported > /tmp/thunderfft-watchdog/status ;;
  *)
    (
      thff_delay=$((DEADLINE - $(date +%s)))
      if [ "$thff_delay" -gt 0 ]; then sleep "$thff_delay"; fi
      while :; do
        thff_code=$(printf 'header = "Authorization: Bearer %s"\n' "$CONTAINER_API_KEY" |
          curl --config - --silent --output /dev/null --write-out '%{http_code}' --max-time 15 \
          --request DELETE "https://console.vast.ai/api/v0/instances/${thff_container_id}/")
        if [ "$thff_code" = 404 ]; then exit 0; fi
        sleep 5
      done
    ) </dev/null >/tmp/thunderfft-watchdog/cleanup.log 2>&1 &
    echo armed > /tmp/thunderfft-watchdog/status ;;
  esac
else echo unsupported > /tmp/thunderfft-watchdog/status
fi
'''.replace("DEADLINE", str(int(deadline)))


def arm_watchdog(folder, lock):
    with (folder / "watchdog.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "-m", "artifact.cloud.watchdog", str(folder)],
            cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, pass_fds=(lock.fileno(),))
    until = time.monotonic() + 10
    while time.monotonic() < until:
        if (folder / "watchdog-ready.json").exists():
            return process
        if process.poll() is not None:
            break
        time.sleep(.1)
    raise RuntimeError("Independent watchdog did not acknowledge readiness; no rental created")


def remaining(deadline, maximum=30):
    left = deadline - time.time()
    if left <= 0:
        raise TimeoutError("Cloud execution phase exceeded its deadline")
    return max(.1, min(left, maximum))


def acquire_instance(client, config, gpu, state, folder, proposal, *,
                     cancel_event=None, log=None, validate_plan=None, acquisition_deadline=None):
    """Replace only definitively unavailable offers, within the original plan.

    The same label, watchdog, startup deadline and budget cover every rejected
    request. An ambiguous create is never retried or replaced here.
    """
    attempts = []
    excluded = set(config["exclude_machine_ids"])
    startup_deadline = state["created_at"] + config["startup_seconds"]
    if acquisition_deadline is not None:
        startup_deadline = min(startup_deadline, acquisition_deadline)

    def checkpoint():
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cloud group acquisition cancelled")
        remaining(startup_deadline)

    for index in range(3):
        checkpoint()
        attempt = {"attempt": index + 1, "plan": dict(proposal),
                   "started_at": time.time(), "status": "requesting"}
        attempts.append(attempt)
        write_json(folder / "creation-attempts.json", attempts)
        state["create_attempted"] = True
        write_json(folder / "state.json", state)
        try:
            state["instance_id"] = client.create_instance(proposal["offer"]["id"],
                image=proposal["image"], disk_gb=config["disk_gb"], label=state["label"],
                onstart=onstart_script(state["label"], state["deadline"]))
        except CreateRejected as error:
            # Only an acknowledged rejection permits another offer. Keep the
            # watchdog armed but record that this request created no rental.
            state["create_attempted"] = False
            write_json(folder / "state.json", state)
            attempt.update(status="rejected", error=str(error),
                           http_status=error.code, reason=error.reason)
            write_json(folder / "creation-attempts.json", attempts)
            if not error.retryable_offer or index == 2:
                raise
            checkpoint()
            excluded.add(proposal["offer"]["machine_id"])
            retry_config = dict(config, exclude_machine_ids=sorted(excluded))
            if log is not None:
                log(f"[{gpu}] selected offer is no longer available; searching another host "
                    f"within the same budget and startup deadline ({index + 2}/3)")
            try:
                replacement = plan(client, retry_config, gpu)
            except ProviderError as replacement_error:
                evidence = {"error_type": type(replacement_error).__name__,
                            "error": str(replacement_error),
                            "previous_create_rejection": {
                                "offer_id": proposal["offer"]["id"],
                                "machine_id": proposal["offer"]["machine_id"],
                                "planned_total_usd": proposal["cost"]["planned_total_usd"],
                                "budget_usd": config["budget_usd"],
                                "balance_usd": proposal["account_before"]["balance_usd"],
                                "http_status": error.code, "reason": error.reason,
                                "error": str(error)}}
                if isinstance(replacement_error, OfferSelectionUnavailable):
                    evidence.update(reason=replacement_error.reason,
                                    diagnostics=replacement_error.diagnostics)
                write_json(folder / "replacement-error.json", evidence)
                raise ProviderError(
                    f"Vast.ai rejected offer {proposal['offer']['id']} on host "
                    f"{proposal['offer']['machine_id']}: {error}; "
                    f"its all-in plan USD {proposal['cost']['planned_total_usd']:.4f} "
                    f"was within allocation USD {config['budget_usd']:.2f}; "
                    f"replacement search failed: {replacement_error}. "
                    "No replacement rental was created. Wait for another eligible offer and retry.") from None
            checkpoint()
            if validate_plan is not None:
                validate_plan(replacement)
            proposal.clear()
            proposal.update(replacement)
            write_json(folder / "plan.json", proposal)
            continue
        except BaseException as error:
            attempt.update(status="outcome_unknown", error_type=type(error).__name__)
            write_json(folder / "creation-attempts.json", attempts)
            raise
        # Save the ID before any other operation, including optional logging.
        write_json(folder / "state.json", state)
        attempt.update(status="created", instance_id=state["instance_id"])
        write_json(folder / "creation-attempts.json", attempts)
        return


class StartupUnavailable(TimeoutError):
    """A rental never reached authenticated SSH; safe to retry after cleanup."""


def wait_ssh(client, state, config, folder, root, deadline, cancel_event=None):
    first_auth_failure = None
    attempts = 0
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cloud group startup cancelled")
        for row in client.instances():
            if int(row["id"]) != state["instance_id"]:
                continue
            host, port = row.get("ssh_host"), row.get("ssh_port")
            if host and port and row.get("actual_status") == "running":
                ssh = SSH(host, int(port), "root", Path(config["ssh_key_file"]), folder / "known_hosts", repo_root=root)
                if cancel_event is not None:
                    ssh.cancel_event = cancel_event
                try:
                    ssh.run(["true"], timeout=max(.1, min(deadline - time.time(), 12)))
                    write_json(folder / "endpoint.json", {"host": host, "port": port, "user": "root"})
                    return ssh
                except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                    if cancel_event is not None and cancel_event.is_set():
                        raise InterruptedError("Cloud group startup cancelled") from None
                    attempts += 1
                    # Keep only a classification: stderr may contain host or
                    # provider text, so it is never copied into the audit.
                    stderr = getattr(error, "stderr", "") or ""
                    if isinstance(stderr, bytes):
                        stderr = stderr.decode("utf-8", errors="replace")
                    auth_rejected = "permission denied (publickey" in stderr.lower()
                    if auth_rejected and first_auth_failure is None:
                        first_auth_failure = time.time()
                    write_json(folder / "ssh-readiness.json", {"attempts": attempts,
                               "last_error_type": type(error).__name__, "publickey_rejected": auth_rejected,
                               "observed_at": time.time()})
                    if first_auth_failure is not None and time.time() - first_auth_failure >= config["ssh_auth_failure_seconds"]:
                        raise StartupUnavailable("SSH public-key authentication remained unavailable") from None
        delay = max(0, min(3, deadline - time.time()))
        if cancel_event is None:
            time.sleep(delay)
        elif cancel_event.wait(delay):
            raise InterruptedError("Cloud group startup cancelled")
    raise StartupUnavailable("Instance startup/SSH deadline exceeded")


def mathdx_script():
    """Download the pinned MathDx archive for the image's CUDA major version and verify it."""
    cases = " ".join(f"{major}) url={shlex.quote(url)}; sha={sha};;" for major, (url, sha) in MATHDX_ARCHIVES.items())
    return [
        "cuda_major=\"$(nvcc --version | sed -n 's/.*release \\([0-9][0-9]*\\)\\..*/\\1/p')\"",
        f'case "$cuda_major" in {cases} *) echo "no pinned MathDx archive for CUDA $cuda_major" >&2; exit 1;; esac',
        'curl -fsSL --retry 3 -o mathdx.tar.gz "$url"',
        'echo "$sha  mathdx.tar.gz" | sha256sum -c -',
        "mkdir -p mathdx && tar -xzf mathdx.tar.gz -C mathdx --strip-components=1 && rm mathdx.tar.gz",
    ]


def cudnn_script():
    cases = " ".join(f"{major}) url={shlex.quote(url)}; sha={sha};;" for major, (url, sha) in CUDNN_ARCHIVES.items())
    return [
        "cuda_major=\"$(nvcc --version | sed -n 's/.*release \\([0-9][0-9]*\\)\\..*/\\1/p')\"",
        f'case "$cuda_major" in {cases} *) echo "no pinned cuDNN archive for CUDA $cuda_major" >&2; exit 1;; esac',
        'curl -fsSL --retry 3 -o cudnn.tar.xz "$url"',
        'echo "$sha  cudnn.tar.xz" | sha256sum -c -',
        'mkdir -p cudnn && tar -xJf cudnn.tar.xz -C cudnn --strip-components=1 && rm cudnn.tar.xz',
        'test -f cudnn/include/cudnn.h && test -f cudnn/lib/libcudnn.so',
    ]


def bootstrap(ssh, remote, deadline, mathdx=False, cudnn=False, torch_headers=False):
    # Python>=3.9, CMake>=3.18: stock CUDA22.04/24.04 images meet these after apt.
    script = "\n".join([
        "set -euo pipefail", "unset CONTAINER_API_KEY VAST_API_KEY",
        "export DEBIAN_FRONTEND=noninteractive",
        f"cd -- {shlex.quote(remote)}", "mkdir -p source",
        "tar -xzf source.tar.gz -C source",
        "mkdir -p source/result/cloud-worker",
        "exec >source/result/cloud-worker/bootstrap.log 2>&1",
        "apt-get -qq update", "apt-get -qq install -y --no-install-recommends python3 python3-dev python3-venv python3-pip cmake build-essential git ca-certificates curl xz-utils patch",
        "python3 -m venv venv", "venv/bin/python -m pip --disable-pip-version-check install -r source/artifact/requirements.txt",
        "venv/bin/python -c 'import sys; assert sys.version_info >= (3,9)'", "nvcc --version", "cmake --version",
        *(mathdx_script() if mathdx else []),
        *(cudnn_script() if cudnn else []),
        *([f"venv/bin/python -m pip --disable-pip-version-check install --no-cache-dir --report source/result/cloud-worker/torch-install.json torch=={TORCH_HEADERS_VERSION} --index-url https://download.pytorch.org/whl/cpu",
           "venv/bin/python -c 'import torch; from pathlib import Path; assert (Path(torch.__file__).parent / \"include/torch/extension.h\").is_file(); print(torch.__version__)'"] if torch_headers else []),
        "cat /tmp/thunderfft-watchdog/status 2>/dev/null || true",
    ])
    ssh.run(["bash", "-c", script], timeout=remaining(deadline, 900))


def fetch_results(ssh, remote, folder, config, absolute_deadline=None):
    deadline = min(time.time() + config["download_seconds"], absolute_deadline or float("inf"))
    for filename in ("worker.exit.json", "worker-console.log"):
        try:
            ssh.download(remote + "/" + filename, folder / filename,
                         timeout=remaining(deadline, 10), max_bytes=1024*1024)
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
            pass  # A failure before worker startup can legitimately lack these.
    # Recover logs and terminal checkpoints even after worker/adapter failure.
    script = f"cd -- {shlex.quote(remote)} && if [ ! -f results.tar.gz ] && [ -d source/result ]; then tar -czf results.tar.gz -C source result; fi"
    ssh.run(["bash", "-c", script], timeout=remaining(deadline, 30))
    bound = int(config["download_gb"] * 1_000_000_000)
    ssh.download(remote + "/results.tar.gz", folder / "results.tar.gz", timeout=remaining(deadline, config["download_seconds"]), max_bytes=bound)
    safe_extract_result(folder / "results.tar.gz", folder / "download", max_bytes=bound)
    return folder / "download/result"


def execute_worker(ssh, remote, config, cancel_event=None, progress_callback=None):
    poll_progress = progress_callback is not None
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Cloud group execution cancelled")
    command = shlex.join([remote + "/venv/bin/python", remote + "/source/artifact/cloud/worker.py",
                          remote + "/source", remote + "/request.json"])
    script = f"cd -- {shlex.quote(remote)}; nohup {command} >worker-console.log 2>&1 </dev/null & echo $! > worker.pid"
    ssh.run(["bash", "-c", script], timeout=20)
    deadline = time.time() + config["execution_seconds"]
    # One SSH request per poll retrieves both files. The optional telemetry is
    # bounded and validated independently; malformed terminal exit evidence
    # must still fail the execution instead of looking like a running worker.
    poll_script = '''import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
sys.path.insert(0, str(root / "source"))
from artifact.progress import read_snapshot
exit_path = root / "worker.exit.json"
exit_value = None
if exit_path.exists():
    with exit_path.open("rb") as stream:
        data = stream.read(4097)
    if len(data) > 4096:
        raise ValueError("Worker exit evidence exceeds its size bound")
    exit_value = json.loads(data)
value = {"exit": exit_value, "progress": read_snapshot(root / "progress.json")}
encoded = json.dumps(value, ensure_ascii=True)
if len(encoded.encode()) >= 65536:
    value["progress"] = None
    encoded = json.dumps(value)
print(encoded)
'''
    while time.time() < deadline:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Cloud group execution cancelled")
        if not poll_progress:
            reply = ssh.run(["bash", "-c", f"cat {shlex.quote(remote + '/worker.exit.json')} 2>/dev/null || true"],
                            timeout=remaining(deadline, 15))
            if reply.stdout.strip():
                value = json.loads(reply.stdout)
                return int(value["exit_code"])
        else:
            reply = ssh.run([remote + "/venv/bin/python", "-c", poll_script, remote],
                            timeout=remaining(deadline, 15))
            if len(reply.stdout.encode()) >= 65536:
                raise ValueError("Remote worker status exceeds its size bound")
            envelope = json.loads(reply.stdout)
            if not isinstance(envelope, dict) or "exit" not in envelope:
                raise ValueError("Invalid remote worker status envelope")
            snapshot = validate_snapshot(envelope.get("progress"))
            if snapshot is not None and progress_callback is not None:
                try:
                    progress_callback(snapshot)
                except InterruptedError:
                    raise
                except Exception:
                    # Display is optional. Keep parsing the same combined
                    # envelope so a callback failure cannot hide worker exits.
                    progress_callback = None
            if envelope["exit"] is not None:
                return int(envelope["exit"]["exit_code"])
        delay = min(3, remaining(deadline, 3))
        if cancel_event is None:
            time.sleep(delay)
        elif cancel_event.wait(delay):
            raise InterruptedError("Cloud group execution cancelled")
    raise TimeoutError("Remote worker reached its execution allowance")


def stop_worker(ssh, remote):
    script = f'''cd -- {shlex.quote(remote)} || exit 0
if [ -f worker.exit.json ] || [ ! -f worker.pid ]; then exit 0; fi
thff_worker_pid=$(cat worker.pid)
case "$thff_worker_pid" in ''|*[!0-9]*) exit 1;; esac
kill -TERM "$thff_worker_pid" 2>/dev/null || true
for thff_wait in $(seq 1 25); do
  [ -f worker.exit.json ] && exit 0
  kill -0 "$thff_worker_pid" 2>/dev/null || exit 0
  sleep 1
done
exit 1
'''
    ssh.run(["bash", "-c", script], timeout=30)


def register_results(args, jobs, result, collection_module):
    if not args.collection:
        return []
    destination = collection_module.location(args.result_dir, args.collection)
    imported = []
    # Remote collection snapshots already carry the exact original part label.
    manifest_path = result / "collections" / args.collection / "manifest.json"
    if not manifest_path.exists():
        return imported
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("profile") != args.profile or manifest.get("collection") != args.collection:
        raise ValueError("remote collection identity/profile does not match the request")
    allowed = {(evaluation, job["part"]) for job in jobs for evaluation in job["evaluations"]}
    checked, seen = [], set()
    for part in manifest["parts"]:
        identity = (part["evaluation"], part["part"])
        if identity not in allowed or identity in seen:
            raise ValueError("remote collection contains an unexpected or duplicate part")
        seen.add(identity)
        expected_path = "parts/" + collection_module.name(identity[0]) + "/" + collection_module.name(identity[1])
        if part["path"] != expected_path:
            raise ValueError("remote collection part path does not match its identity")
        source = manifest_path.parent / part["path"]
        if not source.resolve().is_relative_to(manifest_path.parent.resolve()):
            raise ValueError("remote collection contains a path outside its result bundle")
        if collection_module.file_hashes(source) != part["files_sha256"]:
            raise ValueError("remote part hashes do not match its manifest")
        collection_module.import_status(source, part["evaluation"], args.profile)
        checked.append((part, source))
    for part, source in checked:
        collection_module.add_part(destination, part["evaluation"], part["part"], source, args.profile)
        imported.append({"evaluation": part["evaluation"], "part": part["part"]})
    return imported


def cleanup_and_bill(client, state, folder, seconds):
    state["cleanup_confirmed"] = False
    deadline, cleanup = time.time() + seconds, None
    while time.time() < deadline:
        try:
            cleanup = destroy_owned(client, state)
            write_json(folder / "cleanup.json", cleanup)
            if cleanup["destroyed_ids"] and not state.get("instance_id"):
                state["instance_id"] = cleanup["destroyed_ids"][0]
            if cleanup["confirmed"]:
                state["cleanup_confirmed"] = True
                break
        except ProviderError as error:
            write_json(folder / "cleanup-error.json", {"error": str(error), "observed_at": time.time()})
        time.sleep(3)
    write_json(folder / "state.json", state)
    if state.get("instance_id"):
        samples = []
        for index in range(2):
            try:
                samples.append(client.billing_evidence(state["created_at"], time.time(), state["instance_id"]))
            except ProviderError as error:
                samples.append({"error": str(error), "observed_at": time.time(), "settlement_verified": False})
            write_json(folder / "billing.json", {"samples": samples, "settlement_verified": False,
                       "note": "Per-instance charges may arrive later. Refresh with --cloud-recover; stable reads do not prove settlement."})
            if index == 0:
                time.sleep(3)
    return bool(state.get("cleanup_confirmed"))


def recover(folder):
    folder = folder.expanduser().resolve()
    state = json.loads((folder / "state.json").read_text())
    if not isinstance(state.get("label"), str) or not state["label"].startswith("thunderfft-ae-"):
        raise ValueError("Not a ThunderFFT cloud session")
    watchdog_cleanup = folder / "watchdog-cleanup.json"
    if not state.get("instance_id") and watchdog_cleanup.is_file():
        ids = json.loads(watchdog_cleanup.read_text()).get("destroyed_ids", [])
        if ids:
            state["instance_id"] = int(ids[0])
    client = VastClient(state["api_key_file"])
    clean = cleanup_and_bill(client, state, folder, 120)
    print(f"Cleanup {'confirmed' if clean else 'unresolved'}; billing evidence: {folder / 'billing.json'}", flush=True)
    return 0 if clean else 1


def attempt_cost(proposal, state, folder):
    """Reserve delayed billing and full transfer bounds after verified deletion.

    This is a conservative planning bound, never a provider-enforced cap or a
    declaration of settled charges. Unconfirmed cleanup cannot release budget.
    """
    cost = proposal["cost"]
    if not state["create_attempted"] and state["instance_id"] is None:
        return {"reserved_total_usd": 0.0, "observed_charge_usd": 0.0,
                "billing_settlement_verified": False, "basis": "no rental created"}
    cleanup_path = folder / "cleanup.json"
    cleanup = json.loads(cleanup_path.read_text()) if cleanup_path.exists() else {}
    confirmed = state["cleanup_confirmed"] and cleanup.get("confirmed")
    ended = cleanup.get("observed_at", time.time()) if confirmed else time.time()
    elapsed = max(0, ended - state["created_at"])
    billed_seconds = elapsed + cost["billing_delay_seconds"]
    usage = ((cost["compute_usd_per_hour"] + cost["storage_usd_per_hour"])
             * billed_seconds / 3600 + cost["network_usd"])
    billing_path = folder / "billing.json"
    samples = json.loads(billing_path.read_text()).get("samples", []) if billing_path.exists() else []
    observed = max([0.0] + [float(sample.get("observed_charge_usd", 0.0) or 0.0) for sample in samples])
    reserved = max(usage, observed) + cost["reserve_usd"]
    if not confirmed:
        reserved = max(reserved, cost["planned_total_usd"])
    return {"reserved_total_usd": reserved, "observed_charge_usd": observed,
            "lifetime_seconds": elapsed, "billing_delay_seconds": cost["billing_delay_seconds"],
            "network_bound_usd": cost["network_usd"], "reserve_usd": cost["reserve_usd"],
            "billing_settlement_verified": False,
            "basis": "elapsed lifetime plus billing delay and full transfer bounds" if confirmed else "cleanup unconfirmed"}


def run_attempt(args, root, collection_module, jobs, dependencies, client, config,
                public, lock, label, folder, proposal):
    created = time.time()
    state = {"label": label, "api_key_file": config["api_key_file"], "created_at": created,
             "deadline": created + config["startup_seconds"] + config["execution_seconds"] + config["download_seconds"],
             "instance_id": None, "create_attempted": False, "cleanup_confirmed": False, "finished": False}
    write_json(folder / "state.json", state)
    watchdog = arm_watchdog(folder, lock)
    ssh, fetched, code = None, None, 1
    retryable = False
    remote = "/workspace/" + label
    try:
        # Do not overlap this account's artifact execution batches.
        if any(str(row.get("label", "")).startswith("thunderfft-ae-") for row in client.instances()):
            raise ProviderError("An existing ThunderFFT rental needs cleanup before another execution")
        print(f"[{label}] {args.gpu}; all-in plan ${proposal['cost']['planned_total_usd']:.4f}; budget ${config['budget_usd']:.2f}", flush=True)
        acquire_instance(client, config, args.gpu, state, folder, proposal,
                         log=lambda message: print(message, flush=True))
        client.attach_ssh_key(state["instance_id"], public)
        startup_deadline = created + config["startup_seconds"]
        ssh = wait_ssh(client, state, config, folder, root, startup_deadline)
        ssh.run(["mkdir", "-p", remote], timeout=remaining(startup_deadline))
        ssh.run(["bash", "-c", "command -v python3 >/dev/null || (apt-get -qq update && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends python3)"],
                timeout=remaining(startup_deadline, 300))
        ssh.upload(folder / "source.tar.gz", remote + "/source.tar.gz", timeout=remaining(startup_deadline, 180))
        ssh.upload(folder / "request.json", remote + "/request.json", timeout=remaining(startup_deadline, 30))
        print(f"[{label}] preparing remote CUDA build environment", flush=True)
        bootstrap(ssh, remote, startup_deadline, **dependencies)
        status = ssh.run(["cat", "/tmp/thunderfft-watchdog/status"], timeout=remaining(startup_deadline), check=False)
        write_json(folder / "remote-watchdog.json", {"status": status.stdout.strip(), "credential_source": "provider container-scoped credential"})
        print(f"[{label}] executing {len(jobs)} job(s); results: {folder}", flush=True)
        outcome = execute_worker(ssh, remote, config)
        code = outcome if outcome in (0, 2, 124, 130) else 1
    except StartupUnavailable as error:
        retryable = ssh is None and state["instance_id"] is not None
        write_json(folder / "error.json", {"error_type": type(error).__name__, "error": str(error)})
        print(f"[{label}] startup failed; cleanup must finish before an alternate host is considered", flush=True)
    except KeyboardInterrupt:
        code = 130
        write_json(folder / "error.json", {"error": "Execution interrupted"})
    except Exception as error:
        # Provider/transport exceptions sanitize response bodies. Never log
        # environment dumps or arbitrary exception tracebacks.
        write_json(folder / "error.json", {"error_type": type(error).__name__, "error": str(error)[:1000]})
        print(f"Cloud execution failed ({type(error).__name__}); audit: {folder}", file=sys.stderr, flush=True)
    finally:
        # The detached watchdog remains active throughout interrupted cleanup.
        # Stop the remote runner before archiving terminal checkpoints.
        if ssh is not None:
            try:
                stop_worker(ssh, remote)
                fetched = fetch_results(ssh, remote, folder, config, state["deadline"])
                print(f"Results downloaded: {fetched}", flush=True)
            except Exception as error:
                write_json(folder / "download-error.json", {"error_type": type(error).__name__, "error": str(error)[:1000]})
                code = 1
        print(f"[{label}] destroying instance and recording billing", flush=True)
        clean = cleanup_and_bill(client, state, folder, config["cleanup_seconds"])
        state["finished"] = True
        write_json(folder / "state.json", state)
        if clean or not state["create_attempted"]:
            try:
                watchdog.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass  # It also observes the terminal state independently.
        else:
            print("Cleanup remains unresolved; detached watchdog is still active. Use --cloud-recover with this session directory.", file=sys.stderr)
            code = 1
    if fetched:
        try:
            imported = register_results(args, jobs, fetched, collection_module)
            write_json(folder / "imported.json", imported)
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"Result import failed: {error}; downloaded evidence retained at {fetched}", file=sys.stderr)
            code = 1
    cost = attempt_cost(proposal, state, folder)
    write_json(folder / "cost.json", cost)
    write_json(folder / "status.json", {"exit_code": code, "startup_retryable": retryable, "cleanup_confirmed": state["cleanup_confirmed"],
               "results_retrieved": fetched is not None, "billing_settlement_verified": False})
    print(f"Cloud audit: {folder}; billing settlement is unconfirmed.", flush=True)
    return {"folder": str(folder), "instance_id": state["instance_id"],
            "machine_id": proposal["offer"]["machine_id"], "exit_code": code,
            "cleanup_confirmed": bool(state["cleanup_confirmed"]),
            "startup_retryable": retryable, **cost}


def run(args, entries, registry, root, collection_module, select_tasks, parser):
    args.cloud_execution_summary = None
    if args.cloud_recover:
        return recover(args.cloud_recover)
    if args.assemble or args.import_result:
        raise ValueError("Use --backend local (default) for offline import/assembly")
    if args.device != 0:
        raise ValueError("Vast jobs use one GPU and require --device 0")
    config = load_config(args.cloud_config)
    jobs = make_jobs(args, entries, registry, config, select_tasks, parser)
    dependencies = required_dependencies(jobs, mathdx=config["mathdx"])
    if args.collection:
        existing = collection_module.read_manifest(collection_module.location(args.result_dir, args.collection))
        for job in jobs:
            if existing and any(p["part"] == job["part"] and p["evaluation"] in job["evaluations"] for p in existing["parts"]):
                raise ValueError("A requested collection part already exists; choose a new part before rental")
    if args.dry_run:
        print(json.dumps({"backend": "vast", "gpu": args.gpu, "image": config["images"][args.gpu],
                          "profile": args.profile, "jobs": jobs, "budget_usd": config["budget_usd"],
                          "dependencies": dependencies,
                          "api_calls": False, "instances_created": 0}, indent=2))
        return 0
    client = VastClient(config["api_key_file"], repo_root=root)
    proposal = plan(client, config, args.gpu)
    if args.cloud_plan:
        proposal["dependencies"] = dependencies
        proposal["jobs"] = jobs
        print(json.dumps(proposal, indent=2))
        print("Read-only plan. No instance created; availability and prices will be checked again at execution.")
        return 0
    # Validate private key and pair BEFORE incurring charges. Public-key
    # derivation uses batch mode; encrypted keys require an explicit usable key.
    validate_private_key(Path(config["ssh_key_file"]), repo_root=root)
    public = Path(config["ssh_public_key_file"]).read_text().strip()
    derived = subprocess.run(["ssh-keygen", "-y", "-P", "", "-f", config["ssh_key_file"]],
                             text=True, capture_output=True, timeout=10, check=True).stdout.strip()
    if public.split()[:2] != derived.split()[:2]:
        raise ValueError("SSH public and private key do not match")
    lock_path = Path(config["api_key_file"]).parent / ".thunderfft-execution.lock"
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another cloud execution or cleanup watchdog is active for this key directory") from None
        label = "thunderfft-ae-" + uuid.uuid4().hex
        folder = args.result_dir.resolve() / "cloud" / label
        folder.mkdir(parents=True, exist_ok=False)
        folder.chmod(0o700)
        source = build_source_bundle(root, folder / "source.tar.gz", allow_dirty=args.cloud_allow_dirty,
                                     max_bytes=min(512*1024**2, int(config["upload_gb"] * 1e9)))
        write_json(folder / "source.json", source)
        for index, job in enumerate(jobs):
            job["part"] = job["part"] or label + (f"-{index}" if index else "")
        request = {"gpu": args.gpu, "arch": ARCHS[args.gpu], "profile": args.profile, "jobs": jobs,
                   "build_jobs": args.jobs, "task_timeout": args.timeout, "collection": args.collection,
                   "execution_seconds": config["execution_seconds"], "max_result_bytes": int(config["download_gb"] * 1e9),
                   **dependencies}
        write_json(folder / "request.json", request)
        # Source preparation can take time; obtain fresh offer prices and balance
        # inside the execution lock immediately before arming and creating.
        proposal = plan(client, config, args.gpu)
        write_json(folder / "plan.json", proposal)
        execution_folder = folder
        first_source = folder / "source.tar.gz"
        attempts = []
        execution_budget = config["budget_usd"]
        for index in range(config["max_startup_attempts"]):
            outcome = run_attempt(args, root, collection_module, jobs, dependencies, client,
                                  config, public, lock, label, folder, proposal)
            attempts.append(outcome)
            reserved = sum(item["reserved_total_usd"] for item in attempts)
            summary = {"gpu": args.gpu, "budget_usd": execution_budget, "attempts": attempts,
                       "reserved_total_usd": reserved, "remaining_budget_usd": max(0, execution_budget - reserved),
                       "cleanup_confirmed": all(item["cleanup_confirmed"] for item in attempts),
                       "billing_settlement_verified": False, "exit_code": outcome["exit_code"],
                       "audit_folder": str(execution_folder)}
            args.cloud_execution_summary = summary
            write_json(execution_folder / "execution.json", summary)
            if (not outcome["startup_retryable"] or not outcome["cleanup_confirmed"]
                    or index + 1 == config["max_startup_attempts"]):
                return outcome["exit_code"]
            # Every attempt consumes this same execution budget, including full
            # transfer bounds and delayed charges; a retry never resets it.
            config = dict(config, budget_usd=summary["remaining_budget_usd"],
                          exclude_machine_ids=sorted(set(config["exclude_machine_ids"] + [outcome["machine_id"]])))
            try:
                proposal = plan(client, config, args.gpu)
            except ProviderError as error:
                summary["retry_stopped"] = str(error)
                write_json(execution_folder / "execution.json", summary)
                print(f"No alternate host fits the remaining execution budget: {error}", file=sys.stderr, flush=True)
                return outcome["exit_code"]
            label = "thunderfft-ae-" + uuid.uuid4().hex
            folder = args.result_dir.resolve() / "cloud" / label
            folder.mkdir(parents=True, exist_ok=False, mode=0o700)
            shutil.copyfile(first_source, folder / "source.tar.gz")
            write_json(folder / "source.json", source)
            write_json(folder / "request.json", request)
            write_json(folder / "plan.json", proposal)
            print(f"Retrying startup on machine {proposal['offer']['machine_id']}; "
                  f"${reserved:.4f} reserved for previous attempts, "
                  f"${config['budget_usd']:.4f} remains", flush=True)
