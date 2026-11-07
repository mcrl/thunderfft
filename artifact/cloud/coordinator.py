"""One bounded rental, several sequential AE jobs, fetched immutable evidence."""
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import uuid

from .config import (ARCHS, MATHDX_ARCHIVES, CUDNN_ARCHIVES, TORCH_HEADERS_VERSION,
                     inherited_environment, load_config, measurement_environment, required_dependencies)
from .provider import VastClient, ProviderError, CreateRejected, plan_cost
from .transport import SSH, build_source_bundle, safe_extract_result, validate_private_key
from .watchdog import destroy_owned, write_json


def add_arguments(parser):
    parser.add_argument("--backend", choices=("local", "vast"), default="local")
    parser.add_argument("--gpu", choices=tuple(ARCHS), default="A100", help="Vast GPU SKU: PCIe 80GB A100/H100 or B200")
    parser.add_argument("--cloud-config", type=Path, help="non-secret JSON settings; default ~/cloud_key/thunderfft/config.json")
    parser.add_argument("--cloud-plan", action="store_true", help="read-only account/offer/cost check; never create an instance")
    parser.add_argument("--cloud-batch", type=Path, help="JSON job list, sharing one rental/build; use --collection for named parts")
    parser.add_argument("--cloud-allow-dirty", action="store_true", help="include and hash tracked local edits; untracked source is rejected")
    parser.add_argument("--cloud-recover", type=Path, metavar="SESSION_DIR", help="destroy this saved session's rental and refresh billing; never rent again")


def make_jobs(args, entries, registry, config, select_tasks, parser):
    environment = inherited_environment(config)
    if not args.cloud_batch:
        return [{"evaluations": [e["id"] for e in entries], "part": args.part,
                 "environment": environment}]
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
        jobs.append({"evaluations": [e["id"] for e in selected], "part": label,
                     "environment": dict(environment, **measurement_environment(item.get("environment", {})))})
    return jobs


def plan(client, config, gpu):
    account = client.account_summary()
    client.billing_preflight()
    candidates, rejected = [], 0
    keys = ("disk_gb", "startup_seconds", "execution_seconds", "download_seconds", "cleanup_seconds",
            "billing_delay_seconds", "upload_gb", "download_gb", "budget_usd", "reserve_usd")
    for offer in client.search_offers(gpu, config["disk_gb"]):
        try:
            cost = plan_cost(offer, **{k: config[k] for k in keys})
        except ProviderError:
            rejected += 1
            continue
        if account["balance_usd"] >= cost["planned_total_usd"]:
            candidates.append((cost["planned_total_usd"], offer, cost))
    if not candidates:
        raise ProviderError(f"No matching {gpu} offer fits the all-in budget and available balance ({rejected} price plans rejected)")
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


def wait_ssh(client, state, config, folder, root, deadline):
    while time.time() < deadline:
        for row in client.instances():
            if int(row["id"]) != state["instance_id"]:
                continue
            host, port = row.get("ssh_host"), row.get("ssh_port")
            if host and port and row.get("actual_status") == "running":
                ssh = SSH(host, int(port), "root", Path(config["ssh_key_file"]), folder / "known_hosts", repo_root=root)
                try:
                    ssh.run(["true"], timeout=remaining(deadline, 12))
                    write_json(folder / "endpoint.json", {"host": host, "port": port, "user": "root"})
                    return ssh
                except (OSError, RuntimeError, subprocess.SubprocessError):
                    pass
        time.sleep(min(3, remaining(deadline, 3)))
    raise TimeoutError("Instance startup/SSH deadline exceeded")


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


def execute_worker(ssh, remote, config):
    command = shlex.join([remote + "/venv/bin/python", remote + "/source/artifact/cloud/worker.py",
                          remote + "/source", remote + "/request.json"])
    script = f"cd -- {shlex.quote(remote)}; nohup {command} >worker-console.log 2>&1 </dev/null & echo $! > worker.pid"
    ssh.run(["bash", "-c", script], timeout=20)
    deadline = time.time() + config["execution_seconds"]
    while time.time() < deadline:
        reply = ssh.run(["bash", "-c", f"cat {shlex.quote(remote + '/worker.exit.json')} 2>/dev/null || true"],
                        timeout=remaining(deadline, 15))
        if reply.stdout.strip():
            value = json.loads(reply.stdout)
            return int(value["exit_code"])
        time.sleep(min(3, remaining(deadline, 3)))
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


def run(args, entries, registry, root, collection_module, select_tasks, parser):
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
        created = time.time()
        state = {"label": label, "api_key_file": config["api_key_file"], "created_at": created,
                 "deadline": created + config["startup_seconds"] + config["execution_seconds"] + config["download_seconds"],
                 "instance_id": None, "create_attempted": False, "cleanup_confirmed": False, "finished": False}
        write_json(folder / "state.json", state)
        watchdog = arm_watchdog(folder, lock)
        ssh, fetched, code = None, None, 1
        remote = "/workspace/" + label
        try:
            # Do not overlap this account's artifact execution batches.
            if any(str(row.get("label", "")).startswith("thunderfft-ae-") for row in client.instances()):
                raise ProviderError("An existing ThunderFFT rental needs cleanup before another execution")
            print(f"[{label}] {args.gpu}; all-in plan ${proposal['cost']['planned_total_usd']:.4f}; budget ${config['budget_usd']:.2f}", flush=True)
            state["create_attempted"] = True
            write_json(folder / "state.json", state)
            try:
                state["instance_id"] = client.create_instance(proposal["offer"]["id"], image=proposal["image"],
                    disk_gb=config["disk_gb"], label=label, onstart=onstart_script(label, state["deadline"]))
            except CreateRejected:
                state["create_attempted"] = False
                write_json(folder / "state.json", state)
                raise
            write_json(folder / "state.json", state)
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
        write_json(folder / "status.json", {"exit_code": code, "cleanup_confirmed": state["cleanup_confirmed"],
                   "results_retrieved": fetched is not None, "billing_settlement_verified": False})
        print(f"Cloud audit: {folder}; billing settlement is unconfirmed. No further rentals started.", flush=True)
        return code
