"""Run the complete paper evaluation and assemble its immutable GPU parts."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import uuid

from artifact.cloud import coordinator as cloud
from artifact.cloud import group as cloud_group
from artifact.cloud.config import (CUFFTDX_MODE_ENV, cufftdx_mode, CUDNN_MODE_ENV, cudnn_mode,
                                   TABLE5_RANGE_MODE_ENV, table5_range_mode,
                                   inherited_environment, load_config, preview_modes)
from artifact.cloud.provider import VastClient, ProviderError
from artifact import progress
from artifact.cloud.transport import validate_private_key


def add_arguments(parser):
    parser.add_argument('--progress', choices=('auto', 'plain', 'off'), default='auto',
                        help='live GPU progress on terminals; plain emits stage changes; off hides progress')
    parser.add_argument("--a100-backend", choices=("auto", "local", "vast"), default="auto",
                        help="complete pipeline: use an idle local A100, otherwise rent one (auto)")
    parser.add_argument("--local-wait-seconds", type=nonnegative, default=120,
                        help="wait for the selected local A100 before cloud fallback (default: 120)")
    parser.add_argument("--cloud-stage-seconds", type=positive, default=1200,
                        help="H100/B200 execution allowance, excluding startup (default: 1200)")
    parser.add_argument("--a100-stage-seconds", type=positive, default=3600,
                        help="rented A100 execution allowance, excluding startup (default: 3600)")


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def positive(value):
    result = nonnegative(value)
    if result < 60:
        raise argparse.ArgumentTypeError("must be at least 60 seconds")
    return result


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def inventory(device):
    """Respect the logical visibility mask; utilization=0 alone is not idle."""
    try:
        response = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=15, check=True)
        rows = [[v.strip() for v in row] for row in csv.reader(io.StringIO(response.stdout))]
        if any(len(row) != 4 for row in rows):
            raise ValueError("incomplete GPU inventory")
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            selected = []
            for token in visible.split(","):
                matches = [row for row in rows if row[0] == token.strip() or row[1] == token.strip()]
                if len(matches) != 1:
                    raise ValueError("cannot resolve CUDA_VISIBLE_DEVICES")
                selected.append(matches[0])
            rows = selected
        if device < 0 or device >= len(rows):
            raise ValueError("selected logical GPU is absent")
        row = rows[device]
        if "A100" not in row[2] or "PCIE" not in row[2].upper() or not 79000 <= int(row[3]) <= 84000:
            return {"available": False, "reason": "selected device is not an A100 PCIe 80GB", "name": row[2]}
        processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
                                   text=True, capture_output=True, timeout=15, check=True)
        process_rows = [[v.strip() for v in parts]
                        for parts in csv.reader(io.StringIO(processes.stdout)) if parts]
        if any(len(parts) != 2 or not parts[0].startswith("GPU-")
               or not parts[1].isdigit() or int(parts[1]) <= 0 for parts in process_rows):
            raise ValueError("incomplete GPU process inventory")
        busy = any(parts[0] == row[1] for parts in process_rows)
        return {"available": not busy, "uuid": row[1], "name": row[2],
                "reason": "another compute process owns the GPU" if busy else "idle"}
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return {"available": False, "reason": "local GPU inventory unavailable: " + str(error)}


def choose_a100(args, *, wait=True):
    if args.a100_backend == "vast":
        return "vast", {"reason": "explicit cloud A100 selection"}
    deadline = time.monotonic() + (args.local_wait_seconds if wait else 0)
    while True:
        observed = inventory(args.device)
        if observed["available"]:
            return "local", observed
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if args.a100_backend == "local":
                raise RuntimeError("Local A100 is unavailable: " + observed["reason"])
            return "vast", observed
        print(f"Waiting for local A100 ({observed['reason']}); {int(remaining)}s remaining", flush=True)
        time.sleep(min(10, remaining))


def stage_config(config, gpu, budget, args):
    result = copy.deepcopy(config)
    result["budget_usd"] = budget
    allowance = args.a100_stage_seconds if gpu == "A100" else args.cloud_stage_seconds
    result["execution_seconds"] = min(result["execution_seconds"], allowance)
    return result


class PreflightUnavailable(RuntimeError):
    """All required GPUs were checked; at least one cannot currently be planned."""

    def __init__(self, failures, proposals):
        self.failures, self.proposals = failures, proposals
        super().__init__('; '.join(f'{gpu}: {detail}' for gpu, detail in failures.items()))


def preflight(config, a100_backend, args, root):
    client = VastClient(config["api_key_file"], repo_root=root)
    gpus = (["A100"] if a100_backend == "vast" else []) + ["H100", "B200"]
    proposals, failures = [], {}
    for gpu in gpus:
        try:
            proposal = cloud.plan(client, stage_config(config, gpu, config["budget_usd"], args), gpu)
            proposals.append(proposal)
            print(f'[{gpu}] suitable offer found; planned total ${proposal["cost"]["planned_total_usd"]:.4f}', flush=True)
        except ProviderError as error:
            failures[gpu] = str(error)
            print(f'[{gpu}] preflight could not find a usable offer: {error}', flush=True)
    if failures:
        raise PreflightUnavailable(failures, proposals)
    total = sum(p["cost"]["planned_total_usd"] for p in proposals)
    balance = min(p["account_before"]["balance_usd"] for p in proposals)
    if total > min(config["budget_usd"], balance):
        raise ValueError(f"Complete pipeline plans ${total:.4f}, exceeding the shared "
                         f"${min(config['budget_usd'], balance):.4f} budget/balance; no rentals created")
    return {"planned_at": timestamp(), "a100_backend": a100_backend, "proposals": proposals,
            "planned_total_usd": total, "budget_usd": config["budget_usd"],
            "cufftdx_mode": cufftdx_mode(getattr(args, "cufftdx_mode", None), inherited_environment(config)),
            "cudnn_mode": cudnn_mode(getattr(args, "cudnn_mode", None), inherited_environment(config)),
            "table5_range_mode": table5_range_mode(getattr(args, "table5_range_mode", None), inherited_environment(config)),
            "execution_strategy": "parallel", "provider_enforced_cap": False}


def unavailable_notice(args, error, *, new_collection=False):
    common = ['--profile', args.profile]
    common += ['--progress', getattr(args, 'progress', 'auto')]
    for flag, value in (('--cufftdx-mode', args.cufftdx_mode), ('--cudnn-mode', args.cudnn_mode),
                        ('--table5-range-mode', args.table5_range_mode)):
        common += [flag, value]
    for field in ('result_dir', 'build_dir'):
        common += ['--' + field.replace('_', '-'), str(getattr(args, field))]
    for field in ('device', 'jobs', 'timeout'):
        common += ['--' + field.replace('_', '-'), str(getattr(args, field))]
    cloud_options = []
    if args.cloud_config:
        cloud_options += ['--cloud-config', str(args.cloud_config)]
    cloud_options += ['--a100-backend', args.a100_backend,
                     '--local-wait-seconds', str(args.local_wait_seconds),
                     '--cloud-stage-seconds', str(args.cloud_stage_seconds),
                     '--a100-stage-seconds', str(args.a100_stage_seconds)]
    if args.collection and not new_collection:
        cloud_options += ['--collection', args.collection]
    if args.cloud_allow_dirty:
        cloud_options += ['--cloud-allow-dirty']
    check = shlex.join(['./run.sh', *common, *cloud_options, '--cloud-plan'])
    retry = shlex.join(['./run.sh', *common, *cloud_options])
    local = shlex.join(['./run.sh', 'all', '--backend', 'local', *common])
    print('\nWe are very sorry: the parallel evaluation cannot start because the required\n'
          'cloud servers could not all be secured under the current availability and budget.\n'
          'Provider inventory is outside our control; details of the failed checks follow:\n'
          f'  {error}\n'
          'No instance has been created and no measurement has started.\n'
          'Please wait and check availability again, then rerun:\n'
          f'  {check}\n  {retry}\n'
          'Or run all A100 evaluations locally now, without cloud rentals:\n'
          f'  {local}\n'
          'The local run generates Figure 2-5 PDFs; H100/B200 table cells remain uncollected.\n'
          'See artifact/README.md for collecting those GPUs separately later.\n'
          'We apologize for the inconvenience and appreciate your patience.',
          file=sys.stderr, flush=True)
    if new_collection:
        print('The retry command creates a new collection; the failed attempt remains available for inspection.',
              file=sys.stderr, flush=True)


def run_local(args, root, folder, stop_process, gpu_uuid=None, cancel_event=None, *, quiet=False):
    if not gpu_uuid:
        raise ValueError("Local execution requires the selected GPU UUID")
    if cancel_event is not None and cancel_event.is_set():
        return 130
    mode = cufftdx_mode(getattr(args, "cufftdx_mode", None))
    convolution_mode = cudnn_mode(getattr(args, "cudnn_mode", None))
    range_mode = table5_range_mode(getattr(args, "table5_range_mode", None))
    command = [sys.executable, str(root / "artifact/runner.py"), "all", "--backend", "local",
               "--profile", args.profile, "--collection", args.collection, "--part", "a100",
               "--cufftdx-mode", mode,
               "--cudnn-mode", convolution_mode,
               "--table5-range-mode", range_mode,
               "--result-dir", str(args.result_dir), "--build-dir", str(args.build_dir),
               "--device", "0", "--jobs", str(args.jobs), "--timeout", str(args.timeout)]
    process = None
    with (folder / "a100.log").open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                                                THUNDERFFT_RUN_PROGRESS_FILE=str(folder / 'a100-progress.json'),
                                                **{CUFFTDX_MODE_ENV: mode, CUDNN_MODE_ENV: convolution_mode,
                                                   TABLE5_RANGE_MODE_ENV: range_mode}))
            last_progress = 0.0
            while process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    stop_process(process)
                    return 130
                if not quiet and time.monotonic() - last_progress >= 30:
                    print(f"[A100] local experiments running; log: {folder / 'a100.log'}", flush=True)
                    last_progress = time.monotonic()
                try:
                    return process.wait(timeout=1 if cancel_event is not None else 30)
                except subprocess.TimeoutExpired:
                    pass
            return process.returncode
        except BaseException:
            if process is not None:
                stop_process(process)
            raise


def assembly_result(states, assembled, entries, profile, root, cudnn_mode=None, table5_range_mode=None):
    """Missing required cells fail; validated fast subsets remain explicit limits."""
    from artifact.figure4 import policy
    missing_files, unresolved, limits = [], [], []
    variants = json.loads((root / "artifact/table7/variants.json").read_text())["variants"]
    unavailable = {v["id"] for v in variants if v["status"] != "available"}
    by_id = {s["evaluation"]: s for s in states}
    for entry in entries:
        evaluation = entry["id"]
        for name in entry.get("required_outputs", {}).get(profile, ["data.csv"]):
            path = assembled / evaluation / name
            if not path.is_file() or not path.stat().st_size:
                missing_files.append(str(path.relative_to(assembled)))
        state = by_id.get(evaluation)
        if state is None:
            unresolved.append({"evaluation": evaluation, "reason": "assembly state missing"})
            continue
        if evaluation in ("table2", "table3", "table4", "table5", "table6", "table7") and "presentation" in state:
            for name in ("table.pdf", "table.png", "table.html", "table-render.json"):
                path = assembled / evaluation / name
                if not path.is_file() or not path.stat().st_size:
                    missing_files.append(str(path.relative_to(assembled)))
        if evaluation == "table5" and state.get("range_mode") == "boundary":
            record = {"evaluation": evaluation, "item": "finite_scan",
                      "reason": "Finite boundaries rechecked locally; the full profile scan was not repeated."}
            if table5_range_mode == "full":
                unresolved.append(record)
            else:
                limits.append(record)
        skipped = state.get("policy_skipped", [])
        if skipped:
            filters = tuple(range(3, 54, 2)) if profile == "paper" else (
                (3, 9, 53) if evaluation == "figure4" else (3, 33, 53))
            allowed = ({f"cuDNN:filter={item['filter']}": item["reason"]
                        for item in policy.skipped_cells(evaluation, filters, ["cuDNN"], "fast")}
                       if evaluation in ("figure4", "figure5") else {})
            seen = set()
            for item in skipped:
                record = dict(item, evaluation=evaluation)
                key = item.get("item")
                if (cudnn_mode == "full" or state.get("cudnn_mode") != "fast"
                        or key in seen or key not in allowed or item.get("reason") != allowed[key]):
                    unresolved.append(record)
                else:
                    limits.append(record)
                seen.add(key)
        for item in state.get("missing", []):
            record = dict(item, evaluation=evaluation)
            if evaluation == "table7" and item["item"].split(".")[0] in unavailable:
                limits.append(record)
            else:
                unresolved.append(record)
        if state["status"] not in ("complete", "partial") or (state["status"] == "partial" and not state.get("missing")):
            unresolved.append({"evaluation": evaluation, "reason": state["status"]})
    return {"outputs_complete": not missing_files and not unresolved,
            "missing_files": missing_files, "unresolved": unresolved,
            "coverage_limits": limits, "full_paper_comparison": all(s.get("full_paper_comparison", False) for s in states)}


def run(args, registry, root, collection, select_tasks, parser, stop_process):
    if args.evaluations not in ([], ["all"]):
        raise ValueError("--backend all runs the entire evaluation; omit evaluation IDs or use all")
    if args.part or args.assemble or args.import_result or args.cloud_batch or args.cloud_recover:
        raise ValueError("complete pipeline does not accept part/import/assemble/cloud-batch/recovery options")
    entries = select_tasks(["all"], registry["evaluations"], parser)
    args.profile = args.profile or "paper"
    if args.dry_run:
        modes = preview_modes(args.cloud_config, getattr(args, "cufftdx_mode", None),
                              getattr(args, "cudnn_mode", None), getattr(args, "table5_range_mode", None))
        print(f"Complete pipeline; profile: {args.profile}; A100: {args.a100_backend}")
        print(f"Table 3 cuFFTDx mode: {modes['cufftdx_mode']} (A100/H100/B200)")
        print(f"Figure 4/5 cuDNN mode: {modes['cudnn_mode']}")
        print(f"Table 5 range mode: {modes['table5_range_mode']}")
        print("A100: " + ", ".join(e["id"] for e in entries if e["id"] != "figure2"))
        print("H100 PCIe 80GB: table3, table4; B200: table3, table4")
        print("Check every required cloud offer and combined budget, acquire and prepare all servers,")
        print("then run A100/H100/B200 concurrently. No measurements start before every server is ready.")
        print("Then assemble all tables and figures automatically; Figure 2 reuses A100 Table 3.")
        print("Cloud budget is shared by all GPUs; no automatic group startup retries. No GPU/API queries or files created.")
        return 0
    config = load_config(args.cloud_config)
    args.cufftdx_mode = cufftdx_mode(getattr(args, "cufftdx_mode", None), inherited_environment(config))
    args.cudnn_mode = cudnn_mode(getattr(args, "cudnn_mode", None), inherited_environment(config))
    args.table5_range_mode = table5_range_mode(getattr(args, "table5_range_mode", None), inherited_environment(config))
    config = copy.deepcopy(config)
    config["environment"][CUFFTDX_MODE_ENV] = args.cufftdx_mode
    config["environment"][CUDNN_MODE_ENV] = args.cudnn_mode
    config["environment"][TABLE5_RANGE_MODE_ENV] = args.table5_range_mode
    a100_backend, observed = choose_a100(args, wait=not args.cloud_plan)
    try:
        preview = preflight(config, a100_backend, args, root)
    except (PreflightUnavailable, ValueError) as error:
        unavailable_notice(args, error)
        return 1
    if args.cloud_plan:
        print(json.dumps(preview, indent=2))
        print("Read-only full-pipeline plan. No instance created; all offers are rechecked before group acquisition.")
        return 0
    validate_private_key(Path(config["ssh_key_file"]), repo_root=root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    args.result_dir = args.result_dir.expanduser().resolve()
    args.build_dir = args.build_dir.expanduser().resolve()
    args.collection = args.collection or "full-" + run_id
    destination = collection.location(args.result_dir, args.collection)
    existing = collection.read_manifest(destination)
    if existing and existing["parts"]:
        raise ValueError("Complete pipeline needs a new collection; omit --collection to create one automatically")
    folder = args.result_dir / "pipelines" / run_id
    folder.mkdir(parents=True)
    folder.chmod(0o700)
    collection.initialize(destination, args.profile)
    state = {"run_id": run_id, "started_at": timestamp(), "status": "running", "profile": args.profile,
             "cufftdx_mode": args.cufftdx_mode,
             "cudnn_mode": args.cudnn_mode,
             "table5_range_mode": args.table5_range_mode,
             "collection": args.collection, "assembled_dir": str(destination / "assembled"),
             "a100_backend": a100_backend, "a100_observation": observed, "preflight": preview,
             "budget_usd": config["budget_usd"], "reserved_total_usd": 0.0, "stages": [],
             "billing_settlement_verified": False, "execution_strategy": "parallel"}

    def save():
        write_json(folder / "pipeline.json", state)
        with (folder / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["gpu", "backend", "status", "exit_code", "elapsed_seconds"])
            writer.writeheader()
            writer.writerows({k: s.get(k, "") for k in writer.fieldnames} for s in state["stages"])

    save()
    print(f"Detailed logs: {folder}", flush=True)
    display = progress.Dashboard(mode=getattr(args, 'progress', 'auto'), log_path=folder / 'pipeline.log')
    def update_stage(gpu, update):
        stage = next((item for item in state['stages'] if item['gpu'] == gpu), None)
        if stage is None:
            stage = {'gpu': gpu, 'backend': 'local' if gpu == 'A100' and a100_backend == 'local' else 'vast'}
            state['stages'].append(stage)
        stage.update(update)
        state['stages'].sort(key=lambda item: ('A100', 'H100', 'B200').index(item['gpu']))
        # Progress is optional telemetry. Persist it with the next real stage
        # transition; a display-only update must not abort rented measurements.
        if set(update) != {'progress'}:
            save()
        display.update(gpu, update)
        display.refresh()

    def tick():
        if a100_backend == 'local':
            snapshot = progress.read_snapshot(folder / 'a100-progress.json')
            stage = next((item for item in state['stages'] if item['gpu'] == 'A100'), {})
            if snapshot is not None and 'completed' in snapshot and snapshot != stage.get('progress'):
                update_stage('A100', {'progress': snapshot})
        display.refresh()

    def local_ready():
        if a100_backend == 'local':
            current = inventory(args.device)
            if not current['available'] or current.get('uuid') != observed.get('uuid'):
                raise RuntimeError('Local A100 became unavailable while cloud servers were preparing: '
                                   + current.get('reason', 'GPU identity changed'))

    # Fixed per-GPU allowances add up to the one workflow budget. A worker may
    # never borrow another concurrent worker's allowance or reset spent costs.
    slack = config['budget_usd'] - preview['planned_total_usd']
    configs = {proposal['gpu']: stage_config(
        config, proposal['gpu'], proposal['cost']['planned_total_usd'] + slack / len(preview['proposals']), args)
        for proposal in preview['proposals']}
    state['cloud_budget_allocations'] = {gpu: settings['budget_usd'] for gpu, settings in configs.items()}
    save()
    code = 1
    try:
        for gpu in ('A100', 'H100', 'B200'):
            update_stage(gpu, {'status': 'waiting', 'progress': {
                'schema_version': 1, 'completed': 0,
                'total': len([entry for entry in entries if entry['id'] != 'figure2']) if gpu == 'A100' else 2,
                'evaluation': '', 'status': 'waiting', 'detail': None, 'updated_at': time.time()}})
        try:
            outcome = cloud_group.run(
                args, registry, root, collection, select_tasks, parser, configs, preview['proposals'],
                local_run=(lambda cancel: run_local(args, root, folder, stop_process,
                                                   observed.get('uuid'), cancel_event=cancel, quiet=True))
                          if a100_backend == 'local' else None,
                stage_update=update_stage, ready_check=local_ready,
                log=display.log, tick=tick)
        finally:
            try:
                try:
                    tick()
                except Exception:
                    pass  # Optional final telemetry cannot mask the group outcome.
            finally:
                display.close()
        state['cloud_group'] = outcome
        previous_progress = {item['gpu']: item.get('progress') for item in state['stages']}
        for stage in outcome['stages']:
            if 'progress' not in stage and previous_progress.get(stage['gpu']) is not None:
                stage['progress'] = previous_progress[stage['gpu']]
        state['stages'] = sorted(outcome['stages'], key=lambda item: ('A100', 'H100', 'B200').index(item['gpu']))
        state['reserved_total_usd'] = outcome['reserved_total_usd']
        code = outcome['exit_code']
        for stage in state['stages']:
            if stage.get('exit_code'):
                details = (str(folder / 'a100.log') if stage['backend'] == 'local' else
                           (stage.get('cloud') or {}).get('audit_folder') or outcome.get('audit_folder'))
                print(f"[{stage['gpu']}] {stage['status']} (exit {stage['exit_code']}); details: {details}",
                      file=sys.stderr, flush=True)
        if outcome.get('error'):
            state['error'] = outcome['error']
            print(f"Complete pipeline failed: {outcome['error']}", file=sys.stderr, flush=True)
        if not outcome['cleanup_confirmed']:
            code = 1
            state['error'] = 'Cloud cleanup is unconfirmed; inspect the group audit and active watchdogs.'
            print(state['error'], file=sys.stderr, flush=True)
            for stage in outcome['stages']:
                audit = stage.get('cloud') or {}
                if not audit.get('cleanup_confirmed', True) and audit.get('audit_folder'):
                    print('Recovery: ' + shlex.join(['./run.sh', '--backend', 'vast', '--cloud-recover',
                                                    audit['audit_folder']]), file=sys.stderr, flush=True)
        if state['reserved_total_usd'] > config['budget_usd']:
            code = 1
            state['error'] = 'Observed conservative reservations exceeded the aggregate budget.'
            print(state['error'], file=sys.stderr, flush=True)
        save()
    except cloud_group.PreflightUnavailable as error:
        code = 1
        state['error'] = str(error)
        unavailable_notice(args, error, new_collection=True)
    except KeyboardInterrupt:
        code = 130
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        code = 1
        state["error"] = str(error)
        print(f"Complete pipeline failed: {error}", file=sys.stderr, flush=True)
    finally:
        display.close()
        # Offline assembly must also preserve successful parts on exceptions or
        # interruption. It never starts another GPU execution.
        try:
            print('Assembling tables and figures from the collected results...', flush=True)
            assembled = collection.assemble_collection(destination, entries, args.profile)
            state["assembly"] = assembly_result(assembled, destination / "assembled", entries, args.profile, root,
                                                cudnn_mode=args.cudnn_mode, table5_range_mode=args.table5_range_mode)
            if code == 0 and not state["assembly"]["outputs_complete"]:
                code = 2
        except KeyboardInterrupt:
            code = 130
            state["assembly_error"] = "interrupted"
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            if code != 130:
                code = 1
            state["assembly_error"] = str(error)
            print(f"Collection assembly failed: {error}", file=sys.stderr, flush=True)
        state["status"] = ("interrupted" if code == 130 else
                           "completed_with_coverage_limits" if code == 0 and state["assembly"]["coverage_limits"] else
                           "completed" if code == 0 else "partial" if code == 2 else "failed")
        state.update(exit_code=code, finished_at=timestamp())
        save()
        print(f"Pipeline summary: {folder / 'pipeline.json'}", flush=True)
        print(f"Tables and figures: {destination / 'assembled'}", flush=True)
    return code
