"""Run the complete paper evaluation and assemble its immutable GPU parts."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from artifact.cloud import coordinator as cloud
from artifact.cloud.config import (CUFFTDX_MODE_ENV, cufftdx_mode, CUDNN_MODE_ENV, cudnn_mode,
                                   inherited_environment, load_config, preview_modes)
from artifact.cloud.provider import VastClient
from artifact.cloud.transport import validate_private_key


def add_arguments(parser):
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


def preflight(config, a100_backend, args, root):
    client = VastClient(config["api_key_file"], repo_root=root)
    gpus = (["A100"] if a100_backend == "vast" else []) + ["H100", "B200"]
    proposals = [cloud.plan(client, stage_config(config, gpu, config["budget_usd"], args), gpu)
                 for gpu in gpus]
    total = sum(p["cost"]["planned_total_usd"] for p in proposals)
    balance = min(p["account_before"]["balance_usd"] for p in proposals)
    if total > min(config["budget_usd"], balance):
        raise ValueError(f"Complete pipeline plans ${total:.4f}, exceeding the shared "
                         f"${min(config['budget_usd'], balance):.4f} budget/balance; no rentals created")
    return {"planned_at": timestamp(), "a100_backend": a100_backend, "proposals": proposals,
            "planned_total_usd": total, "budget_usd": config["budget_usd"],
            "cufftdx_mode": cufftdx_mode(getattr(args, "cufftdx_mode", None), inherited_environment(config)),
            "cudnn_mode": cudnn_mode(getattr(args, "cudnn_mode", None), inherited_environment(config)),
            "provider_enforced_cap": False}


def run_local(args, root, folder, stop_process, gpu_uuid=None):
    if not gpu_uuid:
        raise ValueError("Local execution requires the selected GPU UUID")
    mode = cufftdx_mode(getattr(args, "cufftdx_mode", None))
    convolution_mode = cudnn_mode(getattr(args, "cudnn_mode", None))
    command = [sys.executable, str(root / "artifact/runner.py"), "all", "--backend", "local",
               "--profile", args.profile, "--collection", args.collection, "--part", "a100",
               "--cufftdx-mode", mode,
               "--cudnn-mode", convolution_mode,
               "--result-dir", str(args.result_dir), "--build-dir", str(args.build_dir),
               "--device", "0", "--jobs", str(args.jobs), "--timeout", str(args.timeout)]
    process = None
    with (folder / "a100.log").open("w") as log:
        try:
            process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True,
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuid,
                                                **{CUFFTDX_MODE_ENV: mode, CUDNN_MODE_ENV: convolution_mode}))
            while process.poll() is None:
                print(f"[A100] local experiments running; log: {folder / 'a100.log'}", flush=True)
                try:
                    return process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
            return process.returncode
        except BaseException:
            if process is not None:
                stop_process(process)
            raise


def assembly_result(states, assembled, entries, profile, root, cudnn_mode=None):
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
                              getattr(args, "cudnn_mode", None))
        print(f"Complete pipeline; profile: {args.profile}; A100: {args.a100_backend}")
        print(f"Table 3 cuFFTDx mode: {modes['cufftdx_mode']} (A100/H100/B200)")
        print(f"Figure 4/5 cuDNN mode: {modes['cudnn_mode']}")
        print("A100: " + ", ".join(e["id"] for e in entries if e["id"] != "figure2"))
        print("H100 PCIe 80GB: table3, table4; B200: table3, table4")
        print("Then assemble all tables and figures automatically; Figure 2 reuses A100 Table 3.")
        print("Cloud budget is shared by all GPUs and startup retries. No GPU/API queries or files created.")
        return 0
    config = load_config(args.cloud_config)
    args.cufftdx_mode = cufftdx_mode(getattr(args, "cufftdx_mode", None), inherited_environment(config))
    args.cudnn_mode = cudnn_mode(getattr(args, "cudnn_mode", None), inherited_environment(config))
    config = copy.deepcopy(config)
    config["environment"][CUFFTDX_MODE_ENV] = args.cufftdx_mode
    config["environment"][CUDNN_MODE_ENV] = args.cudnn_mode
    a100_backend, observed = choose_a100(args, wait=not args.cloud_plan)
    preview = preflight(config, a100_backend, args, root)
    if args.cloud_plan:
        print(json.dumps(preview, indent=2))
        print("Read-only full-pipeline plan. No instance created; offers are rechecked before each stage.")
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
             "collection": args.collection, "assembled_dir": str(destination / "assembled"),
             "a100_backend": a100_backend, "a100_observation": observed, "preflight": preview,
             "budget_usd": config["budget_usd"], "reserved_total_usd": 0.0, "stages": [],
             "billing_settlement_verified": False}

    def save():
        write_json(folder / "pipeline.json", state)
        with (folder / "summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["gpu", "backend", "status", "exit_code", "elapsed_seconds"])
            writer.writeheader()
            writer.writerows({k: s.get(k, "") for k in writer.fieldnames} for s in state["stages"])

    save()
    code = 1
    try:
        for gpu in ("A100", "H100", "B200"):
            backend = a100_backend if gpu == "A100" else "vast"
            stage = {"gpu": gpu, "backend": backend, "status": "running", "started_at": timestamp()}
            state["stages"].append(stage)
            save()
            started = time.monotonic()
            print(f"[{gpu}] starting {backend} stage; collection: {args.collection}", flush=True)
            child, stage_code, interrupted = None, 1, False
            try:
                if backend == "local":
                    stage_code = run_local(args, root, folder, stop_process, observed.get("uuid"))
                else:
                    available = config["budget_usd"] - state["reserved_total_usd"]
                    configured = stage_config(config, gpu, available, args)
                    # Reserve later GPUs before allowing retries in this stage.
                    order = ("A100", "H100", "B200")
                    later = order[order.index(gpu) + 1:]
                    client = VastClient(config["api_key_file"], repo_root=root)
                    future = sum(cloud.plan(client, stage_config(config, g, available, args), g)["cost"]["planned_total_usd"] for g in later)
                    configured["budget_usd"] = available - future
                    if configured["budget_usd"] <= config["reserve_usd"]:
                        raise ValueError("Remaining aggregate budget cannot cover this stage and later GPUs")
                    path = folder / (gpu.lower() + "-config.json")
                    write_json(path, configured)
                    path.chmod(0o600)
                    child = copy.copy(args)
                    child.backend, child.gpu, child.device = "vast", gpu, 0
                    child.cloud_config, child.part = path, gpu.lower()
                    child.cloud_execution_summary = None
                    child.evaluations = ["all"] if gpu == "A100" else ["table3", "table4"]
                    chosen = [e for e in select_tasks(child.evaluations, registry["evaluations"], parser) if e["id"] != "figure2"]
                    stage_code = cloud.run(child, chosen, registry, root, collection, select_tasks, parser)
                    evidence = getattr(child, "cloud_execution_summary", None)
                    if not evidence or not evidence["cleanup_confirmed"]:
                        stage_code = 1
                        raise RuntimeError("Cloud accounting or cleanup is unconfirmed; no further stage can start")
            except KeyboardInterrupt:
                stage_code, interrupted = 130, True
                raise
            except Exception:
                stage_code = 1
                raise
            finally:
                if child is not None:
                    evidence = getattr(child, "cloud_execution_summary", None)
                    stage["cloud"] = evidence
                    # Preserve child evidence even if interrupted after cleanup. If the
                    # child cannot report costs, reserve its entire allowance and stop.
                    charge = evidence["reserved_total_usd"] if evidence else configured["budget_usd"]
                    state["reserved_total_usd"] += charge
                    if not evidence:
                        stage["accounting_unconfirmed"] = True
                stage.update(exit_code=stage_code, elapsed_seconds=round(time.monotonic() - started, 3),
                             status="interrupted" if interrupted else "completed" if stage_code == 0 else "failed")
                save()
            if state["reserved_total_usd"] > config["budget_usd"]:
                raise RuntimeError("Observed conservative reservations exceeded the aggregate budget")
            save()
            if stage_code:
                code = stage_code
                break
        else:
            code = 0
    except KeyboardInterrupt:
        code = 130
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        code = 1
        state["error"] = str(error)
        print(f"Complete pipeline failed: {error}", file=sys.stderr, flush=True)
    finally:
        # Offline assembly must also preserve successful parts on exceptions or
        # interruption. It never starts another GPU execution.
        try:
            assembled = collection.assemble_collection(destination, entries, args.profile)
            state["assembly"] = assembly_result(assembled, destination / "assembled", entries, args.profile, root,
                                                cudnn_mode=args.cudnn_mode)
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
