#!/usr/bin/env python3
"""Run independently assignable paper evaluations; standard-library only."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = Path(__file__).with_name("registry.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=10, check=False)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def positive(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def request_interrupt(_signum, _frame):
    # Use execute()'s process-group cleanup for scheduler termination as well.
    raise KeyboardInterrupt


def select_tasks(requested, entries, parser):
    by_id = {entry["id"]: entry for entry in entries}
    if not requested or requested == ["available"]:
        requested = [entry["id"] for entry in entries if entry.get("available", False)]
        if not requested:
            parser.error("no available evaluations are registered")
    elif requested == ["all"]:
        requested = list(by_id)
    elif "all" in requested or "available" in requested:
        parser.error("use 'all' or 'available' by itself, or list evaluation IDs")
    unknown = set(requested) - set(by_id)
    if unknown:
        parser.error("unknown evaluation(s): " + ", ".join(sorted(unknown)))
    ordered, active, done = [], set(), set()

    def visit(task_id):
        if task_id in done:
            return
        if task_id in active:
            parser.error("cyclic evaluation dependency: " + task_id)
        if task_id not in by_id:
            parser.error("unknown dependency: " + task_id)
        active.add(task_id)
        for dependency in by_id[task_id].get("depends_on", []):
            visit(dependency)
        active.remove(task_id)
        done.add(task_id)
        ordered.append(by_id[task_id])

    for task_id in requested:
        visit(task_id)
    return ordered


def stop_process(process):
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    except ProcessLookupError:
        pass


def execute(entry, output, args, dependency_results=None):
    command = ["bash", str(ROOT / "artifact" / entry["id"] / "run.sh")]
    state = {"evaluation": entry["id"], "title": entry["title"],
             "profile": args.profile, "status": "running", "exit_code": None,
             "implementation": entry["implementation"], "owner": entry.get("owner"),
             "coverage": entry.get("coverage", "See the task README"),
             "full_paper_comparison": entry.get("full_paper_comparison", False),
             "scientific_validation": "not_assessed", "command": command,
             "started_at": now(), "result_dir": str(output)}
    atomic_json(output / "status.json", state)
    environment = os.environ.copy()
    environment.update({"THUNDERFFT_ROOT": str(ROOT),
                        "THUNDERFFT_RESULT_DIR": str(output),
                        "THUNDERFFT_BUILD_DIR": str(args.build_dir),
                        "THUNDERFFT_PROFILE": args.profile,
                        "THUNDERFFT_DEVICE": str(args.device),
                        "THUNDERFFT_JOBS": str(args.jobs),
                        "THUNDERFFT_PYTHON": sys.executable,
                        "THUNDERFFT_RUN_ID": args.run_id,
                        "THUNDERFFT_RESULTS_ROOT": str(args.result_dir),
                        "THUNDERFFT_DEPENDENCY_RESULTS": json.dumps(dependency_results or {})})
    started = time.monotonic()
    process = None
    try:
        with (output / "run.log").open("w", encoding="utf-8") as log:
            log.write("Command: " + " ".join(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            code = process.wait(timeout=args.timeout or None)
        state["exit_code"] = code
        state["status"] = "completed" if code == 0 else "pending" if code == 78 else "failed"
        if code == 0:
            required = entry.get("required_outputs", {}).get(args.profile, ["data.csv"])
            missing = [name for name in required
                       if not (output / name).is_file() or not (output / name).stat().st_size]
            state["required_outputs"] = required
            if missing:
                state.update(status="failed", runner_exit_code=65,
                             detail="adapter exited 0 but required outputs are missing or empty: " + ", ".join(missing))
        if code:
            lines = (output / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
            state["detail"] = "\n".join(lines[-8:])
    except subprocess.TimeoutExpired:
        stop_process(process)
        state.update(status="failed", exit_code=124, detail="evaluation timeout")
    except KeyboardInterrupt:
        if process:
            stop_process(process)
        state.update(status="interrupted", exit_code=130, detail="runner interrupted")
    except OSError as error:
        state.update(status="failed", exit_code=127, detail=str(error))
    state.update(finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 6))
    atomic_json(output / "status.json", state)
    return state


def write_summary(directory, states):
    atomic_json(directory / "summary.json", states)
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        columns = ["evaluation", "profile", "status", "exit_code", "elapsed_seconds", "result_dir"]
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(states)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="*",
                        help="evaluation IDs, available (default), or all including unfinished tasks")
    parser.add_argument("--list", action="store_true", help="list work items without running")
    parser.add_argument("--dry-run", action="store_true", help="show plan without creating results or querying GPUs")
    parser.add_argument("--profile", choices=["smoke", "paper"], default="smoke")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_artifact")
    parser.add_argument("--device", type=nonnegative, default=0,
                        help="CUDA device ordinal within the inherited visible GPU set")
    parser.add_argument("--jobs", type=positive, default=2)
    parser.add_argument("--timeout", type=nonnegative, default=0,
                        help="per-evaluation timeout in seconds; 0 disables it")
    args = parser.parse_args()
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    requested = ["all"] if args.list and not args.evaluations else args.evaluations
    entries = select_tasks(requested, registry["evaluations"], parser)
    if args.list or args.dry_run:
        for entry in entries:
            dependencies = ",".join(entry.get("depends_on", [])) or "-"
            print(f'{entry["id"]:8} {entry["implementation"]:10} '
                  f'owner={entry.get("owner") or "unassigned"} '
                  f'depends={dependencies} | {entry["title"]}')
        if args.dry_run:
            print(f"Profile: {args.profile}; device: {args.device}; sequential execution")
            print(f"Results: {args.result_dir.resolve()}/<evaluation>/<run_id>/")
            print("No evaluations executed; no measurements or result files created.")
        return 0

    args.result_dir = args.result_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    args.run_id = run_id
    batch_dir = args.result_dir / "runs" / run_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "run_id": run_id, "started_at": now(),
                "profile": args.profile, "evaluations": [e["id"] for e in entries],
                "selection": args.evaluations or ["available"],
                "coverage": {e["id"]: e.get("coverage", "See the task README") for e in entries},
                "device": args.device, "build_dir": str(args.build_dir),
                "jobs": args.jobs, "timeout_seconds": args.timeout,
                "launch_args": sys.argv[1:],
                "registry_sha256": hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
                "python": sys.version, "platform": platform.platform(),
                "git": capture(["git", "rev-parse", "HEAD"]),
                "branch": capture(["git", "branch", "--show-current"]),
                "worktree": capture(["git", "status", "--porcelain"]),
                "gpu": capture(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                "--format=csv,noheader"]),
                "configuration": {key: value for key, value in os.environ.items()
                    if key.startswith(("THUNDERFFT_TABLE3_", "THUNDERFFT_TABLE5_")) or key in
                    ("CUDA_VISIBLE_DEVICES", "CUDACXX", "NVCC", "CUDAToolkit_ROOT", "MATHDX_ROOT", "THUNDERFFT_ARCHS")},
                "paper": registry["paper"], "scientific_validation": "not_assessed"}
    atomic_json(batch_dir / "manifest.json", metadata)
    states, by_id = [], {}
    interrupted = False
    for entry in entries:
        output = args.result_dir / entry["id"] / run_id
        output.mkdir(parents=True, exist_ok=False)
        dependencies = [d for d in entry.get("depends_on", []) if by_id[d]["status"] != "completed"]
        if interrupted or dependencies:
            state = {"evaluation": entry["id"], "profile": args.profile,
                     "status": "interrupted" if interrupted else "blocked", "exit_code": 130 if interrupted else 78,
                     "elapsed_seconds": 0, "result_dir": str(output),
                     "scientific_validation": "not_assessed",
                     "detail": "runner interrupted" if interrupted else "dependencies not completed: " + ", ".join(dependencies)}
            (output / "run.log").write_text(state["detail"] + "\n", encoding="utf-8")
            atomic_json(output / "status.json", state)
        else:
            print(f'[{entry["id"]}] running; log: {output / "run.log"}', flush=True)
            dependency_results = {d: by_id[d]["result_dir"] for d in entry.get("depends_on", [])}
            state = execute(entry, output, args, dependency_results)
        interrupted = interrupted or state["status"] == "interrupted"
        states.append(state)
        by_id[entry["id"]] = state
        write_summary(batch_dir, states)
        print(f'[{entry["id"]}] {state["status"]} (exit {state["exit_code"]})', flush=True)
    if interrupted:
        exit_code = 130
    elif any(s["status"] == "failed" for s in states):
        exit_code = 1
    elif any(s["status"] != "completed" for s in states):
        exit_code = 2
    else:
        exit_code = 0
    metadata.update(finished_at=now(), exit_code=exit_code)
    atomic_json(batch_dir / "manifest.json", metadata)
    print(f"Summary: {batch_dir / 'summary.csv'}", flush=True)
    return exit_code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
