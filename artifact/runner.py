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
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
if sys.version_info < (3, 9):
    sys.exit(f"Python 3.9 or newer is required; found {platform.python_version()}. Set PYTHON=/path/to/python3.9+.")
REGISTRY = Path(__file__).with_name("registry.json")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact.cloud import coordinator as cloud

# Also works when CPU tests load this file by path rather than as a module.
_spec = importlib.util.spec_from_file_location("artifact_collection", Path(__file__).with_name("collection.py"))
collection = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(collection)


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
    selected = set(requested)

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
        # Optional dependencies only order tasks that were selected anyway.
        for dependency in by_id[task_id].get("optional_depends_on", []):
            if dependency in selected:
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
    if getattr(args, "collection", None):
        environment["THUNDERFFT_COLLECTION"] = args.collection
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
    parser.add_argument("--profile", choices=["smoke", "paper"], default=None,
                        help="default: collection profile if it exists, otherwise smoke")
    parser.add_argument("--result-dir", type=Path, default=ROOT / "result")
    parser.add_argument("--build-dir", type=Path, default=ROOT / "build_artifact")
    parser.add_argument("--device", type=nonnegative, default=0,
                        help="CUDA device ordinal within the inherited visible GPU set")
    parser.add_argument("--jobs", type=positive, default=2)
    parser.add_argument("--timeout", type=nonnegative, default=0,
                        help="per-evaluation timeout in seconds; 0 disables it")
    parser.add_argument("--collection", help="retain independent result parts in a named collection")
    parser.add_argument("--part", help="immutable label for this collected/imported part (default: run ID)")
    parser.add_argument("--assemble", action="store_true",
                        help="generate tables/figures and missing.csv from collected CSVs; no GPU execution")
    parser.add_argument("--import-result", action="append", default=[], metavar="EVALUATION=DIR",
                        help="import a completed result bundle with CSVs and metadata; no GPU execution")
    cloud.add_arguments(parser)
    args = parser.parse_args()
    if args.cloud_plan or args.cloud_recover:
        args.backend = "vast"
    if args.backend == "local" and (args.cloud_config or args.cloud_batch or args.cloud_allow_dirty):
        parser.error("cloud options require --backend vast")
    if args.cloud_recover:
        if args.dry_run or args.cloud_plan or args.cloud_batch or args.evaluations or args.collection:
            parser.error("--cloud-recover takes only a saved session directory")
        try:
            return cloud.recover(args.cloud_recover)
        except (ValueError, OSError, RuntimeError) as error:
            print(f"Cloud recovery error: {error}", file=sys.stderr)
            return 1
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    collection_dir = None
    if (args.part or args.assemble or args.import_result) and not args.collection:
        parser.error("--part, --assemble and --import-result require --collection")
    if args.import_result and args.evaluations:
        parser.error("--import-result uses EVALUATION=DIR; omit positional evaluations")
    if args.assemble and args.part and not args.import_result:
        parser.error("--part applies to collection/import, not assembly")
    try:
        if args.collection:
            collection_dir = collection.location(args.result_dir, args.collection)
            if args.part:
                collection.name(args.part)
            existing = collection.read_manifest(collection_dir)
            if existing and args.profile and existing["profile"] != args.profile:
                parser.error(f"collection profile is {existing['profile']}; use a separate collection for {args.profile}")
            args.profile = args.profile or (existing["profile"] if existing else "smoke")
        else:
            args.profile = args.profile or "smoke"
    except (ValueError, OSError) as error:
        parser.error(str(error))
    requested = ["all"] if (args.list or args.assemble) and not args.evaluations else args.evaluations
    entries = select_tasks(requested, registry["evaluations"], parser)
    if args.collection and not args.assemble and not args.import_result and not args.list:
        # Derived figures are regenerated offline from the collection's sources.
        entries = [e for e in entries if e["id"] != "figure2"]
    if args.backend == "vast" and not args.list:
        try:
            return cloud.run(args, entries, registry, ROOT, collection, select_tasks, parser)
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            print(f"Cloud error: {error}", file=sys.stderr)
            return 1
    if args.list or args.dry_run:
        for entry in entries:
            dependencies = ",".join(entry.get("depends_on", [])
                                    + [f"{d}?" for d in entry.get("optional_depends_on", [])]) or "-"
            print(f'{entry["id"]:8} {entry["implementation"]:10} '
                  f'owner={entry.get("owner") or "unassigned"} '
                  f'depends={dependencies} | {entry["title"]}')
        if args.dry_run:
            if args.assemble or args.import_result:
                print(f"Profile: {args.profile}; offline import/assembly; no GPU queries or execution")
            else:
                print(f"Profile: {args.profile}; device: {args.device}; sequential execution")
            if collection_dir:
                print(f"Collection: {collection_dir}")
                if args.assemble and (collection_dir / "assembled/manifest.json").is_file():
                    print("Previously assembled evaluations will also be refreshed and preserved.")
            if args.import_result:
                print("Import: " + ", ".join(args.import_result))
            if not args.assemble and not args.import_result:
                print(f"Results: {args.result_dir.resolve()}/<evaluation>/<run_id>/")
            print("No evaluations executed; no measurements or result files created.")
        return 0

    args.result_dir = args.result_dir.resolve()
    args.build_dir = args.build_dir.resolve()
    if collection_dir:
        try:
            collection.initialize(collection_dir, args.profile)
            registered = {e["id"] for e in registry["evaluations"]}
            if args.part and not args.import_result and not args.assemble:
                previous = collection.read_manifest(collection_dir)["parts"]
                selected_ids = {e["id"] for e in entries}
                duplicates = [p["evaluation"] for p in previous
                              if p["part"] == args.part and p["evaluation"] in selected_ids]
                if duplicates:
                    raise ValueError("part already exists for " + ", ".join(duplicates)
                                     + "; choose another --part before collecting")
            for request in args.import_result:
                evaluation, separator, path = request.partition("=")
                if not separator or evaluation not in registered or not path:
                    raise ValueError("--import-result must be a registered EVALUATION=DIR")
                if evaluation == "figure2":
                    raise ValueError("import Table 3 source data; Figure 2 is generated during --assemble")
                label = args.part or Path(path).resolve().name
                collection.add_part(collection_dir, evaluation, label, Path(path), args.profile)
                print(f"Imported {evaluation}/{label}: {collection_dir / 'parts' / evaluation / label}")
            if args.assemble:
                states = collection.assemble_collection(collection_dir, entries, args.profile)
                for state in states:
                    print(f"[{state['evaluation']}] {state['status']}; missing {len(state.get('missing', []))}")
                print(f"Assembled outputs: {collection_dir / 'assembled'}")
                print("No GPU measurements executed. Missing values remain uncollected.")
                return 0 if all(s["status"] == "complete" for s in states) else 2
            if args.import_result:
                print(f"Assemble with: ./run.sh --collection {args.collection} --assemble --result-dir {args.result_dir}")
                return 0
        except (ValueError, OSError, KeyError, TypeError) as error:
            print(f"Collection error: {error}", file=sys.stderr)
            return 1
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    args.run_id = run_id
    batch_dir = args.result_dir / "runs" / run_id
    batch_dir.mkdir(parents=True, exist_ok=False)
    metadata = {"schema_version": 1, "run_id": run_id, "started_at": now(),
                "profile": args.profile, "evaluations": [e["id"] for e in entries],
                "selection": args.evaluations or ["available"],
                "collection": args.collection, "part": args.part or run_id,
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
                    if key.startswith(("THUNDERFFT_TABLE3_", "THUNDERFFT_TABLE5_", "THUNDERFFT_TABLE7_")) or key in
                    ("CUDA_VISIBLE_DEVICES", "CUDACXX", "NVCC", "CUDAToolkit_ROOT", "MATHDX_ROOT", "THUNDERFFT_ARCHS")},
                "paper": registry["paper"], "scientific_validation": "not_assessed"}
    cloud_source = ROOT / ".thunderfft-cloud-source.json"
    if cloud_source.is_file():
        # A source archive carries original revision + actual per-file hashes;
        # it is deliberately not a synthetic Git repository or invented commit.
        metadata["cloud_source"] = json.loads(cloud_source.read_text())
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
            # A selected optional dependency is passed on only when it completed.
            dependency_results.update({d: by_id[d]["result_dir"] for d in entry.get("optional_depends_on", [])
                                       if d in by_id and by_id[d]["status"] == "completed"})
            state = execute(entry, output, args, dependency_results)
            recover_checkpoint = (entry["id"] == "table3" and state["status"] in ("failed", "interrupted")
                                  and (output / "data.csv").is_file())
            if collection_dir and (state["status"] == "completed" or recover_checkpoint):
                try:
                    record = collection.add_part(collection_dir, entry["id"], args.part or run_id,
                                                 output, args.profile)
                    state["collection_part"] = record["path"]
                except (ValueError, OSError, KeyError, TypeError) as error:
                    if recover_checkpoint:
                        state["checkpoint_import_error"] = str(error)
                    else:
                        state.update(status="failed", runner_exit_code=65,
                                     detail="measurement retained, collection import failed: " + str(error))
                atomic_json(output / "status.json", state)
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
    if collection_dir:
        print(f"Collected parts: {collection_dir / 'parts'}")
        print(f"Assemble with: ./run.sh --collection {args.collection} --assemble --result-dir {args.result_dir}")
    return exit_code


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, request_interrupt)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
