#!/usr/bin/env python3
"""Immutable result parts and GPU-free assembly of an explicit collection."""

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import uuid


HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value or ""):
        raise ValueError("collection and part names must contain 1–96 letters, digits, '.', '_' or '-', starting with a letter or digit")
    return value


def atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def location(result_root, collection):
    return Path(result_root).resolve() / "collections" / name(collection)


def read_manifest(directory):
    path = directory / "manifest.json"
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get("schema_version") != 1 or value.get("profile") not in ("smoke", "paper"):
        raise ValueError(f"unsupported collection manifest: {path}")
    return value


@contextmanager
def locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def initialize(directory, profile):
    with locked(directory):
        manifest = read_manifest(directory)
        if manifest is None:
            manifest = {"schema_version": 1, "collection": directory.name,
                        "profile": profile, "created_at": now(), "parts": []}
            atomic_json(directory / "manifest.json", manifest)
        elif manifest["profile"] != profile:
            raise ValueError(f"collection profile is {manifest['profile']}, not {profile}; use a separate collection")
    return manifest


def file_hashes(directory):
    hashes = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"result bundles cannot contain symlinks: {path}")
        if path.is_file():
            hashes[path.relative_to(directory).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.is_dir():
            raise ValueError(f"result bundles must contain regular files: {path}")
    return hashes


def import_status(source, evaluation, profile):
    """Accept successful subsets, or verified Table 3 terminal checkpoints."""
    status_path = source / "status.json"
    if not status_path.is_file():
        raise ValueError("import requires the result bundle including status.json, CSVs and metadata")
    status = json.loads(status_path.read_text())
    if status.get("evaluation") != evaluation or status.get("profile") != profile:
        raise ValueError(f"{evaluation}: import requires a successful {profile} result for the declared subset")
    if status.get("status") == "completed" and status.get("exit_code") == 0:
        return status, "completed_declared_subset"
    if evaluation == "table3" and status.get("status") in ("failed", "interrupted"):
        # Never accept running results. The terminal runner has joined or stopped
        # the child, and every retained configuration is independently rechecked.
        adapter = load_adapter(evaluation)
        rows, _, _ = adapter.load_part(source, profile)
        if rows:
            return status, "validated_terminal_checkpoint"
    raise ValueError(f"{evaluation}: import requires a successful {profile} result for the declared subset")


def add_part(directory, evaluation, label, source, profile):
    """Snapshot an independently validated subset, retaining its raw evidence."""
    name(evaluation)
    name(label)
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f"missing result directory: {source}")
    if source == directory or source in directory.parents:
        raise ValueError("cannot import a directory containing the collection itself")
    status, acceptance = import_status(source, evaluation, profile)
    if not (source / "data.csv").is_file() or not (source / "data.csv").stat().st_size:
        raise ValueError(f"{evaluation}: result bundle has no data.csv")
    initialize(directory, profile)
    with locked(directory):
        manifest = read_manifest(directory)
        if any(p["evaluation"] == evaluation and p["part"] == label for p in manifest["parts"]):
            raise ValueError(f"part already exists: {evaluation}/{label}; existing results are immutable")
        destination = directory / "parts" / evaluation / label
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(".import-" + uuid.uuid4().hex)
        if destination.exists():
            raise ValueError(f"part directory already exists: {destination}")
        before = file_hashes(source)
        try:
            shutil.copytree(source, temporary, symlinks=True)
            copied = file_hashes(temporary)
            if copied != before or file_hashes(source) != before:
                raise ValueError("result bundle changed during import; wait for its collection to finish")
            copied_status, copied_acceptance = import_status(temporary, evaluation, profile)
            if copied_status != status or copied_acceptance != acceptance:
                raise ValueError("result status changed during import")
            record = {"evaluation": evaluation, "part": label,
                      "path": destination.relative_to(directory).as_posix(),
                      "imported_at": now(), "original_result_dir": str(source),
                      "acceptance": acceptance, "original_status": status["status"],
                      "files_sha256": copied}
            temporary.rename(destination)
            manifest["parts"].append(record)
            atomic_json(directory / "manifest.json", manifest)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    return record


def load_adapter(evaluation):
    path = HERE / evaluation / "assemble.py"
    if not path.is_file():
        return None
    module_name = "artifact_collection_" + evaluation
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def simple_assembly(evaluation, parts, output, profile):
    """Reuse validated complete model/validation bundles without GPU work."""
    missing = []
    if evaluation not in ("table2", "validation"):
        return {"status": "pending", "full_paper_comparison": False,
                "missing": [{"item": "adapter", "reason": "incremental measurement/assembly adapter is not implemented"}]}
    if not parts:
        missing.append({"item": "data.csv", "reason": "not collected"})
    else:
        first = parts[0]
        for other in parts[1:]:
            # Repeated identical model results are harmless; different measurements
            # must be put in a separate collection rather than silently replaced.
            for filename in ("data.csv", "table.csv", "validation.json", "metadata.json"):
                a, b = first / filename, other / filename
                if a.exists() != b.exists() or (a.exists() and a.read_bytes() != b.read_bytes()):
                    raise ValueError(f"{evaluation}: overlapping parts conflict in {filename}; use a separate collection")
        for path in first.iterdir():
            if path.name != "status.json":
                if path.is_dir():
                    shutil.copytree(path, output / path.name)
                else:
                    shutil.copy2(path, output / path.name)
    if evaluation == "table2":
        missing.append({"item": "cuFFTDx measured row", "reason": "external parameters/measurement pending"})
    return {"status": "partial" if parts and missing else "complete" if parts else "pending",
            "missing": missing, "full_paper_comparison": False}


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def assemble_collection(directory, entries, profile):
    """Validate explicit parts and atomically publish a CPU-only assembly."""
    initialize(directory, profile)
    with locked(directory):
        manifest = read_manifest(directory)
        previous_path = directory / "assembled/manifest.json"
        if previous_path.is_file():
            # Refresh earlier published evaluations too, so a selected update
            # cannot remove existing tables or leave derived figures stale.
            previous = json.loads(previous_path.read_text())
            selected = {entry["id"] for entry in entries}
            selected.update(state["evaluation"] for state in previous["evaluations"])
            registry = json.loads((HERE / "registry.json").read_text())
            registered = {entry["id"]: entry for entry in registry["evaluations"]}
            registered.update({entry["id"]: entry for entry in entries})
            ordered, visited = [], set()
            def visit(evaluation):
                if evaluation in visited:
                    return
                visited.add(evaluation)
                for dependency in registered[evaluation].get("depends_on", []):
                    visit(dependency)
                ordered.append(registered[evaluation])
            for evaluation in registered:
                if evaluation in selected:
                    visit(evaluation)
            entries = ordered
        by_id = {}
        try:
            for record in manifest["parts"]:
                path = (directory / record["path"]).resolve()
                if not path.is_relative_to((directory / "parts").resolve()):
                    raise ValueError("invalid collection part path")
                if not path.is_dir() or file_hashes(path) != record["files_sha256"]:
                    raise ValueError(f"collected evidence was changed or removed: {record['evaluation']}/{record['part']}")
                by_id.setdefault(record["evaluation"], []).append(path)
        except Exception as error:
            atomic_json(directory / "assembly-error.json", {"at": now(), "error": str(error),
                        "previous_assembly_retained": (directory / "assembled").exists()})
            raise
        staging = directory / (".assemble-" + uuid.uuid4().hex)
        staging.mkdir()
        states, missing = [], []
        try:
            for entry in entries:
                evaluation = entry["id"]
                output = staging / evaluation
                output.mkdir()
                adapter = load_adapter(evaluation)
                sources = by_id.get(evaluation, [])
                if evaluation == "figure2":
                    # Figure 2 is derived from this assembly, without rerunning Table 3.
                    if sources:
                        raise ValueError("collect Table 3 CSVs for Figure 2; Figure 2 is regenerated during assembly")
                    sources = [staging / "table3"] if (staging / "table3").is_dir() else []
                elif evaluation == "table4":
                    # Reuse collected timings directly, including Table 3 bundles
                    # with cuFFTDx rows. No GPU work or full Table 3 assembly is needed.
                    sources = [*by_id.get("table3", []), *sources]
                state = (adapter.assemble(sources, output, profile) if adapter else
                         simple_assembly(evaluation, sources, output, profile))
                if state.get("status") not in ("complete", "partial", "pending"):
                    raise ValueError(f"{evaluation}: assembler returned an invalid status")
                provenance_sources = sources if evaluation == "table4" else by_id.get(evaluation, [])
                full_comparison = bool(state.get("full_paper_comparison")) if evaluation == "table4" else False
                state.update(evaluation=evaluation, profile=profile, operation="offline_assembly",
                             source_parts=[str(p.relative_to(directory)) if p.is_relative_to(directory) else str(p)
                                           for p in provenance_sources],
                             assembled_at=now(), full_paper_comparison=full_comparison)
                atomic_json(output / "status.json", state)
                for item in state.get("missing", []):
                    missing.append({"evaluation": evaluation, "item": item["item"], "reason": item["reason"]})
                states.append(state)
            write_csv(staging / "missing.csv", missing, ("evaluation", "item", "reason"))
            write_csv(staging / "status.csv", [{"evaluation": s["evaluation"], "status": s["status"],
                       "parts": len(s["source_parts"]), "missing_items": len(s.get("missing", [])),
                       "full_paper_comparison": s["full_paper_comparison"]} for s in states],
                      ("evaluation", "status", "parts", "missing_items", "full_paper_comparison"))
            atomic_json(staging / "manifest.json", {
                "schema_version": 1, "operation": "offline_assembly", "collection": directory.name,
                "profile": profile, "assembled_at": now(), "new_gpu_measurements": False,
                "parts": manifest["parts"], "evaluations": states,
                "complete": all(s["status"] == "complete" for s in states),
                "full_paper_comparison": bool(states) and all(s["full_paper_comparison"] for s in states)})
            final = directory / "assembled"
            backup = directory / (".previous-" + uuid.uuid4().hex)
            if final.exists():
                final.rename(backup)
            try:
                staging.rename(final)
            except BaseException:
                if backup.exists():
                    backup.rename(final)
                raise
            if backup.exists():
                shutil.rmtree(backup)
            (directory / "assembly-error.json").unlink(missing_ok=True)
        except Exception as error:
            atomic_json(directory / "assembly-error.json", {"at": now(), "error": str(error),
                        "previous_assembly_retained": (directory / "assembled").exists()})
            raise
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return states
