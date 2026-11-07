"""Offline validation helpers for Table 6 and Figure 3 result bundles."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence_file(part, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("missing evidence path")
    result = (part / relative).resolve()
    if not result.is_relative_to(part.resolve()) or not result.is_file():
        raise ValueError(f"missing or escaping evidence path: {relative}")
    return result


def strict_csv(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"{path}: unexpected CSV header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value)


def metadata(part, evaluation, profile):
    result = json.loads(evidence_file(part, "metadata.json").read_text())
    if (result.get("evaluation"), result.get("profile"), result.get("status")) != (evaluation, profile, "completed"):
        raise ValueError(f"{part}: requires a completed {evaluation}/{profile} measurement")
    if result.get("numerical_and_timing_gates_passed") is not True:
        raise ValueError(f"{part}: numerical/timing gates have not passed")
    settings = result.get("settings", {})
    if any(type(settings.get(key)) is not int or settings[key] <= 0 for key in ("batch", "repeats", "trials")):
        raise ValueError(f"{part}: invalid settings")
    if profile == "paper" and settings["batch"] != 65536:
        raise ValueError("paper assembly requires batch 65536")
    sources = result.get("source_sha256", {})
    if not sources or not all(digest(value) for value in sources.values()):
        raise ValueError(f"{part}: missing source SHA256 evidence")
    return result


def native_identity(meta, log, table3):
    observed = table3.gpu_name(log)
    if observed != (meta.get("gpu"), meta.get("gpu_name"), meta.get("compute_capability")):
        raise ValueError("native GPU identity disagrees with metadata")
    uuid = table3.gpu_uuid(log)
    if not uuid or (meta.get("gpu_uuid") is not None and meta["gpu_uuid"] != uuid):
        raise ValueError("native GPU UUID missing or mismatched")
    inventory = meta.get("gpu_inventory_before", {})
    devices = list(csv.reader(io.StringIO(inventory.get("stdout", "")), skipinitialspace=True))
    matches = [device for device in devices if len(device) >= 4 and device[1].strip() == uuid]
    if inventory.get("exit_code") != 0 or len(matches) != 1 or matches[0][2].strip() != observed[1]:
        raise ValueError("native GPU UUID/name is absent or ambiguous in inventory")
    mask = meta.get("effective_CUDA_VISIBLE_DEVICES")
    if mask and mask.startswith("GPU-") and mask != uuid:
        raise ValueError("native GPU disagrees with visibility mask")
    versions = re.findall(r"^compiled_CUDART=(\d+) runtime=(\d+) driver=(\d+)", log, re.MULTILINE)
    if len(versions) != 1:
        raise ValueError("missing or duplicate native CUDA version evidence")
    return {"gpu": observed[0], "gpu_name": observed[1], "compute_capability": observed[2],
            "gpu_uuid": uuid, "driver_version": matches[0][3].strip(), "native_cuda": versions[0],
            **{key: meta["settings"][key] for key in ("batch", "repeats", "trials")},
            "source_sha256": meta["source_sha256"]}


def check_saved(stored, parsed, fields):
    # Native parsers canonicalize all numeric fields before measurement saves them.
    expected = [{key: str(row[key]) for key in fields} for row in parsed]
    key = lambda row: tuple(row[field] for field in fields)
    if sorted(stored, key=key) != sorted(expected, key=key):
        raise ValueError("data.csv disagrees with validated native CSV")


def merge_identity(identities, identity):
    gpu = identity["gpu"]
    if gpu in identities and identities[gpu] != identity:
        differences = [key for key in identity if identities[gpu].get(key) != identity[key]]
        raise ValueError(f"incompatible {gpu} parts ({', '.join(differences)}); use separate collections")
    identities[gpu] = identity


def assembly_paths(parts, output):
    """Reject output paths that could overwrite or become part of source evidence."""
    sources = sorted({Path(part).resolve() for part in parts})
    destination = Path(output).resolve()
    for source in sources:
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("assembly output must be separate from every source: neither equal, inside nor an ancestor")
    return sources, destination
