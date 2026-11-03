#!/usr/bin/env python3
"""Assemble immutable Table 3 result parts without launching CUDA workloads."""

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re

_SPEC = importlib.util.spec_from_file_location("table3_measure_contract", Path(__file__).with_name("measure.py"))
MEASURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(MEASURE)
GPUS = ("A100", "H100", "B200")
DATA_FIELDS = (*MEASURE.FIELDS, "source_part", "source_data_sha256", "source_raw_sha256")
TABLE_FIELDS = ("gpu", "N", "precision", "placement", "library", "status", "batch",
                "inside_repeats", "trials", "median_time_us", "min_time_us", "max_time_us",
                "median_e2e_out_of_place_us", "max_rel_l2", "passed", "paper_thunderfft_target_us",
                "relative_delta_from_paper", "cufftdx_comparison", "source_part")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_csv(path, fields):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(fields):
            raise ValueError(f"{path}: unexpected CSV header")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed CSV row")
    return rows


def evidence_file(part, name):
    path = (part / name).resolve()
    if not path.is_relative_to(part.resolve()) or not path.is_file():
        raise ValueError(f"{part}: missing or escaping evidence path {name!r}")
    return path


def device_identity(metadata, native_uuid=None):
    inventory = metadata.get("gpu_inventory_before", {})
    if inventory.get("exit_code") != 0:
        raise ValueError("Table 3 part lacks a successful GPU inventory")
    devices = list(csv.reader(io.StringIO(inventory.get("stdout", "")), skipinitialspace=True))
    mask = metadata.get("effective_CUDA_VISIBLE_DEVICES")
    if mask is None:
        mask = str(metadata.get("requested_logical_device", "0"))
    if not isinstance(mask, str) or not mask or "," in mask:
        raise ValueError("Table 3 part must identify exactly one selected GPU")
    if native_uuid is not None:
        matches = [row for row in devices if len(row) >= 4 and row[1].strip() == native_uuid]
        if mask.startswith("GPU-") and mask != native_uuid:
            raise ValueError("Table 3 native GPU UUID disagrees with the UUID visibility mask")
    elif mask.startswith("GPU-"):
        matches = [row for row in devices if len(row) >= 4 and row[1].strip() == mask]
    elif mask.isdecimal():
        # CUDA enumeration can differ from nvidia-smi index order. Historical logs
        # identify a physical GPU only if the observed model occurs exactly once.
        matches = [row for row in devices if len(row) >= 4 and row[2].strip() == metadata.get("gpu_name")]
    else:
        raise ValueError("Table 3 visibility mask is neither a CUDA ordinal nor a GPU UUID")
    if len(matches) != 1:
        raise ValueError("Table 3 selected GPU is ambiguous or absent; native UUID evidence is required")
    device = matches[0]
    if device[2].strip() != metadata.get("gpu_name"):
        raise ValueError("Table 3 GPU inventory and native device names disagree")
    return device[1].strip(), device[3].strip()


def fingerprint(metadata, profile, native_uuid=None):
    settings = metadata.get("settings", {})
    batch, repeats, trials = (settings.get(key) for key in ("batch", "repeats", "trials"))
    if any(type(value) is not int or value <= 0 for value in (batch, repeats, trials)) or batch % 16:
        raise ValueError("invalid Table 3 settings")
    if profile == "paper" and (batch != 65536 or repeats != 1000):
        raise ValueError("paper assembly requires batch 65536 and R=1000")
    hashes = metadata.get("source_sha256", {})
    binary = metadata.get("binary_sha256", "")
    if not isinstance(hashes, dict) or not hashes or not all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) for value in [binary, *hashes.values()]):
        raise ValueError("Table 3 part lacks valid source/binary SHA256 evidence")
    uuid, driver = device_identity(metadata, native_uuid)
    return {"batch": batch, "repeats": repeats, "trials": trials,
            "gpu_name": metadata.get("gpu_name"), "compute_capability": metadata.get("compute_capability"),
            "gpu_uuid": uuid, "driver_version": driver,
            "source_sha256": hashes, "binary_sha256": binary,
            "build_configuration": metadata.get("build_configuration", {})}


def load_part(part, profile):
    """Recover only complete, independently validated configurations, even after interruption."""
    part = part.resolve()
    metadata = json.loads(evidence_file(part, "metadata.json").read_text())
    if metadata.get("evaluation") != "table3" or metadata.get("profile") != profile:
        raise ValueError(f"{part}: evaluation/profile mismatch")
    if metadata.get("status") not in ("completed", "running", "failed"):
        raise ValueError(f"{part}: unsupported measurement status")
    gpu = metadata.get("gpu")
    if gpu not in GPUS:
        raise ValueError(f"{part}: expected an A100, H100 or B200 result, got {gpu!r}")
    data_path = evidence_file(part, "data.csv")
    raw_rows = strict_csv(data_path, MEASURE.FIELDS)
    native_uuids = set()
    for relative in {row["raw_csv"] for row in raw_rows}:
        commands = [entry for entry in metadata.get("commands", []) if entry.get("stdout") == relative]
        if len(commands) != 1 or commands[0].get("exit_code") != 0:
            raise ValueError(f"{part}: native measurement lacks one successful command record")
        log_path = evidence_file(part, commands[0].get("stderr", ""))
        native_uuids.add(MEASURE.gpu_uuid(log_path.read_text()))
    if len(native_uuids) > 1:
        raise ValueError(f"{part}: native GPU UUID evidence changes across configurations")
    native_uuid = next(iter(native_uuids), None)
    if metadata.get("gpu_uuid") is not None and metadata["gpu_uuid"] != native_uuid:
        raise ValueError(f"{part}: metadata and native GPU UUID disagree")
    identity = fingerprint(metadata, profile, native_uuid)
    groups = {}
    for row in raw_rows:
        key = (int(row["N"]), row["precision"], row["placement"])
        if key[0] not in MEASURE.SIZES or key[1] not in MEASURE.PRECISIONS or key[2] not in MEASURE.PLACEMENTS:
            raise ValueError(f"{part}: unexpected Table 3 configuration {key}")
        if row["gpu"] != gpu or row["library"] != "ThunderFFT":
            raise ValueError(f"{part}: source GPU/library mismatch")
        groups.setdefault(key, []).append(row)
    validated = []
    native_versions = set()
    artifacts = {"data.csv": sha256(data_path), "metadata.json": sha256(part / "metadata.json")}
    for (n, precision, placement), stored in sorted(groups.items()):
        paths = {row["raw_csv"] for row in stored}
        if len(paths) != 1:
            raise ValueError(f"{part}: configuration has multiple native CSV sources")
        relative = paths.pop()
        native_path = evidence_file(part, relative)
        command = [entry for entry in metadata.get("commands", []) if entry.get("stdout") == relative]
        if len(command) != 1 or command[0].get("exit_code") != 0:
            raise ValueError(f"{part}: native measurement lacks one successful command record")
        command = command[0]
        argv = command.get("argv", [])
        if len(argv) != 7 or argv[1:] != [str(n), precision, placement, str(identity["repeats"]),
                                          str(identity["trials"]), str(identity["batch"])]:
            raise ValueError(f"{part}: native argv disagrees with its measurement settings")
        log_path = evidence_file(part, command.get("stderr", ""))
        native_log = log_path.read_text()
        if MEASURE.gpu_name(native_log) != (gpu, metadata.get("gpu_name"), metadata.get("compute_capability")):
            raise ValueError(f"{part}: native GPU identity mismatch")
        versions = re.findall(r"^compiled_CUDART=.*$", native_log, re.MULTILINE)
        if len(versions) != 1:
            raise ValueError(f"{part}: missing native CUDA version evidence")
        native_versions.add(versions[0])
        parsed = MEASURE.parse_raw(native_path.read_text(), n=n, precision=precision, placement=placement,
                                   batch=identity["batch"], repeats=identity["repeats"], trials=identity["trials"],
                                   gpu=gpu, raw_csv=relative)
        by_trial = {int(row["trial"]): row for row in stored}
        if len(by_trial) != len(stored) or len(stored) != len(parsed):
            raise ValueError(f"{part}: incomplete or duplicate checkpoint trials")
        for row in parsed:
            saved = by_trial.get(row["trial"])
            if saved is None or any(str(row[key]) != saved[key] for key in MEASURE.FIELDS):
                raise ValueError(f"{part}: data.csv disagrees with validated native CSV")
            validated.append({**row, "source_part": str(part), "source_data_sha256": artifacts["data.csv"],
                              "source_raw_sha256": sha256(native_path)})
        artifacts[relative] = sha256(native_path)
        artifacts[command["stderr"]] = sha256(log_path)
    if len(native_versions) > 1:
        raise ValueError(f"{part}: CUDA runtime/compiler evidence changes across configurations")
    identity["native_versions"] = next(iter(native_versions), None)
    return validated, identity, {"path": str(part), "status": metadata["status"], "sha256": artifacts}


def assemble(parts, output, profile):
    output.mkdir(parents=True, exist_ok=True)
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported Table 3 profile")
    rows, sources, identities, configurations = [], [], {}, {}
    common_source = None
    unused = []
    for part in sorted({Path(part).resolve() for part in parts}):
        if not (part / "data.csv").is_file():
            unused.append({"path": str(part), "reason": "no complete configuration CSV collected"})
            continue
        measured, identity, evidence = load_part(part, profile)
        if not measured:
            unused.append({"path": str(part), "reason": "no complete configuration rows collected"})
            continue
        gpu = measured[0]["gpu"]
        if common_source is not None and identity["source_sha256"] != common_source:
            raise ValueError("incompatible source_sha256 across GPU families; use separate collections")
        common_source = identity["source_sha256"]
        if gpu in identities and identity != identities[gpu]:
            differing = [key for key in identity if identity[key] != identities[gpu].get(key)]
            raise ValueError(f"incompatible {gpu} parts ({', '.join(differing)}); use separate collections")
        identities[gpu] = identity
        part_keys = {(row["gpu"], row["N"], row["precision"], row["placement"]) for row in measured}
        overlap = part_keys & set(configurations)
        if overlap:
            raise ValueError(f"overlapping Table 3 configurations: {sorted(overlap)}; retain one part per configuration")
        configurations.update({key: str(part) for key in part_keys})
        rows.extend(measured)
        sources.append(evidence)
    rows.sort(key=lambda row: (GPUS.index(row["gpu"]), row["N"], row["precision"], row["placement"], row["trial"]))
    summaries = {(row["gpu"], row["N"], row["precision"], row["placement"]): row
                 for row in MEASURE.summarize(rows, profile)}
    table, missing = [], []
    for gpu in GPUS:
        for n in MEASURE.SIZES:
            for precision in MEASURE.PRECISIONS:
                for placement in MEASURE.PLACEMENTS:
                    key = gpu, n, precision, placement
                    base = dict(gpu=gpu, N=n, precision=precision, placement=placement,
                                library="ThunderFFT", cufftdx_comparison="not_measured")
                    if key in summaries:
                        table.append({**base, **summaries[key], "status": "collected", "source_part": configurations[key]})
                    else:
                        table.append({**base, "status": "pending"})
                        missing.append({"item": f"{gpu}/N={n}/{precision}/{placement}", "reason": "measurement CSV not collected"})
    missing.append({"item": "cuFFTDx comparisons", "reason": "tuned FP16/FP32 baseline measurements are not supplied"})
    status = "partial" if rows else "pending"
    result = {"status": status, "missing": missing, "full_paper_comparison": False,
              "complete_declared_subset": len(summaries) == 84,
              "excluded": ["cuFFTDx tuning and comparison measurements"],
              "collected_configurations": len(summaries), "expected_configurations": 84}
    MEASURE.write_csv(output / "data.csv", rows, DATA_FIELDS)
    MEASURE.write_csv(output / "table.csv", table, TABLE_FIELDS)
    metadata = {"schema_version": 1, "assembly_schema": "table3-validated-parts-v1", "evaluation": "table3",
                "profile": profile, **result, "sources": sources, "unused_parts": unused,
                "compatibility_by_gpu": identities,
                "coverage": {"expected_thunderfft_configurations": 84,
                             "measured_thunderfft_configurations": len(summaries)},
                "operation": "offline assembly of previously collected evidence; no new measurements",
                "outputs": {name: sha256(output / name) for name in ("data.csv", "table.csv")}}
    MEASURE.atomic_json(output / "metadata.json", metadata)
    return result
