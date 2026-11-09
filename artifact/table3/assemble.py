#!/usr/bin/env python3
"""Assemble immutable Table 3 result parts without launching CUDA workloads."""

import csv
import hashlib
import importlib.util
import io
import json
import os
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
                "relative_delta_from_paper", "cufftdx_comparison", "source_part", "candidate",
                "ffts_per_block", "elements_per_thread", "threads_per_block", "paper_target_us",
                "cufftdx_candidates")
SUMMARY_FIELDS = ("gpu", "N", "precision", "placement", "library", "candidate", "batch",
                  "inside_repeats", "trials", "median_time_us", "min_time_us", "max_time_us",
                  "median_e2e_out_of_place_us", "max_rel_l2", "passed", "ffts_per_block",
                  "elements_per_thread", "threads_per_block", "paper_target_us", "relative_delta_from_paper")
CANDIDATE_FIELDS = (*SUMMARY_FIELDS, "selected")
SPEEDUP_FIELDS = ("gpu", "placement", "thunderfft_precision", "cufftdx_precision", "N",
                  "thunderfft_median_us", "cufftdx_median_us", "cufftdx_candidate", "speedup", "paper_speedup")
LIBRARIES = {"ThunderFFT": MEASURE.PRECISIONS, "cuFFTDx": MEASURE.CUFFTDX_PRECISIONS}


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
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in [*([binary] if binary is not None else []), *hashes.values()]):
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
        key = (row["library"], int(row["N"]), row["precision"], row["placement"])
        if key[0] not in LIBRARIES or key[1] not in MEASURE.SIZES or key[2] not in LIBRARIES[key[0]] or key[3] not in MEASURE.PLACEMENTS:
            raise ValueError(f"{part}: unexpected Table 3 configuration {key}")
        if row["gpu"] != gpu:
            raise ValueError(f"{part}: source GPU/library mismatch")
        groups.setdefault(key, []).append(row)
    validated = []
    native_versions = set()
    artifacts = {"data.csv": sha256(data_path), "metadata.json": sha256(part / "metadata.json")}
    libraries = {}
    for (library, n, precision, placement), stored in sorted(groups.items()):
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
        selection = MEASURE.cufftdx_contract(metadata) if library == "cuFFTDx" else None
        candidate = (MEASURE.validate_preset_rows(stored, metadata, n, precision, placement)
                     if library == "cuFFTDx" else None)
        expected_argv = (["--library", "cufftdx"] if library == "cuFFTDx" else []) + (
            ["--candidate", candidate] if candidate is not None else []) + [
            str(n), precision, placement, str(identity["repeats"]), str(identity["trials"]), str(identity["batch"])]
        if len(argv) != len(expected_argv) + 1 or argv[1:] != expected_argv:
            raise ValueError(f"{part}: native argv disagrees with its measurement settings")
        log_path = evidence_file(part, command.get("stderr", ""))
        native_log = log_path.read_text()
        if MEASURE.gpu_name(native_log) != (gpu, metadata.get("gpu_name"), metadata.get("compute_capability")):
            raise ValueError(f"{part}: native GPU identity mismatch")
        versions = re.findall(r"^compiled_CUDART=.*$", native_log, re.MULTILINE)
        if len(versions) != 1:
            raise ValueError(f"{part}: missing native CUDA version evidence")
        native_versions.add(versions[0])
        binary_key = "cufftdx_binary_sha256" if library == "cuFFTDx" else "binary_sha256"
        digest = metadata.get(binary_key)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"{part}: missing {library} binary SHA256 evidence")
        library_identity = {"binary_sha256": digest}
        if library == "cuFFTDx":
            if identity["batch"] % MEASURE.CUFFTDX_BATCH_MULTIPLE:
                raise ValueError(f"{part}: cuFFTDx batch must be a multiple of 512")
            native = MEASURE.native_identity(native_log)
            if not native or metadata.get("cufftdx_build") != native:
                raise ValueError(f"{part}: missing or inconsistent cuFFTDx build identity")
            capability = metadata.get("compute_capability", "").split(".")
            if len(capability) != 2 or native["cufftdx_target_sm"] != int(capability[0]) * 100 + int(capability[1]) * 10:
                raise ValueError(f"{part}: cuFFTDx target SM disagrees with its measured device")
            library_identity["cufftdx_build"] = native
            library_identity["selection"] = selection
        if library in libraries and libraries[library] != library_identity:
            raise ValueError(f"{part}: library build identity changes across configurations")
        libraries[library] = library_identity
        parsed = MEASURE.parse_raw(native_path.read_text(), n=n, precision=precision, placement=placement,
                                   batch=identity["batch"], repeats=identity["repeats"], trials=identity["trials"],
                                   gpu=gpu, raw_csv=relative, library=library)
        by_trial = {(row["candidate"], int(row["trial"])): row for row in stored}
        if len(by_trial) != len(stored) or len(stored) != len(parsed):
            raise ValueError(f"{part}: incomplete or duplicate checkpoint trials")
        for row in parsed:
            saved = by_trial.get((row["candidate"], row["trial"]))
            if saved is None or any(str(row[key]) != saved[key] for key in MEASURE.FIELDS):
                raise ValueError(f"{part}: data.csv disagrees with validated native CSV")
            validated.append({**row, "source_part": str(part), "source_data_sha256": artifacts["data.csv"],
                              "source_raw_sha256": sha256(native_path)})
        artifacts[relative] = sha256(native_path)
        artifacts[command["stderr"]] = sha256(log_path)
    if len(native_versions) > 1:
        raise ValueError(f"{part}: CUDA runtime/compiler evidence changes across configurations")
    identity["native_versions"] = next(iter(native_versions), None)
    identity["libraries"] = libraries
    return validated, identity, {"path": str(part), "status": metadata["status"], "sha256": artifacts,
                                 "build_configuration": metadata.get("build_configuration", {})}


def row_sort_key(row):
    return (GPUS.index(row["gpu"]), row["N"], row["library"], row["precision"],
            row["placement"], row["trial"])


def resolve_source(directory, path):
    """Resolve new portable paths and historical absolute source references."""
    path = Path(path)
    return (path if path.is_absolute() else directory / path).resolve()


def relative_source(directory, path):
    return Path(os.path.relpath(Path(path).resolve(), directory.resolve())).as_posix()


def configuration(row):
    return row["gpu"], row["library"], row["N"], row["precision"], row["placement"]


def collect_parts(parts, profile):
    """Revalidate immutable native evidence and join disjoint configurations.

    Timing/device/software contracts are common to a GPU. Binary identities are
    compared per library: adding a cuFFTDx executable does not invalidate a
    previously collected ThunderFFT-only part. Every build record is retained.
    """
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported Table 3 profile")
    rows, sources, identities, configurations = [], [], {}, {}
    common_source = None
    common_cufftdx_selection = None
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
        contract = {key: value for key, value in identity.items()
                    if key not in ("binary_sha256", "build_configuration", "libraries")}
        previous = identities.setdefault(gpu, {**contract, "libraries": {}})
        differing = [key for key, value in contract.items() if value != previous.get(key)]
        if differing:
            raise ValueError(f"incompatible {gpu} parts ({', '.join(differing)}); use separate collections")
        for library, build in identity["libraries"].items():
            if library == "cuFFTDx":
                selection = build["selection"]
                if common_cufftdx_selection is not None and selection != common_cufftdx_selection:
                    raise ValueError("incompatible cuFFTDx selection modes or presets; use separate collections")
                common_cufftdx_selection = selection
            existing = previous["libraries"].get(library)
            if existing is not None and existing != build:
                raise ValueError(f"incompatible {gpu} {library} binary/build identity; use separate collections")
            previous["libraries"][library] = build
        part_keys = {configuration(row) for row in measured}
        overlap = part_keys & set(configurations)
        if overlap:
            raise ValueError(f"overlapping Table 3 configurations: {sorted(overlap)}; retain one part per configuration")
        configurations.update({key: str(part) for key in part_keys})
        rows.extend(measured)
        sources.append(evidence)
    rows.sort(key=row_sort_key)
    return rows, sources, identities, configurations, unused


def coverage_result(rows, profile):
    keys = {configuration(row) for row in rows}
    missing = [{"item": f"{gpu}/{library}/N={n}/{precision}/{placement}",
                "reason": "measurement CSV not collected"}
               for gpu in GPUS for library, precisions in LIBRARIES.items()
               for n in MEASURE.SIZES for precision in precisions for placement in MEASURE.PLACEMENTS
               if (gpu, library, n, precision, placement) not in keys]
    counts = {library: sum(key[1] == library for key in keys) for library in LIBRARIES}
    return {"status": "complete" if not missing else "partial" if rows else "pending",
            "missing": missing, "full_paper_comparison": not missing and profile == "paper",
            "complete_declared_subset": counts["ThunderFFT"] == 84,
            "excluded": [library for library, count in counts.items() if not count],
            "collected_configurations": len(keys), "expected_configurations": 168,
            "coverage": {"expected_thunderfft_configurations": 84,
                         "measured_thunderfft_configurations": counts["ThunderFFT"],
                         "expected_cufftdx_configurations": 84,
                         "measured_cufftdx_configurations": counts["cuFFTDx"]}}


def read_assembled(directory, profile):
    """Verify published assembly against its original native measurement files."""
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("assembly_schema") != "table3-validated-parts-v1" or metadata.get("evaluation") != "table3":
        raise ValueError("expected a validated Table 3 assembly")
    if metadata.get("profile") != profile or metadata.get("status") not in ("complete", "partial", "pending"):
        raise ValueError("Table 3 assembly profile/status mismatch")
    for relative, digest in metadata.get("outputs", {}).items():
        if sha256(evidence_file(directory, relative)) != digest:
            raise ValueError("assembled Table 3 output hash mismatch")
    if "data.csv" not in metadata.get("outputs", {}):
        raise ValueError("assembled Table 3 lacks data hash evidence")
    paths = []
    for source in metadata.get("sources", []):
        path = resolve_source(directory, source["path"])
        for relative, digest in source["sha256"].items():
            if sha256(evidence_file(path, relative)) != digest:
                raise ValueError("Table 3 source evidence changed after assembly")
        paths.append(path)
    rows, _, identities, _, _ = collect_parts(paths, profile)
    actual = strict_csv(directory / "data.csv", DATA_FIELDS)
    actual = [{**row, "source_part": str(resolve_source(directory, row["source_part"]))} for row in actual]
    if actual != [{key: str(row[key]) for key in DATA_FIELDS} for row in rows]:
        raise ValueError("assembled Table 3 rows disagree with validated original measurement evidence")
    expected = coverage_result(rows, profile)
    if metadata.get("coverage") != expected["coverage"] or metadata["status"] != expected["status"]:
        raise ValueError("assembled Table 3 coverage/status disagrees with its data")
    selection = selection_by_gpu(identities)
    if ("cufftdx_selection" in metadata or any(value["mode"] == "preset" for value in selection.values())) and metadata.get("cufftdx_selection") != selection:
        raise ValueError("assembled Table 3 cuFFTDx selection disagrees with its evidence")
    return rows, metadata


def selection_by_gpu(identities):
    return {gpu: identity["libraries"]["cuFFTDx"]["selection"]
            for gpu, identity in identities.items() if "cuFFTDx" in identity["libraries"]}


def assemble(parts, output, profile):
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, identities, configurations, unused = collect_parts(parts, profile)
    summaries, candidates, speedups = MEASURE.summarize(rows, profile)
    summaries = {configuration(row): row for row in summaries}
    table = []
    for gpu in GPUS:
        for library, precisions in LIBRARIES.items():
            for n in MEASURE.SIZES:
                for precision in precisions:
                    for placement in MEASURE.PLACEMENTS:
                        key = gpu, library, n, precision, placement
                        base = dict(gpu=gpu, N=n, precision=precision, placement=placement, library=library)
                        if key in summaries:
                            summary = summaries[key]
                            matched = (gpu, "cuFFTDx", n, MEASURE.MATCHED_PRECISION.get(precision), placement)
                            base.update(summary, status="collected", source_part=relative_source(output, configurations[key]),
                                        paper_thunderfft_target_us=summary["paper_target_us"] if library == "ThunderFFT" else "",
                                        cufftdx_comparison=("measured" if matched in summaries else "not_measured")
                                        if library == "ThunderFFT" else "baseline")
                        else:
                            base["status"] = "pending"
                        table.append(base)
    result = coverage_result(rows, profile)
    portable_rows = [{**row, "source_part": relative_source(output, row["source_part"])} for row in rows]
    MEASURE.write_csv(output / "data.csv", portable_rows, DATA_FIELDS)
    MEASURE.write_csv(output / "table.csv", table, TABLE_FIELDS)
    MEASURE.write_csv(output / "cufftdx_candidates.csv", candidates, CANDIDATE_FIELDS)
    MEASURE.write_csv(output / "speedup.csv", speedups, SPEEDUP_FIELDS)
    metadata = {"schema_version": 2, "assembly_schema": "table3-validated-parts-v1", "evaluation": "table3",
                "profile": profile, **result,
                "sources": [{**source, "path": relative_source(output, source["path"])} for source in sources],
                "unused_parts": [{**source, "path": relative_source(output, source["path"])} for source in unused],
                "source_path_base": "assembly_directory",
                "compatibility_by_gpu": identities,
                "cufftdx_selection": selection_by_gpu(identities),
                "candidate_selection": ("remeasure the saved GPU-specific candidate; no candidate search"
                    if any(value["mode"] == "preset" for value in selection_by_gpu(identities).values())
                    else "smallest median compute_us; ties retain native candidate order"),
                "precision_pairing": MEASURE.MATCHED_PRECISION,
                "operation": "offline assembly of previously collected evidence; no new measurements",
                "outputs": {name: sha256(output / name) for name in
                            ("data.csv", "table.csv", "cufftdx_candidates.csv", "speedup.csv")}}
    MEASURE.atomic_json(output / "metadata.json", metadata)
    return result
