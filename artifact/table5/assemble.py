#!/usr/bin/env python3
"""Assemble validated Table 5 phase CSVs offline, preserving missing cells."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

_spec = importlib.util.spec_from_file_location("thunderfft_table5_summary_helpers", Path(__file__).with_name("summarize.py"))
_summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_summary)


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: missing or invalid JSON: {error}") from error


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_gpu_uuid(value):
    if not isinstance(value, str) or not re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", value):
        return False
    return value[4:].replace("-", "") != "0" * 32


def _identity(metadata, environment):
    """Ignore part selection/paths; compare the actual measurement conditions."""
    try:
        identity = {key: metadata[key] for key in (
            "profile", "batch", "seed", "input", "reference", "implementations", "range_exponents")}
        source = metadata["source"]["sha256"]
        build = metadata["build"]
        binary = build["binary_sha256"]
        cache = build["cmake_cache"]
        compiler = build["cuda_compiler_version"]
    except (KeyError, TypeError) as error:
        raise ValueError(f"missing source/build or measurement provenance: {error}") from error
    valid_hash = lambda value: isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
    if not isinstance(source, dict) or not source or not all(valid_hash(value) for value in source.values()):
        raise ValueError("source.sha256 must contain nonempty SHA-256 provenance")
    if not valid_hash(binary) or not isinstance(cache, dict) or not cache or not isinstance(compiler, dict) or compiler.get("exit_code") != 0 or not compiler.get("stdout"):
        raise ValueError("binary hash, CMake build settings, and successful CUDA compiler version are required")
    environment_fields = {"gpu_name", "local_cuda_device", "pci_bus_id", "compute_capability",
                          "total_global_memory_bytes", "cuda_driver_api_version", "cuda_runtime_version",
                          "cuda_compile_version", "cufft_version"}
    if not isinstance(environment, dict) or environment_fields - environment.keys():
        raise ValueError("complete measured GPU/library environment is required")
    if "gpu_uuid" in environment and not _valid_gpu_uuid(environment["gpu_uuid"]):
        raise ValueError("measured gpu_uuid must be a valid nonzero CUDA device UUID")
    if not isinstance(identity["seed"], int) or not 0 <= identity["seed"] < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")
    if not identity["input"] or not identity["reference"]:
        raise ValueError("input and reference provenance are required")
    identity.update(source_sha256=source, binary_sha256=binary, cmake_cache=cache,
                    cuda_compiler_version=compiler, measured_environment=environment)
    return identity


def _canonical_rows(rows):
    return sorted(rows, key=lambda row: (int(row["N"]), row["precision"], float(row["alpha"])))


def assemble(parts: list[Path], output: Path, profile: str) -> dict:
    if profile not in {"smoke", "paper"}:
        raise ValueError("Table 5 assembly profile must be smoke or paper")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_sizes = [64] if profile == "smoke" else [64, 256, 1024, 4096]
    expected_ranges = ({"fp16": [8, 13], "tf32": [120, 127], "fp32": [120, 127]}
                       if profile == "smoke" else {precision: [0, 127] for precision in _summary.PRECISIONS})
    base, identity = None, None
    components, accuracy, ranges, raw_data, sources = {}, {}, {}, [], []
    seen_parts = set()
    for part in parts:
        part = Path(part).resolve()
        if part in seen_parts:
            continue
        seen_parts.add(part)
        if output.resolve() == part or part in output.resolve().parents:
            raise ValueError("assembly output must not overwrite or be inside an immutable source part")
        status = _read_json(part / "status.json")
        if status.get("evaluation") != "table5" or status.get("status") != "completed" or status.get("exit_code") != 0:
            raise ValueError(f"{part}: only successfully collected Table 5 parts can be assembled")
        if status.get("profile") != profile:
            raise ValueError(f"{part}: status profile does not match assembly profile")
        metadata = _read_json(part / "metadata.json")
        if metadata.get("evaluation") != "table5" or metadata.get("profile") != profile:
            raise ValueError(f"{part}: metadata evaluation/profile does not match Table 5 assembly")
        if metadata.get("range_exponents") != expected_ranges:
            raise ValueError(f"{part}: finite scan ranges do not match the {profile} profile")
        checked = _summary.validate_bundle(part, metadata)
        if checked["errors"]:
            raise ValueError(f"{part}: phase validation failed: {'; '.join(checked['errors'])}")
        if metadata.get("measured_environment") != checked["environment"]:
            raise ValueError(f"{part}: recorded measured environment differs from raw phase environment")
        part_identity = _identity(metadata, checked["environment"])
        incoming_components = {key: _canonical_rows(rows) for key, rows in checked["components"].items()}
        if identity is not None and (
                not _valid_gpu_uuid(identity["measured_environment"].get("gpu_uuid")) or
                not _valid_gpu_uuid(checked["environment"].get("gpu_uuid"))):
            if incoming_components != components:
                raise ValueError(f"{part}: GPU UUID is required to combine independently collected parts; "
                                 "a legacy bundle without a UUID is accepted only alone or as an identical duplicate")
        if identity is None:
            base, identity = metadata, part_identity
        elif identity != part_identity:
            differing = [key for key in identity if identity[key] != part_identity[key]]
            raise ValueError(f"{part}: incompatible Table 5 measurement conditions: {', '.join(differing)}")
        files = {name: _hash(part / name) for name in ("status.json", "metadata.json")}
        for phase in _summary.selected_phases(metadata):
            for suffix in ("csv", "environment.json"):
                name = f"{phase}.{suffix}"
                files[name] = _hash(part / name)
        sources.append({"result_dir": str(part), "sha256": files,
                        "N": metadata["N"], "selected_phases": _summary.selected_phases(metadata)})
        for (phase, n), rows in checked["components"].items():
            key, rows = (phase, n), _canonical_rows(rows)
            if key in components:
                if components[key] != rows:
                    raise ValueError(f"{part}: conflicting duplicate component {phase}:N={n}")
                continue
            components[key] = rows
            for row in rows:
                precision = row["precision"]
                is_accuracy = phase.startswith("accuracy_")
                raw_data.append({"phase": "accuracy" if is_accuracy else "finite_scan",
                                 "source_csv": str(part / f"{phase}.csv"),
                                 "implementation": metadata["implementations"][precision],
                                 "reference": "cuFFT FP64" if is_accuracy else "none (finite-output check)", **row})
                if is_accuracy:
                    accuracy[n, precision] = checked["accuracy"][n, precision]
                else:
                    ranges[n, precision] = checked["ranges"][n, precision]
    missing = _summary.missing_components(expected_sizes, components)
    state = "complete" if not missing else "partial" if components else "pending"
    metadata = dict(base) if base else {
        "evaluation": "table5", "profile": profile,
        "implementations": {"fp16": "ThunderFFT", "tf32": "ThunderFFT", "fp32": "cuFFT C2C"},
    }
    sizes = sorted(set(expected_sizes) | {n for _, n in components})
    metadata.update(N=sizes, expected_profile_sizes=expected_sizes, assembly_status=state,
                    selected_phases=list(_summary.PHASES), missing_components=missing,
                    full_paper_comparison=False, scientific_validation="not_assessed",
                    assembled_from=sources, compatibility=identity,
                    device_identity_scope=("CUDA device UUID" if identity and _valid_gpu_uuid(identity["measured_environment"].get("gpu_uuid"))
                                           else "single legacy bundle without GPU UUID" if identity else "pending"),
                    assembly_scope="Completeness of the requested profile CSV grid; paper population/provenance remains unestablished.")
    _summary.write_outputs(output, metadata, raw_data, _summary.summary_rows(sizes, metadata, accuracy, ranges))
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "validation.json").write_text(json.dumps({
        "numerical_sanity": "passed" if components else "not_assessed", "errors": [],
        "assembly_status": state, "accepted_components": len(components),
        "missing_components": missing, "full_paper_comparison": False,
        "scope": "Each imported phase is revalidated from its original CSV; missing phases remain pending.",
    }, indent=2) + "\n")
    return {"status": state, "missing": missing, "full_paper_comparison": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts, args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
