#!/usr/bin/env python3
"""Combine validated convolution sweep parts; missing library/filter cells stay blank."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


MEASURE = module("conv_assembly_measure", HERE / "measure.py")
PLOT = module("conv_assembly_plot", HERE / "plot.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"{path}: missing or invalid JSON") from error


def evidence_file(part, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("evidence paths must be relative to the result bundle")
    path = (part / relative).resolve()
    if not path.is_relative_to(part.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping raw evidence: {relative}")
    return path


def native_contract(record, spec):
    """Recover library, shape, filter list and trial count from native argv."""
    argv = record.get("argv", [])
    if not isinstance(argv, list) or any(not isinstance(value, str) for value in argv):
        raise ValueError("invalid native argv evidence")
    if spec["evaluation"] == "figure4":
        if len(argv) == 14 and argv[1] == "4096":
            library, filter_arg, batch, channels, trials = "ThunderFFT", argv[2], int(argv[4]), int(argv[5]), int(argv[13])
            if argv[3] != "0" or argv[8:12] != ["1"] * 4:
                raise ValueError("ThunderFFT command must validate the real circular convolution and time the filter FFT")
        elif len(argv) == 10 and argv[1:3] == ["--figure4", "4096"]:
            library, filter_arg, batch, channels, trials = "cuDNN", argv[3], int(argv[4]), int(argv[5]), int(argv[9])
        elif len(argv) == 8:
            library, filter_arg, batch, channels, trials = "FlashFFTConv", argv[1], int(argv[2]), int(argv[3]), int(argv[5])
        else:
            raise ValueError("unexpected Figure 4 native command")
        shape = f"L=4096,B={batch},D={channels}"
        parser = MEASURE.figure4_parse
    else:
        if len(argv) != 7:
            raise ValueError("unexpected Figure 5 native command")
        names = {name.lower(): name for name in spec["libraries"]}
        if argv[3] not in names:
            raise ValueError("unsupported Figure 5 native library")
        library, filter_arg, trials = names[argv[3]], argv[2], int(argv[6])
        shape = f"N={int(argv[1])}"
        parser = module("conv_assembly_figure5_gate", HERE.parent / "figure5/measure.py").parse
    if record.get("library", library) != library:
        raise ValueError("native command library disagrees with metadata")
    filters = tuple(int(value) for value in filter_arg.split(","))
    if not filters or len(filters) != len(set(filters)):
        raise ValueError("empty or duplicate native filter list")
    return library, shape, filters, trials, parser


def canonical(row):
    result = {key: row[key] for key in MEASURE.FIELDS}
    for key in ("filter", "repeat"):
        result[key] = int(result[key])
    for key in ("time_ms", "rel_l2"):
        result[key] = float(result[key])
    result["passed"] = str(result["passed"]).lower() in ("true", "1")
    return result


def policy_evidence(metadata, incoming, spec):
    """Validate intentional omissions separately from uncollected measurements."""
    settings = metadata["settings"]
    mode = settings.get("cudnn_mode", "full")
    if "cudnn_mode" in settings:
        selected = settings.get("selected_libraries")
        if not isinstance(selected, list) or not selected or len(selected) != len(set(selected)) \
                or not set(selected) <= set(spec["libraries"]):
            raise ValueError("invalid selected libraries for cuDNN policy")
        if not set(metadata.get("binary_sha256", {})) <= set(selected):
            raise ValueError("native libraries disagree with selected libraries")
        if metadata.get("source_sha256", {}).get("artifact/figure4/policy.py") != digest(HERE / "policy.py"):
            raise ValueError("cuDNN policy lacks matching source hash evidence")
    else:
        # Historical bundles measured the full requested grid by default.
        selected = sorted({row["library"] for row in incoming})
    expected = MEASURE.POLICY.skipped_cells(spec["evaluation"], settings["filters"], selected, mode)
    if metadata.get("policy_skipped", []) != expected:
        raise ValueError("cuDNN policy skips disagree with selected filters/libraries/mode")
    cells = {(item["filter"], item["library"]) for item in expected}
    if any((int(row["filter"]), row["library"]) in cells for row in incoming):
        raise ValueError("cuDNN policy skips overlap measured rows in the same source")
    return mode, selected, expected


def revalidate_evidence(part, metadata, incoming, spec):
    """Reparse immutable native output and copied imports, then compare every row."""
    parsed, hashes, commands, libraries = [], {}, set(), set()
    settings = metadata["settings"]
    for record in metadata.get("commands", []):
        if record.get("exit_code") != 0:
            raise ValueError("native command evidence did not complete successfully")
        relative = record.get("stdout")
        if relative in commands:
            raise ValueError("duplicate native stdout command evidence")
        commands.add(relative)
        stdout = evidence_file(part, relative)
        # Early bundles recorded only stdout; their collector used the sibling .log.
        stderr_relative = record.get("stderr", str(Path(relative).with_suffix(".log")))
        stderr = evidence_file(part, stderr_relative)
        observed = MEASURE.measured_identity(metadata, stderr.read_text())
        if observed[:3] != (metadata.get("gpu"), metadata.get("gpu_name"), metadata.get("compute_capability")):
            raise ValueError("native GPU identity disagrees with metadata")
        if metadata.get("driver_version") not in (None, observed[4]):
            raise ValueError("native inventory driver disagrees with metadata")
        library, shape, filters, trials, parser = native_contract(record, spec)
        libraries.add(library)
        if library not in metadata.get("binary_sha256", {}):
            raise ValueError("native library lacks binary hash evidence")
        if trials != settings["trials"] or not set(filters) <= set(settings["filters"]):
            raise ValueError("native argv disagrees with filter/trial settings")
        results = parser(stdout.read_text())
        if sorted((size, trial) for size, trial, *_ in results) != sorted(
                (size, trial) for size in filters for trial in range(trials)):
            raise ValueError("incomplete or duplicate native raw timing trials")
        for size, trial, elapsed, error, validated in results:
            parsed.append({"gpu": observed[0], "shape": shape, "filter": size, "library": library,
                           "precision": spec["precision"], "repeat": trial, "time_ms": elapsed,
                           "filter_fft_included": "n/a" if library == "cuDNN" else "true",
                           "rel_l2": error, "validated": validated, "passed": "true",
                           "operation": spec["operation"], "source": relative})
        hashes[relative], hashes[stderr_relative] = digest(stdout), digest(stderr)
    if libraries != set(metadata.get("binary_sha256", {})):
        raise ValueError("native libraries disagree with command/binary evidence")
    shapes = {row["shape"] for row in incoming}
    if len(shapes) != 1:
        raise ValueError("raw bundle has inconsistent workloads")
    for relative, expected_hash in metadata.get("backend_sha256", {}).items():
        saved = evidence_file(part, relative)
        if digest(saved) != expected_hash:
            raise ValueError("copied backend CSV hash disagrees with metadata")
        rows, _ = MEASURE.read_backends([saved], spec, metadata["gpu"], next(iter(shapes)),
                                      settings["filters"], tuple(libraries))
        for row in rows:
            row["source"] = relative
        parsed.extend(rows)
        hashes[relative] = digest(saved)
    if not commands and not metadata.get("backend_sha256"):
        raise ValueError("no raw measurement evidence")
    key = lambda row: tuple(str(row[field]) for field in MEASURE.FIELDS)
    if sorted(map(canonical, incoming), key=key) != sorted(map(canonical, parsed), key=key):
        raise ValueError("data.csv disagrees with validated raw stdout/backend evidence")
    return hashes


def assemble_sweep(parts, output, profile, spec):
    if profile not in ("smoke", "paper"):
        raise ValueError("profile must be smoke or paper")
    output = Path(output).resolve()
    expected_filters = spec["filters"][profile]
    rows, sources, signatures, skipped = {}, [], {}, {}
    cudnn_modes = set()
    identity, base = None, None
    for part in sorted({Path(p).resolve() for p in parts}):
        if part == output or part in output.parents or output in part.parents:
            raise ValueError("assembly output cannot overwrite or be inside a source part")
        status, metadata = read_json(part / "status.json"), read_json(part / "metadata.json")
        for record in (status, metadata):
            if record.get("evaluation") != spec["evaluation"] or record.get("profile") != profile or record.get("status") != "completed":
                raise ValueError(f"{part}: only completed {spec['evaluation']} {profile} parts can be assembled")
        if status.get("exit_code") != 0 or metadata.get("numerical_and_timing_gates_passed") is not True:
            raise ValueError(f"{part}: measurement gates did not pass")
        if metadata.get("operation") != spec["operation"]:
            raise ValueError(f"{part}: operation contract differs")
        settings = metadata.get("settings", {})
        if not isinstance(settings.get("trials"), int) or settings["trials"] <= 0:
            raise ValueError(f"{part}: valid trial count is required")
        if not set(settings.get("filters", ())) <= set(spec["allowed_filters"]):
            raise ValueError(f"{part}: invalid filter selection")
        with (part / "data.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(MEASURE.FIELDS) <= set(reader.fieldnames or ()):
                raise ValueError(f"{part}: data.csv lacks required columns")
            incoming = list(reader)
        if not incoming:
            raise ValueError(f"{part}: completed part has no rows")
        mode, selected, policy_skipped = policy_evidence(metadata, incoming, spec)
        if "cuDNN" in selected or any(row["library"] == "cuDNN" for row in incoming):
            cudnn_modes.add(mode)
            if len(cudnn_modes) > 1:
                raise ValueError("incompatible cuDNN modes across parts")
        for item in policy_skipped:
            skipped[(item["filter"], item["library"])] = item["reason"]
        shapes = {row["shape"] for row in incoming}
        if len(shapes) != 1:
            raise ValueError(f"{part}: multiple workloads in one part")
        shape = next(iter(shapes))
        expected_shape = "L=4096,B=64,D=768" if spec["evaluation"] == "figure4" else "N=16384"
        if profile == "paper" and shape != expected_shape:
            raise ValueError(f"{part}: paper shape must be {expected_shape}")
        incoming_identity = {"gpu": metadata.get("gpu"), "gpu_name": metadata.get("gpu_name"),
                             "compute_capability": metadata.get("compute_capability"), "shape": shape,
                             "operation": metadata["operation"], "profile": profile,
                             "gpu_uuid": metadata.get("gpu_uuid"), "driver_version": metadata.get("driver_version")}
        if not incoming_identity["gpu"] or not incoming_identity["gpu_name"]:
            raise ValueError(f"{part}: measured GPU identity is required")
        if identity is not None:
            if identity != incoming_identity:
                raise ValueError(f"{part}: incompatible GPU or workload")
            if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", identity.get("gpu_uuid") or ""):
                raise ValueError("GPU UUID is required when merging independently collected parts")
        else:
            identity, base = incoming_identity, metadata
        local_keys = set()
        for row in incoming:
            library = row["library"]
            if library not in spec["libraries"] or row["gpu"] != identity["gpu"]:
                raise ValueError(f"{part}: invalid library/GPU")
            size, repeat = int(row["filter"]), int(row["repeat"])
            if size not in settings["filters"] or not 0 <= repeat < settings["trials"]:
                raise ValueError(f"{part}: row lies outside the declared filter/trial selection")
            if row["passed"].lower() not in ("1", "true") or row["precision"] != spec["precision"] or row["operation"] != spec["operation"]:
                raise ValueError(f"{part}: row failed correctness or operation contract")
            MEASURE.check_row(row)
            if row["filter_fft_included"] not in (("n/a", "false", "0") if library == "cuDNN" else ("true", "1", "unknown")):
                raise ValueError(f"{part}: unexpected filter FFT timing scope")
            key = (size, library, repeat)
            if key in local_keys:
                raise ValueError(f"{part}: duplicate row {key}")
            local_keys.add(key)
            canonical = {**row, "filter": size, "repeat": repeat,
                         "time_ms": float(row["time_ms"]), "rel_l2": float(row["rel_l2"]), "passed": "true"}
            comparison = {k: v for k, v in canonical.items() if k != "source"}
            if key in rows and {k: v for k, v in rows[key].items() if k != "source"} != comparison:
                raise ValueError(f"{part}: conflicting duplicate measurement {key}")
            binary = metadata.get("binary_sha256", {}).get(library)
            source_hashes = metadata.get("source_sha256")
            if binary:
                if not re.fullmatch(r"[0-9a-f]{64}", binary) or not isinstance(source_hashes, dict) or not source_hashes or not all(re.fullmatch(r"[0-9a-f]{64}", value) for value in source_hashes.values()):
                    raise ValueError(f"{part}: missing native source/binary provenance")
                signature = {"binary": binary, "source": source_hashes}
                if library in signatures and signatures[library] != signature:
                    raise ValueError(f"{part}: incompatible {library} implementation across parts")
                signatures[library] = signature
            elif not metadata.get("backend_sha256", {}).get(row["source"]):
                raise ValueError(f"{part}: imported backend has no recorded CSV hash")
            rows[key] = canonical
        groups = {(size, library) for size, library, _ in local_keys}
        for size, library in groups:
            repeats = {repeat for f, lib, repeat in local_keys if f == size and lib == library}
            if repeats != set(range(settings["trials"])):
                raise ValueError(f"{part}: incomplete trial set for {library}:filter={size}")
        raw_hashes = revalidate_evidence(part, metadata, incoming, spec)
        raw_hashes.update({name: digest(part / name) for name in ("status.json", "metadata.json", "data.csv")})
        sources.append({"result_dir": str(part), "sha256": raw_hashes, "gpu_uuid": metadata.get("gpu_uuid"),
                        "external_build": metadata.get("external_build"), "truncated": metadata.get("truncated", {})})
    missing, policy_skipped = [], []
    # Include custom filters as well as the profile grid; this preserves incomplete work.
    filters = sorted(set(expected_filters) | {key[0] for key in rows} | {key[0] for key in skipped})
    for library in spec["libraries"]:
        for size in filters:
            group = [row for (f, lib, _), row in rows.items() if f == size and lib == library]
            if not group:
                item = f"{library}:filter={size}"
                if (size, library) in skipped:
                    policy_skipped.append({"item": item, "reason": skipped[(size, library)]})
                else:
                    missing.append({"item": item, "reason": "validated timing not collected"})
            elif any(row["filter_fft_included"] == "unknown" for row in group):
                missing.append({"item": f"{library}:filter={size}:timing_scope", "reason": "imported CSV does not state whether filter FFT is timed"})
    state = "complete" if not missing else "partial" if rows else "pending"
    ordered = [rows[key] for key in sorted(rows)]
    output.mkdir(parents=True, exist_ok=True)
    MEASURE.TABLE3.write_csv(output / "data.csv", ordered, MEASURE.FIELDS)
    MEASURE.write_summary(output / "summary.csv", ordered, spec)
    metadata = {"schema_version": 1, "evaluation": spec["evaluation"], "profile": profile,
                "status": state, "operation": spec["operation"], "assembled_from": sources,
                "identity": identity, "implementation_signatures": signatures, "missing": missing,
                "cudnn_mode": next(iter(cudnn_modes), "full"), "policy_skipped": policy_skipped,
                "full_paper_comparison": False, "coverage_complete": not missing and not policy_skipped,
                "scope": "Validated rows are merged by library/filter/trial. Completeness does not establish original paper binary identity.",
                "notes": spec.get("notes")}
    if ordered:
        try:
            plot_rows = PLOT.read_summary(output / "summary.csv")
            # Include expected cells as gaps so uncollected interior filters are not connected.
            for library in spec["libraries"]:
                for size in filters:
                    if not any(row["library"] == library and int(row["filter"]) == size for row in plot_rows):
                        plot_rows.append({"library": library, "filter": size, "time_ms": "nan"})
            PLOT.make_plot(plot_rows, output / "figure.pdf", spec["evaluation"], policy_skipped=policy_skipped)
            metadata["plot"] = "figure.pdf"
        except PLOT.Pending as error:
            metadata["plot_pending"] = str(error)
    MEASURE.TABLE3.atomic_json(output / "metadata.json", metadata)
    MEASURE.TABLE3.atomic_json(output / "coverage.json", {"status": state, "missing": missing,
                             "cudnn_mode": metadata["cudnn_mode"], "policy_skipped": policy_skipped,
                             "coverage_complete": metadata["coverage_complete"], "full_paper_comparison": False})
    return {"status": state, "missing": missing, "cudnn_mode": metadata["cudnn_mode"],
            "policy_skipped": policy_skipped, "coverage_complete": metadata["coverage_complete"],
            "full_paper_comparison": False}


def assemble(parts, output, profile):
    return assemble_sweep(parts, output, profile, MEASURE.FIGURE4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts, args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
