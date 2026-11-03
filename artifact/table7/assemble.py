#!/usr/bin/env python3
"""Assemble separately collected Table 7 CSVs without launching GPU work."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = _module("table7_assembly_measure", "measure.py")
profiler = _module("table7_assembly_profiler", "collect_profile.py")
TABLE_FIELDS = ("variant", "N", "batch", "precision", "median_time_ms", "min_time_ms", "max_time_ms",
                "max_transform_relative_l2", "utilization_pct", "utilization_metric", "radix16_speedup",
                "timing_status", "utilization_status", "paper_identity")


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def native_identity(path):
    text = path.read_text()
    uuid = re.findall(r"^gpu_uuid=(GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})$", text, re.M)
    details = re.findall(r"^gpu=(.*?) device=0 sm=(\d+\.\d+) sm_count=(\d+) driver=(\d+) runtime=(\d+) cuda_compiler=(\S+)", text, re.M)
    if len(uuid) != 1 or len(details) != 1:
        raise ValueError(f"Missing or ambiguous native GPU identity: {path}")
    return dict(zip(("uuid", "name", "sm", "sm_count", "driver", "runtime", "compiler"), (uuid[0], *details[0])))


def validate_profile_log(path, variant, batch):
    text = path.read_text()
    expected = {"bounded_complex_gaussian", "complex_tone"}
    checks = re.findall(r"^validation_input=(\S+) ([^\n]+)$", text, re.M)
    if len(checks) != 2 or {key for key, _ in checks} != expected:
        raise ValueError(f"Expected both native correctness checks in {path}")
    for _, line in checks:
        fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
        exact = {"validation_batch": "64", "R": "1", "reference": "cpu_fp64_dft", "passed": "1",
                 "nonfinite_outputs": "0", "output_guard_changed": "0"}
        if any(fields.get(key) != value for key, value in exact.items()):
            raise ValueError(f"Profile correctness or guard gate failed: {path}")
        for key in ("relative_l2", "max_transform_relative_l2"):
            value = float(fields.get(key, "nan"))
            if not math.isfinite(value) or not 0 <= value <= 0.005:
                raise ValueError(f"Profile numerical gate failed: {path}")
    config = re.findall(r"^variant=(\S+) ([^\n]+)$", text, re.M)
    if len(config) != 1 or config[0][0] != variant:
        raise ValueError("Profile variant does not match its sidecar")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", config[0][1]))
    if any(fields.get(k) != v for k, v in {"N": "64", "batch": str(batch), "precision": "tf32", "R": "32", "trials": "1", "mode": "profile"}.items()):
        raise ValueError("Profile native configuration mismatch")
    if text.count("profile_launch_complete=1 profile_R=32 profile_warmups=0 profile_timing_rows=0") != 1:
        raise ValueError("Profile launch completion is missing")


def metadata_contract(meta, profile):
    if meta.get("evaluation") != "table7" or meta.get("profile") != profile:
        raise ValueError("Table 7 evaluation/profile mismatch")
    if meta.get("status") != "completed_declared_subset":
        raise ValueError("Only successfully completed Table 7 parts can be assembled")
    settings = meta.get("settings", {})
    for key in ("batch", "R", "trials"):
        if type(settings.get(key)) is not int or not 0 < settings[key] <= 1000000:
            raise ValueError(f"Invalid Table 7 {key} setting")
    if settings["batch"] % 16 or settings.get("precision") != "TF32":
        raise ValueError("Invalid Table 7 batch or precision")
    if profile == "paper" and (settings["batch"] != 65536 or settings["R"] != 1000):
        raise ValueError("Paper-sized protocol requires batch65536 and R1000")
    hashes = meta.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()):
        raise ValueError("Missing Table 7 source hashes")
    binary = meta.get("binary_sha256", "")
    if not isinstance(binary, str) or not re.fullmatch(r"[0-9a-f]{64}", binary):
        raise ValueError("Missing Table 7 binary hash")
    return {"settings": settings, "source_sha256": hashes, "binary_sha256": binary}


def canonical_data(path, variants, settings):
    reader = csv.DictReader(io.StringIO(path.read_text()))
    if tuple(reader.fieldnames or ()) != measure.RAW_FIELDS:
        raise ValueError(f"Invalid data.csv header: {path}")
    groups = {}
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("Malformed data.csv")
        if row["variant"] not in variants:
            raise ValueError("data.csv contains an undeclared variant")
        # Older exports wrote Python's bool spelling; native raw still requires 1.
        if row["passed"] == "True":
            row["passed"] = "1"
        groups.setdefault(row["variant"], []).append(row)
    parsed = {}
    for variant, rows in groups.items():
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=measure.RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        parsed[variant] = sorted(measure.parse_raw(stream.getvalue(), variant, settings["batch"], settings["R"], settings["trials"]), key=lambda row: row["trial"])
    return parsed


def assemble(parts: list[Path], output: Path, profile: str) -> dict:
    if profile not in ("smoke", "paper"):
        raise ValueError("Unknown Table 7 protocol profile")
    manifest = read_json(HERE / "variants.json")
    variants = {entry["id"]: entry for entry in manifest["variants"]}
    timings, counters, identities, sources = {}, {}, [], []
    contract = None
    profiling_contract = None
    for part in parts:
        part = Path(part)
        meta = read_json(part / "metadata.json")
        if (part / "status.json").exists():
            state = read_json(part / "status.json")
            if state.get("status") != "completed":
                raise ValueError(f"Incomplete or failed part: {part}")
        this_contract = metadata_contract(meta, profile)
        if contract is not None and this_contract != contract:
            raise ValueError("Cannot combine Table 7 parts with different settings, source hashes, or binary hashes")
        contract = this_contract
        settings = contract["settings"]
        selected = meta.get("coverage", {}).get("measured_variants", [])
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or any(v not in variants for v in selected):
            raise ValueError("Missing or invalid measured-variant coverage")
        phase = meta.get("phase", "both" if any(part.glob("*.profile.json")) else "timing")
        if phase not in ("timing", "profile", "both"):
            raise ValueError("Unknown Table 7 collection phase")
        data = canonical_data(part / "data.csv", selected, settings)
        if set(data) != (set(selected) if phase in ("timing", "both") else set()):
            raise ValueError("Timing CSV coverage does not match the declared phase")
        part_files = [part / "metadata.json", part / "data.csv"]
        for variant in selected:
            if variant in data:
                raw_file = part / f"{variant}.csv"
                rows = sorted(measure.parse_raw(raw_file.read_text(), variant, settings["batch"], settings["R"], settings["trials"]), key=lambda row: row["trial"])
                if rows != data[variant]:
                    raise ValueError("Raw and exported timing CSV disagree")
                if variant in timings and rows != timings[variant]:
                    raise ValueError(f"Conflicting timing measurements for {variant}; choose a single part")
                timings[variant] = rows
                log = part / f"{variant}.log"
                identities.append(native_identity(log))
                part_files.extend((raw_file, log))
            if phase in ("profile", "both"):
                record_file = part / f"{variant}.profile.json"
                record = read_json(record_file)
                if (record.get("exit_code") != 0 or record.get("variant") != variant or record.get("batch") != settings["batch"]
                        or record.get("R") != 32 or record.get("binary_sha256") != contract["binary_sha256"]
                        or record.get("clock_control") != "none" or record.get("cache_control") != "none"):
                    raise ValueError("Invalid profiling provenance or configuration")
                command = record.get("argv", [])
                if (not isinstance(command, list) or command[-5:] != [variant, str(settings["batch"]), "32", "1", "--profile"]
                        or not isinstance(record.get("profiler_version"), str) or not record["profiler_version"].strip()):
                    raise ValueError("Missing profiling command or tool version")
                for option, expected in (("--launch-skip", "2"), ("--launch-count", "1"), ("--clock-control", "none"), ("--cache-control", "none")):
                    if command.count(option) != 1 or command.index(option) + 1 >= len(command) or command[command.index(option) + 1] != expected:
                        raise ValueError("Profiling launch selection or controls mismatch")
                raw_file = part / f"{variant}.ncu.csv"
                measured = profiler.parse_metric(raw_file.read_text(), record.get("metric"))
                if measured != record.get("measurement"):
                    raise ValueError("Raw and exported utilization measurements disagree")
                expected_kernel = "fft_kernel_radix64_batch16_branch" if variant == "register_branch" else "fft_kernel_radix64_batch16"
                if expected_kernel not in measured["kernel"] or (variant == "register_uniform" and "fft_kernel_radix64_batch16_branch" in measured["kernel"]):
                    raise ValueError("Profile counter belongs to the wrong kernel")
                counter = {k: measured[k] for k in ("metric", "value", "kernel")}
                if variant in counters and counters[variant] != counter:
                    raise ValueError(f"Conflicting utilization measurements for {variant}; choose a single part")
                current_profile = {"metric": measured["metric"], "profiler_version": record["profiler_version"], "R": 32}
                if profiling_contract is not None and current_profile != profiling_contract:
                    raise ValueError("Cannot combine different utilization metrics or profiler versions")
                profiling_contract = current_profile
                counters[variant] = counter
                log = part / f"{variant}.profile.log"
                validate_profile_log(log, variant, settings["batch"])
                identities.append(native_identity(log))
                part_files.extend((record_file, raw_file, log))
        sources.append({"part": str(part.resolve()), "phase": phase, "variants": selected,
                        "files": {str(path.relative_to(part)): hashlib.sha256(path.read_bytes()).hexdigest() for path in part_files}})
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Cannot combine Table 7 parts from different GPUs, drivers, runtimes, or compilers")
    missing, table, raw_rows = [], [], []
    for variant, specification in variants.items():
        row = dict.fromkeys(TABLE_FIELDS, "")
        row.update(variant=variant, N=64, batch=contract["settings"]["batch"] if contract else "", precision="tf32",
                   paper_identity="unconfirmed", timing_status="missing", utilization_status="missing")
        if variant in timings:
            rows = timings[variant]
            raw_rows.extend(dict(r, passed=int(r["passed"])) for r in rows)
            times = [r["compute_us"] / 1000 for r in rows]
            row.update(median_time_ms=statistics.median(times), min_time_ms=min(times), max_time_ms=max(times),
                       max_transform_relative_l2=max(r["max_transform_relative_l2"] for r in rows), timing_status="collected")
        if variant in counters:
            row.update(utilization_pct=counters[variant]["value"], utilization_metric=counters[variant]["metric"], utilization_status="collected")
        for kind, values in (("timing", timings), ("utilization", counters)):
            if variant not in values:
                reason = specification.get("reason", "CSV has not been collected") if specification["status"] != "available" else "CSV has not been collected"
                missing.append({"item": f"{variant}.{kind}", "reason": reason})
        table.append(row)
    if "radix16" in timings:
        baseline = statistics.median(r["compute_us"] for r in timings["radix16"])
        for row in table:
            if row["median_time_ms"] != "":
                row["radix16_speedup"] = baseline / (1000 * row["median_time_ms"])
    status = "complete" if not missing else "partial" if timings or counters else "pending"
    result = {"evaluation": "table7", "profile": profile, "status": status, "missing": missing,
              "full_paper_comparison": False, "paper_identity": "unconfirmed", "parts": sources,
              "measurement_contract": contract, "gpu": identities[0] if identities else None,
              "profiling_contract": profiling_contract,
              "collected": {"timing": list(timings), "utilization": list(counters)}}
    output.mkdir(parents=True, exist_ok=True)
    measure.write_csv(output / "data.csv", raw_rows, measure.RAW_FIELDS)
    measure.write_csv(output / "table.csv", table, TABLE_FIELDS)
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts, args.output, args.profile), indent=2))


if __name__ == "__main__":
    main()
