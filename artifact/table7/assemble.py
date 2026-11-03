#!/usr/bin/env python3
"""Assemble separately collected Table 7 CSVs without launching GPU work."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
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
utilization = measure.utilization
TABLE_FIELDS = measure.TABLE_FIELDS
native_identity = measure.native_identity


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


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
    timings, identities, sources, ignored_parts = {}, [], [], []
    contract = None
    for part in parts:
        part = Path(part)
        meta = read_json(part / "metadata.json")
        if (part / "status.json").exists():
            state = read_json(part / "status.json")
            if state.get("status") != "completed":
                raise ValueError(f"Incomplete or failed part: {part}")
        this_contract = metadata_contract(meta, profile)
        selected = meta.get("coverage", {}).get("measured_variants", [])
        if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected) or any(v not in variants for v in selected):
            raise ValueError("Missing or invalid measured-variant coverage")
        phase = meta.get("phase", "both" if any(part.glob("*.profile.json")) else "timing")
        if phase not in ("timing", "profile", "both"):
            raise ValueError("Unknown Table 7 collection phase")
        if phase == "profile":
            ignored_parts.append({"part": str(part.resolve()), "phase": phase, "variants": selected,
                                  "reason": "Profiler counters do not measure the operation-count/timing utilization used by Table 7; collect timing CSVs",
                                  "metadata_sha256": hashlib.sha256((part / "metadata.json").read_bytes()).hexdigest()})
            continue
        if contract is not None and this_contract != contract:
            raise ValueError("Cannot combine Table 7 parts with different settings, source hashes, or binary hashes")
        contract = this_contract
        settings = contract["settings"]
        data = canonical_data(part / "data.csv", selected, settings)
        if set(data) != set(selected):
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
        source = {"part": str(part.resolve()), "phase": phase, "variants": selected,
                  "files": {str(path.relative_to(part)): hashlib.sha256(path.read_bytes()).hexdigest() for path in part_files}}
        if phase == "both":
            source["ignored_profile_counters"] = {
                "reason": "Utilization is recomputed from the timing CSV and operation count, not profiler counters",
                "files": sorted(path.name for path in part.iterdir()
                                if path.name.endswith((".profile.json", ".profile.log", ".ncu.csv")))}
        sources.append(source)
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Cannot combine Table 7 parts from different GPUs, drivers, runtimes, or compilers")
    missing, table, raw_rows, calculated = [], [], [], []
    for variant, specification in variants.items():
        row = dict.fromkeys(TABLE_FIELDS, "")
        row.update(variant=variant, N=64, batch=contract["settings"]["batch"] if contract else "", precision="tf32",
                   paper_identity="unconfirmed", timing_status="missing", utilization_status="missing")
        if variant in timings:
            rows = timings[variant]
            raw_rows.extend(dict(r, passed=int(r["passed"])) for r in rows)
            times = [r["compute_us"] for r in rows]
            median_us = statistics.median(times)
            row.update(median_time_ms=median_us / 1000, min_time_ms=min(times) / 1000, max_time_ms=max(times) / 1000,
                       max_transform_relative_l2=max(r["max_transform_relative_l2"] for r in rows), timing_status="collected")
            row.update(utilization.calculate(variant, contract["settings"]["batch"], median_us, identities[0]["name"]))
            if row["utilization_status"] == "calculated":
                calculated.append(variant)
            else:
                missing.append({"item": f"{variant}.utilization",
                                "reason": f"Hardware peak reference is not confirmed for {identities[0]['name']} and tf32; achieved throughput is available"})
        else:
            reason = specification.get("reason", "Timing CSV has not been collected") if specification["status"] != "available" else "Timing CSV has not been collected"
            missing.append({"item": f"{variant}.timing", "reason": reason})
            missing.append({"item": f"{variant}.utilization", "reason": "Utilization requires a collected timing CSV and a confirmed hardware peak reference"})
        table.append(row)
    if "radix16" in timings:
        baseline = statistics.median(r["compute_us"] for r in timings["radix16"])
        for row in table:
            if row["median_time_ms"] != "":
                row["radix16_speedup"] = baseline / (1000 * row["median_time_ms"])
    status = "complete" if not missing else "partial" if timings else "pending"
    result = {"evaluation": "table7", "profile": profile, "status": status, "missing": missing,
              "full_paper_comparison": False, "paper_identity": "unconfirmed", "parts": sources, "ignored_parts": ignored_parts,
              "measurement_contract": contract, "gpu": identities[0] if identities else None,
              "utilization": {"method": utilization.METHOD, "reference": utilization.REFERENCE,
                              "calculation_sha256": hashlib.sha256((HERE / "utilization.py").read_bytes()).hexdigest(),
                              "timing_basis": "median compute_us=(t_2R_us-t_R_us)/R; no second division by R"},
              "collected": {"timing": list(timings), "utilization": calculated}}
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
