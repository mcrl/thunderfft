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
import shutil
import statistics

HERE = Path(__file__).resolve().parent


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


measure = _module("table7_assembly_measure", "measure.py")
utilization = measure.utilization
projection = _module("table7_assembly_projection", "project_baseline.py")
ESTIMATE_FIELDS = ("estimated_time_ms", "estimated_utilization_pct", "speedup_vs_estimated_radix16")
TABLE_FIELDS = (*measure.TABLE_FIELDS, *ESTIMATE_FIELDS)
native_identity = measure.native_identity


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path}")
    return value


def metadata_contract(meta, profile):
    if meta.get("evaluation") != "table7" or meta.get("profile") != profile:
        raise ValueError("Table 7 evaluation/profile mismatch")
    if meta.get("status") not in ("completed_declared_subset", "completed_timing_only"):
        raise ValueError("Only successfully completed Table 7 parts can be assembled")
    timing_only = meta["status"] == "completed_timing_only"
    if timing_only:
        if meta.get("timing_only") is not True or meta.get("accuracy_status") != "unchecked":
            raise ValueError("Timing-only metadata must explicitly declare unchecked accuracy")
    elif meta.get("timing_only", False) or meta.get("accuracy_status", "passed") != "passed":
        raise ValueError("Validated part metadata has inconsistent accuracy status")
    settings = meta.get("settings", {})
    for key, maximum in (("batch", 1048576), ("R", 1000000), ("trials", 100)):
        if type(settings.get(key)) is not int or not 0 < settings[key] <= maximum:
            raise ValueError(f"Invalid Table 7 {key} setting")
    if settings["batch"] % 16 or settings.get("precision") != "TF32":
        raise ValueError("Invalid Table 7 batch or precision")
    warmups = settings.get("warmups_per_repeat_count")
    if warmups is not None and (type(warmups) is not int or not 0 <= warmups <= 100):
        raise ValueError("Invalid Table 7 warmup setting")
    if profile == "paper" and (settings["batch"] != 65536 or settings["R"] != 1000):
        raise ValueError("Paper-sized protocol requires batch65536 and R1000")
    hashes = meta.get("source_sha256")
    if not isinstance(hashes, dict) or not hashes or not all(isinstance(v, str) and re.fullmatch(r"[0-9a-f]{64}", v) for v in hashes.values()):
        raise ValueError("Missing Table 7 source hashes")
    binary = meta.get("binary_sha256", "")
    if not isinstance(binary, str) or not re.fullmatch(r"[0-9a-f]{64}", binary):
        raise ValueError("Missing Table 7 binary hash")
    return {"settings": settings, "source_sha256": hashes, "binary_sha256": binary,
            "timing_only": timing_only, "accuracy_status": "unchecked" if timing_only else "passed",
            "compile_commands": meta.get("compile_commands")}


def canonical_data(path, variants, settings, timing_only=False):
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
        parsed[variant] = sorted(measure.parse_raw(stream.getvalue(), variant, settings["batch"], settings["R"], settings["trials"],
                                                 timing_only=timing_only), key=lambda row: row["trial"])
    return parsed


def same_csv_rows(left, right):
    """The explicit NaN unchecked sentinel is equal across independent parses."""
    def normalize(rows):
        return [{key: "unchecked_nan" if isinstance(value, float) and math.isnan(value) else value
                 for key, value in row.items()} for row in rows]
    return normalize(left) == normalize(right)


def apply_baseline_projection(table, missing, output, table6, profile, identities):
    """Populate declared estimates without changing measured timing fields."""
    def unavailable(reason):
        return {"status": "unavailable", "reason": reason, "gpu_work_performed": False}

    if profile != "paper":
        return unavailable("The tcFFT projection requires the A100 paper workload with batch 65536.")
    if table6 is None:
        return unavailable("Collect Table 6 tcFFT in the same collection to estimate the radix-16 baseline.")
    if identities and not projection.is_a100(identities[0]["name"]):
        return unavailable("The Table 6 tcFFT projection applies only to A100 Table 7 results.")
    baseline = next(row for row in table if row["variant"] == "radix16")
    if baseline["timing_status"] != "missing":
        return unavailable("An existing radix-16 measurement takes precedence over the estimate.")
    source = Path(table6)
    metadata = read_json(source / "metadata.json")
    if (metadata.get("evaluation") != "table6" or metadata.get("profile") != "paper"
            or metadata.get("assembly_schema") != "table6-validated-parts-v1"):
        raise ValueError("Table 7 projection requires a validated paper Table 6 assembly")
    for name in ("table.csv", "data.csv"):
        if metadata.get("outputs", {}).get(name) != projection.file_record(source / name)["sha256"]:
            raise ValueError("Table 6 projection input hash disagrees with assembly metadata")
    _, candidates = projection.read_csv(source / "table.csv")
    candidates = [row for row in candidates if row.get("library") == "tcFFT"]
    if not candidates or all(row.get("status") == "pending" for row in candidates):
        return unavailable("Table 6 tcFFT has not been collected; no reference timing is substituted.")
    snapshot = output / "inputs/tcfft-table6"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("table.csv", "data.csv", "metadata.json"):
        shutil.copy2(source / name, snapshot / name)
    sources, records = projection.table6_sources(snapshot / "table.csv", libraries=("tcFFT",))
    estimate = projection.project(sources[0])
    baseline.update(batch=projection.BATCH, timing_status="estimated", utilization_status="estimated",
                    accuracy_status="not_applicable_projection", implementation_kind="model_projection",
                    operation_coverage="theoretical_32Nlog2N_work_model",
                    estimated_time_ms=estimate["projected_time_us"] / 1000,
                    estimated_utilization_pct=estimate["projected_utilization_pct"],
                    speedup_vs_estimated_radix16=1.0)
    for row in table:
        if (row["timing_status"] == "collected" and row["accuracy_status"] == "passed"
                and row["N"] == 64 and row["batch"] == projection.BATCH and row["precision"] == "tf32"):
            row["speedup_vs_estimated_radix16"] = baseline["estimated_time_ms"] / row["median_time_ms"]
    missing[:] = [item for item in missing if item["item"] not in ("radix16.timing", "radix16.utilization")]
    records.append(projection.file_record(snapshot / "metadata.json"))
    return {"status": "estimated", "estimate": estimate, "gpu_work_performed": False,
            "assumption": "Source and target retain the same fraction of their respective dense hardware peaks.",
            "formula": "estimated_time_us = source_time_us * (12288 / 65536) * (312 / 156) = source_time_us * 0.375",
            "source_operation_count": "2 complex matrix products * 4 WMMA per product * (2*16*16*16) = 65536 FLOPs/FFT",
            "target_work_model": "32*N*log2(N) = 12288 FLOPs at N=64; theoretical normalization, not an executed radix-16 kernel",
            "inputs": [{"path": str(Path(record["path"]).relative_to(output.resolve())),
                        "sha256": record["sha256"]} for record in records],
            "calculation_sha256": hashlib.sha256((HERE / "project_baseline.py").read_bytes()).hexdigest(),
            "full_paper_comparison": False}


def assemble(parts: list[Path], output: Path, profile: str, table6=None) -> dict:
    if profile not in ("smoke", "paper"):
        raise ValueError("Unknown Table 7 protocol profile")
    manifest = read_json(HERE / "variants.json")
    variants = {entry["id"]: entry for entry in measure.manifest_entries(manifest)}
    canonical_ids = [entry["id"] for entry in manifest["variants"]]
    timings, identities, sources, ignored_parts, ignored_variants = {}, [], [], [], []
    contract, variant_contracts, provenance_by_variant, protocols = None, {}, {}, []
    for part in parts:
        part = Path(part)
        meta = read_json(part / "metadata.json")
        if (part / "status.json").exists():
            state = read_json(part / "status.json")
            if state.get("status") != "completed":
                raise ValueError(f"Incomplete or failed part: {part}")
        this_contract = metadata_contract(meta, profile)
        timing_only = this_contract["timing_only"]
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
        eligible = []
        for variant in selected:
            specification = variants[variant]
            if specification.get("require_timing_only") and not timing_only:
                raise ValueError(f"Function-limited historical microbenchmark {variant} requires timing-only input")
            reason = None
            if specification["status"] != "available":
                reason = specification.get("reason", "The original source is unavailable")
            else:
                source = specification["source"]
                expected_hash = hashlib.sha256((HERE / source).read_bytes()).hexdigest()
                if this_contract["source_sha256"].get(source) != expected_hash:
                    reason = (f"The recorded source hash does not match the current selected source {source}; "
                              "collect this candidate again before using its kernel comparison row")
            if reason:
                ignored_variants.append({"part": str(part.resolve()), "variant": variant,
                                         "reason": reason,
                                         "metadata_sha256": hashlib.sha256((part / "metadata.json").read_bytes()).hexdigest()})
            else:
                eligible.append(variant)
        if not eligible:
            ignored_parts.append({"part": str(part.resolve()), "phase": phase, "variants": selected,
                                  "reason": "No timing rows correspond to available current recovered candidates"})
            continue
        settings = this_contract["settings"]
        common = {"N": 64, **{key: settings[key] for key in ("batch", "R", "precision")}}
        if contract is not None and common != contract:
            raise ValueError("Cannot combine Table 7 parts with different batch, R, or precision")
        contract = common
        data = canonical_data(part / "data.csv", selected, settings, timing_only=timing_only)
        if set(data) != set(selected):
            raise ValueError("Timing CSV coverage does not match the declared phase")
        part_files = [part / "metadata.json", part / "data.csv"]
        for variant in eligible:
            if variant in data:
                raw_file = part / f"{variant}.csv"
                rows = sorted(measure.parse_raw(raw_file.read_text(), variant, settings["batch"], settings["R"], settings["trials"],
                                                timing_only=timing_only), key=lambda row: row["trial"])
                if not same_csv_rows(rows, data[variant]):
                    raise ValueError("Raw and exported timing CSV disagree")
                provenance = measure.variant_provenance(variants[variant])
                recorded = meta.get("variant_provenance", {}).get(variant)
                if recorded is not None and recorded != provenance:
                    raise ValueError(f"Recorded source/function coverage does not match the selected candidate {variant}")
                if variant in timings and not same_csv_rows(rows, timings[variant]):
                    raise ValueError(f"Conflicting timing measurements for {variant}; choose a single part")
                if variant in variant_contracts and (this_contract != variant_contracts[variant]
                                                    or provenance != provenance_by_variant[variant]):
                    raise ValueError(f"Conflicting source or measurement provenance for {variant}; choose a single part")
                timings[variant] = rows
                variant_contracts[variant] = this_contract
                provenance_by_variant[variant] = provenance
                log = part / f"{variant}.log"
                if timing_only:
                    measure.require_unchecked_native_log(log)
                logged_input = measure.native_input_pattern(log)
                if settings.get("input_pattern", "unknown_legacy_log") != logged_input:
                    raise ValueError("Recorded timing input pattern disagrees with the native log")
                identities.append(native_identity(log))
                part_files.extend((raw_file, log))
        source = {"part": str(part.resolve()), "phase": phase, "variants": eligible,
                  "measurement_contract": this_contract,
                  "variant_provenance": {variant: provenance_by_variant[variant] for variant in eligible},
                  "files": {str(path.relative_to(part)): hashlib.sha256(path.read_bytes()).hexdigest() for path in part_files}}
        protocols.append({"settings": settings, "compile_commands": this_contract["compile_commands"]})
        if phase == "both":
            source["ignored_profile_counters"] = {
                "reason": "Utilization is recomputed from the timing CSV and operation count, not profiler counters",
                "files": sorted(path.name for path in part.iterdir()
                                if path.name.endswith((".profile.json", ".profile.log", ".ncu.csv")))}
        sources.append(source)
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Cannot combine Table 7 parts from different GPUs, drivers, runtimes, or compilers")
    missing, table, raw_rows, calculated = [], [], [], []
    # Supplemental microbenchmarks appear only when collected. Their absence
    # must not turn optional measurements into missing original paper stages.
    displayed_ids = canonical_ids + [variant for variant in variants if variant not in canonical_ids and variant in timings]
    for variant in displayed_ids:
        specification = variants[variant]
        row = dict.fromkeys(TABLE_FIELDS, "")
        row.update(variant=variant, N=64, batch=contract["batch"] if contract else "", precision="tf32",
                   accuracy_status="not_collected", timing_status="missing", utilization_status="missing")
        row.update(measure.variant_provenance(specification))
        if variant in timings:
            rows = timings[variant]
            variant_contract = variant_contracts[variant]
            timing_only = variant_contract["timing_only"]
            settings = variant_contract["settings"]
            raw_rows.extend(dict(r, passed=int(r["passed"])) for r in rows)
            times = [r["compute_us"] for r in rows]
            median_us = statistics.median(times)
            row.update(median_time_ms=median_us / 1000, min_time_ms=min(times) / 1000, max_time_ms=max(times) / 1000,
                       max_transform_relative_l2="" if timing_only else max(r["max_transform_relative_l2"] for r in rows),
                       timing_status="collected_unvalidated" if timing_only else "collected",
                       accuracy_status=variant_contract["accuracy_status"],
                       source_sha256=variant_contract["source_sha256"].get(specification.get("source"), ""),
                       binary_sha256=variant_contract["binary_sha256"], trials=settings["trials"],
                       warmups_per_repeat_count=settings.get("warmups_per_repeat_count", ""),
                       input_pattern=settings.get("input_pattern", "unknown_legacy_log"))
            row.update(measure.utilization_summary(variant, settings["batch"], median_us, identities[0]["name"], timing_only))
            if row["utilization_status"] in ("calculated", "calculated_unvalidated"):
                calculated.append(variant)
            else:
                reason = ("Executed Tensor Core operation count has not been verified for this source"
                          if row["utilization_status"] == "pending_operation_count" else
                          f"Hardware peak reference is not confirmed for {identities[0]['name']} and tf32; achieved throughput is available")
                missing.append({"item": f"{variant}.utilization",
                                "reason": reason})
        else:
            reason = specification.get("reason", "Timing CSV has not been collected") if specification["status"] != "available" else "Timing CSV has not been collected"
            missing.append({"item": f"{variant}.timing", "reason": reason})
            missing.append({"item": f"{variant}.utilization", "reason": "Utilization requires a collected timing CSV and a confirmed hardware peak reference"})
        table.append(row)
    measure.add_baseline_speedups(table)
    output.mkdir(parents=True, exist_ok=True)
    baseline_projection = apply_baseline_projection(table, missing, output, table6, profile, identities)
    status = "complete" if not missing else "partial" if timings or baseline_projection["status"] == "estimated" else "pending"
    accuracy_statuses = {value["accuracy_status"] for value in variant_contracts.values()}
    accuracy_status = (next(iter(accuracy_statuses)) if len(accuracy_statuses) == 1
                       else "mixed" if accuracy_statuses else "not_collected")
    result = {"evaluation": "table7", "profile": profile, "status": status, "missing": missing,
              "full_paper_comparison": False, "paper_identity": "unconfirmed", "parts": sources, "ignored_parts": ignored_parts,
              "ignored_variants": ignored_variants,
              "measurement_contract": contract, "variant_contracts": variant_contracts,
              "accuracy_status": accuracy_status,
              "baseline_projection": baseline_projection,
              "protocol_heterogeneous": bool(protocols) and any(protocol != protocols[0] for protocol in protocols[1:]),
              "gpu": identities[0] if identities else None,
              "utilization": {"method": utilization.METHOD, "reference": utilization.REFERENCE,
                              "calculation_sha256": hashlib.sha256((HERE / "utilization.py").read_bytes()).hexdigest(),
                              "timing_basis": "median compute_us=(t_2R_us-t_R_us)/R; no second division by R"},
              "collected": {"timing": list(timings), "utilization": calculated,
                            "validated": [variant for variant, value in variant_contracts.items() if not value["timing_only"]],
                            "unchecked": [variant for variant, value in variant_contracts.items() if value["timing_only"]]},
              "unit_kernel_comparison": {"available": bool(timings), "variants": list(timings),
                                         "file": "kernel_comparison.csv", "full_paper_comparison": False}}
    output.mkdir(parents=True, exist_ok=True)
    measure.write_csv(output / "data.csv", raw_rows, measure.RAW_FIELDS)
    measure.write_csv(output / "table.csv", table, TABLE_FIELDS)
    measure.write_csv(output / "kernel_comparison.csv",
                      [{field: row[field] for field in measure.TABLE_FIELDS} for row in table if row["variant"] in timings],
                      measure.TABLE_FIELDS)
    (output / "metadata.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", type=Path, nargs="*")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--table6", type=Path, help="Validated assembled Table 6 directory for the tcFFT baseline estimate")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts, args.output, args.profile, table6=args.table6), indent=2))


if __name__ == "__main__":
    main()
