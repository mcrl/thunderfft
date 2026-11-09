#!/usr/bin/env python3
"""Build-independent selection and measurement orchestration for ablation."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from artifact.progress import report as report_progress

_UTILIZATION_SPEC = importlib.util.spec_from_file_location("table7_utilization", HERE / "utilization.py")
utilization = importlib.util.module_from_spec(_UTILIZATION_SPEC)
_UTILIZATION_SPEC.loader.exec_module(utilization)
RAW_FIELDS = ("variant", "N", "batch", "precision", "trial", "inside_repeats", "t_R_us", "t_2R_us",
              "compute_us", "relative_l2", "max_transform_relative_l2", "nonfinite_outputs", "output_guard_changed", "passed")
TABLE_FIELDS = ("variant", "N", "batch", "precision", "median_time_ms", "min_time_ms", "max_time_ms",
                "max_transform_relative_l2", "tensor_flops_per_fft", "tensor_flops_per_batch",
                "achieved_tflops", "reference_peak_tflops", "utilization_pct", "utilization_metric",
                "reference_source", "radix16_speedup", "timing_status", "utilization_status", "paper_identity",
                "accuracy_status", "source_commit", "implementation_kind", "kernel", "source", "source_sha256",
                "binary_sha256", "trials", "warmups_per_repeat_count", "input_pattern", "timing_scope", "operation_coverage",
                "known_source_issues")


class MissingVariants(ValueError):
    pass


def manifest_entries(manifest):
    entries = manifest["variants"] + manifest.get("microbenchmarks", [])
    if len({entry["id"] for entry in entries}) != len(entries):
        raise ValueError("Manifest variant IDs must be unique")
    return entries


def select_variants(text, manifest, timing_only=False):
    by_id = {v["id"]: v for v in manifest_entries(manifest)}
    requested = list(by_id) if text is None or text == "all" else [v.strip() for v in text.split(",")]
    if not requested or any(not v for v in requested) or len(set(requested)) != len(requested):
        raise ValueError("variant selection must contain unique, nonempty IDs")
    unknown = set(requested) - set(by_id)
    if unknown:
        raise ValueError("unknown variants: " + ", ".join(sorted(unknown)))
    missing = [v for v in requested if by_id[v]["status"] != "available"]
    if missing:
        raise MissingVariants("Source not recovered: " + ", ".join(missing))
    unchecked = [v for v in requested if by_id[v].get("require_timing_only")]
    if unchecked and not timing_only:
        raise ValueError("Function-limited historical microbenchmarks require THUNDERFFT_TABLE7_TIMING_ONLY=1: "
                         + ", ".join(unchecked))
    return [by_id[v] for v in requested]


def variant_provenance(specification):
    """Source/function scope is independent of whether outputs were validated."""
    fields = ("source_commit", "implementation_kind", "kernel", "source")
    result = {key: specification.get(key, "") for key in fields}
    issues = specification.get("known_source_issues", [])
    result.update(paper_identity=specification.get("paper_identity", "unconfirmed"),
                  timing_scope=specification.get("timing_scope", "repeated_kernel_compute"),
                  operation_coverage=specification.get("operation_coverage", "not_declared"),
                  known_source_issues="; ".join(issues) if isinstance(issues, list) else issues)
    return result


def utilization_summary(variant, batch, median_us, gpu_name, timing_only=False):
    if variant not in utilization.FLOPS_PER_FFT:
        return {"utilization_status": "pending_operation_count",
                "reference_source": "Executed Tensor Core operation count has not been verified for this source"}
    result = utilization.calculate(variant, batch, median_us, gpu_name)
    if timing_only:
        result["utilization_status"] += "_unvalidated"
    return result


def parse_raw(text, variant, batch, repeats, trials, timing_only=False):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != RAW_FIELDS:
        raise ValueError("ablation CSV header does not match the native contract")
    rows, seen = [], set()
    for row in reader:
        if None in row or any(v is None for v in row.values()):
            raise ValueError("malformed ablation CSV row")
        if (row["variant"], int(row["N"]), int(row["batch"]), row["precision"], int(row["inside_repeats"])) != (variant, 64, batch, "tf32", repeats):
            raise ValueError("ablation CSV configuration mismatch")
        trial = int(row["trial"])
        if trial in seen or not 0 <= trial < trials:
            raise ValueError("duplicate or out-of-range ablation trial")
        seen.add(trial)
        for key in ("N", "batch", "trial", "inside_repeats", "nonfinite_outputs", "output_guard_changed"):
            row[key] = int(row[key])
        for key in ("t_R_us", "t_2R_us", "compute_us", "relative_l2", "max_transform_relative_l2"):
            row[key] = float(row[key])
        if not all(math.isfinite(row[k]) and row[k] > 0 for k in ("t_R_us", "t_2R_us", "compute_us")):
            raise ValueError("ablation timings must be finite and positive")
        if row["t_2R_us"] <= row["t_R_us"] or not math.isclose(row["compute_us"], (row["t_2R_us"] - row["t_R_us"]) / repeats, rel_tol=1e-4, abs_tol=1e-6):
            raise ValueError("invalid differential timing")
        if timing_only:
            if (not all(math.isnan(row[k]) for k in ("relative_l2", "max_transform_relative_l2"))
                    or row["nonfinite_outputs"] != -1 or row["output_guard_changed"] != -1
                    or row["passed"] != "0"):
                raise ValueError("unchecked timing requires NaN errors, -1 guard/finite sentinels and passed=0")
            row["passed"] = False
        else:
            if not all(math.isfinite(row[k]) and 0 <= row[k] <= 0.005 for k in ("relative_l2", "max_transform_relative_l2")):
                raise ValueError("ablation accuracy gate failed")
            if row["nonfinite_outputs"] or row["output_guard_changed"] or row["passed"] != "1":
                raise ValueError("ablation finite-output/guard/pass gate failed")
            row["passed"] = True
        rows.append(row)
    if seen != set(range(trials)):
        raise ValueError("missing timing trials")
    return rows


def capture(argv):
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"argv": argv, "exit_code": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def add_baseline_speedups(rows):
    """Populate only comparisons whose radix-16 baseline was collected."""
    baseline = next((row["median_time_ms"] for row in rows
                     if row["variant"] == "radix16" and row["timing_status"] == "collected"), None)
    if baseline is not None:
        for row in rows:
            if row["timing_status"] == "collected":
                row["radix16_speedup"] = baseline / row["median_time_ms"]


def requested_variants(manifest):
    explicit = os.environ.get("THUNDERFFT_TABLE7_VARIANTS")
    if explicit is None:
        timing_only = selected_timing_only()
        return ",".join(v["id"] for v in manifest_entries(manifest)
                        if v["status"] == "available" and (timing_only or not v.get("require_timing_only")))
    return explicit


def selected_phase():
    phase = os.environ.get("THUNDERFFT_TABLE7_PHASE")
    if os.environ.get("THUNDERFFT_TABLE7_NCU", "0") != "0" or phase not in (None, "timing"):
        raise ValueError("Table 7 utilization is calculated from operation count and timing against hardware peak; "
                         "use THUNDERFFT_TABLE7_PHASE=timing and unset THUNDERFFT_TABLE7_NCU. "
                         "Profiler collection is not a Table 7 utilization phase.")
    return "timing"


def selected_timing_only():
    value = os.environ.get("THUNDERFFT_TABLE7_TIMING_ONLY", "0")
    if value not in ("0", "1"):
        raise ValueError("THUNDERFFT_TABLE7_TIMING_ONLY must be 0 or 1")
    return value == "1"


def require_unchecked_native_log(path):
    text = path.read_text()
    accuracy = re.findall(r"(?:^|\s)accuracy_status=(\S+)", text)
    timing = re.findall(r"(?:^|\s)timing_only=(\S+)", text)
    if accuracy != ["unchecked"] or timing != ["1"]:
        raise ValueError("Native log must confirm accuracy_status=unchecked timing_only=1")


def native_identity(path):
    text = path.read_text()
    uuid = re.findall(r"^gpu_uuid=(GPU-[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12})$", text, re.M)
    details = re.findall(r"^gpu=(.*?) device=0 sm=(\d+\.\d+) sm_count=(\d+) driver=(\d+) runtime=(\d+) cuda_compiler=(\S+)", text, re.M)
    if len(uuid) != 1 or len(details) != 1:
        raise ValueError(f"Missing or ambiguous native GPU identity: {path}")
    return dict(zip(("uuid", "name", "sm", "sm_count", "driver", "runtime", "compiler"), (uuid[0], *details[0])))


def native_input_pattern(path):
    patterns = re.findall(r"(?:^|\s)timing_input=(\S+)", path.read_text())
    if len(patterns) > 1:
        raise ValueError("Native timing input pattern is ambiguous")
    return patterns[0] if patterns else "unknown_legacy_log"


def setting(name, default, maximum=1000000):
    value = os.environ.get(name, str(default))
    if not value.isdecimal() or not 0 < int(value) <= maximum:
        raise ValueError(f"{name} must be an integer in 1..{maximum}")
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--check-selection", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((HERE / "variants.json").read_text())
    selection = requested_variants(manifest)
    phase = selected_phase()
    timing_only = selected_timing_only()
    coverage = {"scope": manifest["scope"], "requested": selection or "all", "phase": phase,
                "source_available": [v["id"] for v in manifest_entries(manifest) if v["status"] == "available"],
                "source_missing": [v["id"] for v in manifest["variants"] if v["status"] != "available"],
                "full_paper_comparison": False, "gpu_validation": "not_run_by_selection_check",
                "timing_only": timing_only, "accuracy_status": "unchecked" if timing_only else "pending_validation"}
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    try:
        variants = select_variants(selection, manifest, timing_only=timing_only)
    except MissingVariants as error:
        print(f"PENDING [table7]: {error}. Select available candidates explicitly with THUNDERFFT_TABLE7_VARIANTS.", file=sys.stderr)
        return 78
    if args.check_selection:
        print("Ablation source selection available: " + ", ".join(v["id"] for v in variants))
        return 0
    if args.binary is None or args.build_dir is None:
        parser.error("--binary and --build-dir are required for measurement")
    batch = setting("THUNDERFFT_TABLE7_BATCH", 1024 if args.profile == "smoke" else 65536, maximum=1048576)
    repeats = setting("THUNDERFFT_TABLE7_REPEATS", 256 if args.profile == "smoke" else 1000)
    trials = setting("THUNDERFFT_TABLE7_TRIALS", 2 if args.profile == "smoke" else 3, maximum=100)
    if batch % 16:
        raise ValueError("batch must be a multiple of 16")
    if args.profile == "paper" and (batch != 65536 or repeats != 1000):
        raise ValueError("paper-sized ablation protocol requires batch65536 and R1000")
    sources = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".py", ".sh", ".json", ".txt") and "__pycache__" not in p.parts]
    metadata = {"evaluation": "table7", "profile": args.profile, "phase": phase, "coverage": coverage,
                "settings": {"batch": batch, "R": repeats, "trials": trials, "precision": "TF32",
                             "warmups_per_repeat_count": 1 if timing_only else 3},
                "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "git": capture(["git", "rev-parse", "HEAD"]), "worktree": capture(["git", "status", "--porcelain"]),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "previous_CUDA_VISIBLE_DEVICES": os.environ.get("TABLE7_PREVIOUS_CUDA_VISIBLE_DEVICES"),
                "requested_logical_device": os.environ.get("THUNDERFFT_DEVICE", "0"),
                "overrides": {k: v for k, v in os.environ.items() if k.startswith("THUNDERFFT_TABLE7_")},
                "gpu_inventory": capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": [], "full_paper_comparison": False,
                "timing_only": timing_only,
                "variant_provenance": {v["id"]: variant_provenance(v) for v in variants},
                "accuracy_status": "unchecked" if timing_only else "pending_validation",
                "paper_settings_unconfirmed": ["exact original radix16 and stage-wise radix4 implementations",
                                               "original repetition count", "exact paper identity of the candidates"],
                "utilization": {"method": utilization.METHOD, "reference": utilization.REFERENCE,
                                "calculation_sha256": hashlib.sha256((HERE / "utilization.py").read_bytes()).hexdigest(),
                                "timing_basis": "median compute_us=(t_2R_us-t_R_us)/R; no second division by R",
                                "accuracy_status": "unchecked" if timing_only else "pending_validation"},
                "correctness": ("Not checked: THUNDERFFT_TABLE7_TIMING_ONLY=1 explicitly selected"
                                if timing_only else "separate R1 Gaussian/tone CPU-FP64-DFT comparison, per-transform relL2<=0.005 and output guards"),
                "repeated_outputs_used_for_correctness": False}
    compilation = args.build_dir / "compile_commands.json"
    if compilation.is_file():
        metadata["compile_commands"] = json.loads(compilation.read_text())
    rows, summaries, identities = [], [], []
    try:
        for completed, variant in enumerate(variants):
            identifier = variant["id"]
            summary = dict.fromkeys(TABLE_FIELDS, "")
            summary.update(variant=identifier, N=64, batch=batch, precision="tf32",
                           accuracy_status="unchecked" if timing_only else "passed",
                           binary_sha256=metadata["binary_sha256"], trials=trials,
                           warmups_per_repeat_count=metadata["settings"]["warmups_per_repeat_count"],
                           source_sha256=metadata["source_sha256"].get(variant.get("source"), ""))
            summary.update(variant_provenance(variant))
            summaries.append(summary)
            command = [str(args.binary.resolve()), identifier, str(batch), str(repeats), str(trials)]
            if timing_only:
                command.append("--timing-only")
            metadata["commands"].append(command)
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            report_progress(completed, len(variants), unit="variants", detail=identifier)
            with (args.output / f"{identifier}.csv").open("w") as out, (args.output / f"{identifier}.log").open("w") as log:
                run = subprocess.run(command, stdout=out, stderr=log, timeout=180)
            if run.returncode:
                raise ValueError(f"{identifier} failed with exit {run.returncode}; see its log")
            measured = parse_raw((args.output / f"{identifier}.csv").read_text(), identifier, batch, repeats, trials,
                                 timing_only=timing_only)
            log_path = args.output / f"{identifier}.log"
            if timing_only:
                require_unchecked_native_log(log_path)
            identity = native_identity(log_path)
            input_pattern = native_input_pattern(log_path)
            if "input_pattern" in metadata["settings"] and metadata["settings"]["input_pattern"] != input_pattern:
                raise ValueError("Native timing input pattern changed between variants")
            metadata["settings"]["input_pattern"] = input_pattern
            summary["input_pattern"] = input_pattern
            if identities and identity != identities[0]:
                raise ValueError("Native GPU, driver, runtime, or compiler changed between variants")
            identities.append(identity)
            metadata["gpu"] = identity
            rows.extend(measured)
            times = [r["compute_us"] for r in measured]
            median_us = statistics.median(times)
            summary.update(median_time_ms=median_us / 1000,
                           min_time_ms=min(times) / 1000, max_time_ms=max(times) / 1000,
                           max_transform_relative_l2="" if timing_only else max(r["max_transform_relative_l2"] for r in measured),
                           timing_status="collected_unvalidated" if timing_only else "collected")
            summary.update(utilization_summary(identifier, batch, median_us, identity["name"], timing_only))
            report_progress(completed + 1, len(variants), unit="variants", detail=identifier)
        add_baseline_speedups(summaries)
        write_csv(args.output / "data.csv", [dict(r, passed=int(r["passed"])) for r in rows], RAW_FIELDS)
        write_csv(args.output / "table.csv", summaries, TABLE_FIELDS)
        write_csv(args.output / "kernel_comparison.csv", summaries, TABLE_FIELDS)
        (args.output / "commands.txt").write_text("\n".join(shlex.join(c) for c in metadata["commands"]) + "\n")
        metadata["status"] = "completed_timing_only" if timing_only else "completed_declared_subset"
        metadata["accuracy_status"] = "unchecked" if timing_only else "passed"
        metadata["utilization"]["accuracy_status"] = metadata["accuracy_status"]
        metadata["coverage"]["accuracy_status"] = metadata["accuracy_status"]
        metadata["coverage"]["measured_variants"] = [v["id"] for v in variants]
        metadata["coverage"]["gpu_validation"] = ("not_run_timing_only" if timing_only
                                                    else "selected variants passed separate R1 numerical gates")
        metadata["unit_kernel_comparison"] = {"available": True, "variants": [v["id"] for v in variants],
                                              "file": "kernel_comparison.csv", "full_paper_comparison": False}
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (args.output / "coverage.json").write_text(json.dumps(metadata["coverage"], indent=2) + "\n")
    print("Compute timings collected without accuracy checks; assembled rows retain unchecked status and function coverage"
          if timing_only else "Selected ablation candidates completed; utilization calculated from timing where the hardware peak is confirmed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
