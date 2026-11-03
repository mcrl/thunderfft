#!/usr/bin/env python3
"""Build-independent selection and measurement orchestration for ablation."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RAW_FIELDS = ("variant", "N", "batch", "precision", "trial", "inside_repeats", "t_R_us", "t_2R_us",
              "compute_us", "relative_l2", "max_transform_relative_l2", "nonfinite_outputs", "output_guard_changed", "passed")


class MissingVariants(ValueError):
    pass


def select_variants(text, manifest):
    by_id = {v["id"]: v for v in manifest["variants"]}
    requested = list(by_id) if text is None or text == "all" else [v.strip() for v in text.split(",")]
    if not requested or any(not v for v in requested) or len(set(requested)) != len(requested):
        raise ValueError("variant selection must contain unique, nonempty IDs")
    unknown = set(requested) - set(by_id)
    if unknown:
        raise ValueError("unknown variants: " + ", ".join(sorted(unknown)))
    missing = [v for v in requested if by_id[v]["status"] != "available"]
    if missing:
        raise MissingVariants("Source not recovered: " + ", ".join(missing))
    return [by_id[v] for v in requested]


def parse_raw(text, variant, batch, repeats, trials):
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


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def setting(name, default):
    value = os.environ.get(name, str(default))
    if not value.isdecimal() or not 0 < int(value) <= 1000000:
        raise ValueError(f"{name} must be an integer in 1..1000000")
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
    coverage = {"scope": manifest["scope"], "requested": os.environ.get("THUNDERFFT_TABLE7_VARIANTS", "all"),
                "source_available": [v["id"] for v in manifest["variants"] if v["status"] == "available"],
                "source_missing": [v["id"] for v in manifest["variants"] if v["status"] != "available"],
                "full_paper_comparison": False, "gpu_validation": "not_run_by_selection_check"}
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n")
    try:
        variants = select_variants(os.environ.get("THUNDERFFT_TABLE7_VARIANTS"), manifest)
    except MissingVariants as error:
        print(f"PENDING [table7]: {error}. Select recovered candidates explicitly with THUNDERFFT_TABLE7_VARIANTS.", file=sys.stderr)
        return 78
    if args.check_selection:
        print("Ablation source selection available: " + ", ".join(v["id"] for v in variants))
        return 0
    if args.binary is None or args.build_dir is None:
        parser.error("--binary and --build-dir are required for measurement")
    batch = setting("THUNDERFFT_TABLE7_BATCH", 1024 if args.profile == "smoke" else 65536)
    repeats = setting("THUNDERFFT_TABLE7_REPEATS", 256 if args.profile == "smoke" else 1000)
    trials = setting("THUNDERFFT_TABLE7_TRIALS", 2 if args.profile == "smoke" else 3)
    if batch % 16:
        raise ValueError("batch must be a multiple of 16")
    if args.profile == "paper" and (batch != 65536 or repeats != 1000):
        raise ValueError("paper-sized ablation protocol requires batch65536 and R1000")
    sources = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".py", ".sh", ".json", ".txt") and "__pycache__" not in p.parts]
    metadata = {"evaluation": "table7", "profile": args.profile, "coverage": coverage,
                "settings": {"batch": batch, "R": repeats, "trials": trials, "precision": "TF32"},
                "source_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "git": capture(["git", "rev-parse", "HEAD"]), "worktree": capture(["git", "status", "--porcelain"]),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "previous_CUDA_VISIBLE_DEVICES": os.environ.get("TABLE7_PREVIOUS_CUDA_VISIBLE_DEVICES"),
                "requested_logical_device": os.environ.get("THUNDERFFT_DEVICE", "0"),
                "overrides": {k: v for k, v in os.environ.items() if k.startswith("THUNDERFFT_TABLE7_")},
                "gpu_inventory": capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": [], "full_paper_comparison": False,
                "paper_settings_unconfirmed": ["GPU", "precision", "original intermediate variants", "utilization metric"],
                "correctness": "separate R1 Gaussian/tone CPU-FP64-DFT comparison, per-transform relL2<=0.005 and output guards",
                "repeated_outputs_used_for_correctness": False}
    compilation = args.build_dir / "compile_commands.json"
    if compilation.is_file():
        metadata["compile_commands"] = json.loads(compilation.read_text())
    rows, summaries = [], []
    try:
        for variant in variants:
            identifier = variant["id"]
            command = [str(args.binary.resolve()), identifier, str(batch), str(repeats), str(trials)]
            metadata["commands"].append(command)
            (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            with (args.output / f"{identifier}.csv").open("w") as out, (args.output / f"{identifier}.log").open("w") as log:
                run = subprocess.run(command, stdout=out, stderr=log, timeout=180)
            if run.returncode:
                raise ValueError(f"{identifier} failed with exit {run.returncode}; see its log")
            measured = parse_raw((args.output / f"{identifier}.csv").read_text(), identifier, batch, repeats, trials)
            rows.extend(measured)
            times = [r["compute_us"] for r in measured]
            summaries.append({"variant": identifier, "N": 64, "batch": batch, "precision": "tf32",
                              "median_time_ms": statistics.median(times) / 1000,
                              "min_time_ms": min(times) / 1000, "max_time_ms": max(times) / 1000,
                              "max_transform_relative_l2": max(r["max_transform_relative_l2"] for r in measured),
                              "utilization_pct": "", "radix16_speedup": "", "paper_identity": "unconfirmed"})
            if os.environ.get("THUNDERFFT_TABLE7_NCU", "0") == "1":
                from collect_profile import collect, DEFAULT_METRIC
                counter = collect(args.binary, identifier, batch, args.output,
                                  os.environ.get("THUNDERFFT_NCU", "ncu"),
                                  os.environ.get("THUNDERFFT_TABLE7_METRIC", DEFAULT_METRIC))
                summaries[-1].update(utilization_pct=counter["value"], utilization_metric=counter["metric"])
        write_csv(args.output / "data.csv", rows)
        write_csv(args.output / "table.csv", summaries)
        (args.output / "commands.txt").write_text("\n".join(shlex.join(c) for c in metadata["commands"]) + "\n")
        metadata["status"] = "completed_declared_subset"
        metadata["coverage"]["measured_variants"] = [v["id"] for v in variants]
        metadata["coverage"]["gpu_validation"] = "selected variants passed separate R1 numerical gates"
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        metadata.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("Ablation candidate subset completed; missing original variants and profiler metrics remain explicit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
