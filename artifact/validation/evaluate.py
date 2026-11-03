#!/usr/bin/env python3
"""Run and validate the existing FFT/API correctness suite."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)


def expected_cases(profile):
    """The native suite's documented inputs, tail batches and cache rechecks."""
    batch = 16 if profile == "smoke" else 64
    inputs = ("gaussian_rms1",) if profile == "smoke" else (
        "gaussian_rms1", "offset_impulse", "complex_tone", "constant")
    bpb = {64: 16, 128: 8, 256: 16, 512: 2, 1024: 1, 2048: 1, 4096: 1}
    cases = Counter()
    for precision in ("fp16", "tf32"):
        for n in SIZES:
            tails = {1} if profile == "smoke" else {1, bpb[n] - 1, bpb[n] + 1}
            for direction in ("forward", "inverse"):
                for kind in inputs:
                    cases[("direct_core_guarded", precision, n, direction, kind, batch)] += 1
                    for count in {batch} | {t for t in tails if t > 0}:
                        cases[("public_global", precision, n, direction, kind, count)] += 1
        for direction in ("forward", "inverse"):
            cases[("public_global", precision, 64, direction, "gaussian_rms1", 1)] += 1
    return cases


def validate(rows, profile):
    expected = 88 if profile == "smoke" else 500
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} correctness cases, found {len(rows)}")
    observed = Counter((r["path"], r["precision"], int(r["N"]), r["direction"],
                        r["input"], int(r["batch"])) for r in rows)
    if observed != expected_cases(profile):
        raise ValueError("Input/path/batch coverage does not match the complete correctness suite")
    groups = defaultdict(list)
    for row in rows:
        key = row["precision"], int(row["N"]), row["direction"]
        if key[0] not in ("fp16", "tf32") or key[1] not in SIZES or key[2] not in ("forward", "inverse"):
            raise ValueError(f"Unexpected FFT configuration: {key}")
        if row["path"] not in ("direct_core_guarded", "public_global") or int(row["batch"]) <= 0:
            raise ValueError("Invalid path or batch")
        error = float(row["relative_l2"])
        if row["pass"] != "1" or not math.isfinite(error) or not 0 <= error <= 0.005:
            raise ValueError(f"Numerical correctness failure: {key}")
        for name in ("nonfinite_outputs", "shared_guard_changed_max", "output_guard_changed"):
            if float(row[name]) != 0:
                raise ValueError(f"Finite-output or memory-guard failure: {key}, {name}")
        groups[key].append(row)
    required = {(p, n, d) for p in ("fp16", "tf32") for n in SIZES for d in ("forward", "inverse")}
    if set(groups) != required:
        raise ValueError("Missing size/precision/direction coverage")
    summary = []
    for (precision, n, direction), cases in sorted(groups.items()):
        if {r["path"] for r in cases} != {"direct_core_guarded", "public_global"}:
            raise ValueError("Missing direct-core or public-wrapper path")
        summary.append({"precision": precision, "N": n, "direction": direction,
                        "cases": len(cases), "max_relative_l2": max(float(r["relative_l2"]) for r in cases),
                        "nonfinite_outputs": 0, "guard_changes": 0, "passed": True})
    return summary


def capture(command):
    try:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
        return {"command": command, "exit_code": run.returncode,
                "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    command = [str(args.binary.resolve()), "--device", "0", "--csv", str(args.output / "data.csv")]
    if args.profile == "smoke":
        command.append("--quick")
    (args.output / "commands.txt").write_text(shlex.join(command) + "\n")
    paths = list((ROOT / "include").rglob("*.cuh")) + list((ROOT / "include").rglob("*.h"))
    paths += [ROOT / "sample/validation/fft_validation.cu", Path(__file__), Path(__file__).with_name("run.sh")]
    metadata = {"evaluation": "validation", "profile": args.profile,
                "scope": "Implementation correctness and guards, not a numbered paper performance result",
                "command": command, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "requested_logical_device": os.environ.get("THUNDERFFT_DEVICE", "0"),
                "previous_CUDA_VISIBLE_DEVICES": os.environ.get("THUNDERFFT_VALIDATION_PREVIOUS_CUDA_VISIBLE_DEVICES")
                  if os.environ.get("THUNDERFFT_VALIDATION_PREVIOUS_CUDA_VISIBLE_DEVICES_SET") else None,
                "git": capture(["git", "rev-parse", "HEAD"]),
                "worktree": capture(["git", "status", "--porcelain"]),
                "gpu_inventory": capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "reference": "cuFFT FP64 on exact represented input", "relative_l2_limit": 0.005,
                "full_paper_comparison": False}
    compilation = args.build_dir / "compile_commands.json"
    if compilation.is_file():
        metadata["compile_commands"] = [c for c in json.loads(compilation.read_text())
                                        if Path(c["file"]).name == "fft_validation.cu"]
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    try:
        with (args.output / "native.log").open("w") as log:
            run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        if run.returncode:
            raise ValueError(f"Native correctness program failed with exit {run.returncode}; see native.log")
        if "Host API validation: 18 checks passed" not in (args.output / "native.log").read_text():
            raise ValueError("Missing host API validation evidence")
        with (args.output / "data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        summary = validate(rows, args.profile)
        with (args.output / "table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        report = {"status": "passed", "cases": len(rows), "host_api_checks": 18,
                  "relative_l2_limit": 0.005, "nonfinite_outputs": 0, "guard_changes": 0,
                  "max_relative_l2": {p: max(float(r["relative_l2"]) for r in rows if r["precision"] == p)
                                      for p in ("fp16", "tf32")},
                  "scope": "Bounded inputs; passing is not an arbitrary-input accuracy guarantee"}
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        (args.output / "validation.json").write_text(json.dumps({"status": "failed", "error": str(error)}, indent=2) + "\n")
        raise
    (args.output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"FFT correctness: {len(rows)} cases and 18 host API checks passed")


if __name__ == "__main__":
    main()
