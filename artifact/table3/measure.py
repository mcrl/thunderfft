#!/usr/bin/env python3
"""Collect selected ThunderFFT Table 3 configurations and retain their evidence."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
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


ROOT = Path(__file__).resolve().parents[2]
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
PRECISIONS = ("fp16", "tf32")
PLACEMENTS = ("register", "shared")
RAW_FIELDS = ("N", "precision", "placement", "batch", "inside_repeats", "trial",
              "t_R_us", "t_2R_us", "compute_us", "e2e_out_of_place_us",
              "single_fft_relL2", "shared_bytes")
FIELDS = ("gpu", "N", "batch", "placement", "library", "precision", "trial",
          "time_us", "e2e_out_of_place_us", "rel_l2", "passed", "inside_repeats",
          "t_R_us", "t_2R_us", "shared_bytes", "raw_csv")


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def capture(command):
    try:
        result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True,
                                timeout=15, check=False)
        return {"command": command, "exit_code": result.returncode,
                "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def source_hashes():
    paths = list((ROOT / "include").rglob("*.cuh"))
    paths += list((ROOT / "include").rglob("*.h"))
    paths += [ROOT / "CMakeLists.txt", ROOT / "sample/CMakeLists.txt",
              ROOT / "sample/bench/CMakeLists.txt", ROOT / "sample/bench/paper_fft.cu",
              ROOT / "sample/bench/baseline/thunderfft_bench.h",
              ROOT / "sample/bench/util/helper.h", ROOT / "artifact/common.sh",
              ROOT / "artifact/table5/select_device.py"]
    paths += list(Path(__file__).parent.glob("*.py"))
    paths += [Path(__file__).with_name("run.sh")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths)) if path.is_file()}


def build_evidence(build):
    evidence = {}
    for relative in ("CMakeCache.txt", "sample/bench/CMakeFiles/thff_paper_fft.dir/flags.make",
                     "sample/bench/CMakeFiles/thff_paper_fft.dir/link.txt"):
        source = build / relative
        if source.is_file():
            if relative == "CMakeCache.txt":
                names = ("CMAKE_BUILD_TYPE", "CMAKE_CUDA_COMPILER", "CMAKE_CUDA_FLAGS",
                         "CMAKE_CUDA_ARCHITECTURES", "THUNDERFFT_ARCHS", "CUDAToolkit_ROOT")
                evidence[relative] = "\n".join(line for line in source.read_text().splitlines()
                                               if any(line.startswith(name + ":") for name in names))
            else:
                evidence[relative] = source.read_text()
    commands = build / "compile_commands.json"
    if commands.is_file():
        evidence["compile_commands"] = [entry for entry in json.loads(commands.read_text())
                                        if Path(entry["file"]).name == "paper_fft.cu"]
    return evidence


def gpu_name(log):
    match = re.search(r"^GPU=(.+?) compute=(\d+)\.(\d+) batch=", log, re.MULTILINE)
    if not match:
        raise ValueError("native executable did not identify its CUDA device")
    name, major, minor = match.groups()
    family = re.search(r"\b(A100|H100|B200)\b", name)
    return family.group(1) if family else name, name, f"{major}.{minor}"


def gpu_uuid(log):
    entries = re.findall(r"^GPU_UUID=(.*)$", log, re.MULTILINE)
    if not entries:
        return None  # Historical logs predate native UUID reporting.
    if len(entries) != 1 or not re.fullmatch(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", entries[0]):
        raise ValueError("native executable reported an invalid or duplicate GPU UUID")
    return entries[0]


def parse_raw(text, *, n, precision, placement, batch, repeats, trials, gpu, raw_csv):
    """Validate the native CSV before treating any measurement as successful."""
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != RAW_FIELDS:
        raise ValueError("native CSV header does not match the versioned Table 3 contract")
    rows = []
    seen = set()
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("malformed native CSV row")
        expected = (n, precision, placement, batch, repeats)
        actual = (int(raw["N"]), raw["precision"], raw["placement"], int(raw["batch"]),
                  int(raw["inside_repeats"]))
        if actual != expected:
            raise ValueError(f"native CSV configuration mismatch: {actual} != {expected}")
        trial = int(raw["trial"])
        if trial in seen or not 0 <= trial < trials:
            raise ValueError("duplicate or out-of-range native trial")
        seen.add(trial)
        t_r, t_2r, compute, e2e = (float(raw[key]) for key in
                                   ("t_R_us", "t_2R_us", "compute_us", "e2e_out_of_place_us"))
        if not all(math.isfinite(value) and value > 0 for value in (t_r, t_2r, compute, e2e)):
            raise ValueError("all native timings must be positive and finite")
        if t_2r <= t_r or not math.isclose(compute, (t_2r - t_r) / repeats,
                                          rel_tol=1e-4, abs_tol=1e-6):
            raise ValueError("native differential timing is invalid or inconsistent")
        rel_l2 = float(raw["single_fft_relL2"])
        if not math.isfinite(rel_l2) or not 0 <= rel_l2 <= 0.005:
            raise ValueError("native single-FFT relative L2 gate failed")
        shared = int(raw["shared_bytes"])
        if shared <= 0:
            raise ValueError("invalid shared-memory workspace size")
        rows.append(dict(zip(FIELDS, (gpu, n, batch, placement, "ThunderFFT", precision, trial,
                         compute, e2e, rel_l2, "true", repeats, t_r, t_2r, shared, raw_csv))))
    if seen != set(range(trials)):
        raise ValueError(f"expected {trials} complete timing trials, got {len(seen)}")
    return rows


def summarize(rows, profile):
    targets = {}
    with Path(__file__).with_name("paper-table3-xfft-targets.csv").open() as stream:
        for target in csv.DictReader(stream):
            targets[(target["gpu"], int(target["N"]), target["precision"], target["placement"])] = float(target["target_us"])
    groups = {}
    for row in rows:
        key = row["gpu"], row["N"], row["precision"], row["placement"]
        groups.setdefault(key, []).append(row)
    results = []
    for key, samples in sorted(groups.items()):
        times = [row["time_us"] for row in samples]
        median = statistics.median(times)
        target = targets.get(key) if profile == "paper" and samples[0]["batch"] == 65536 else None
        results.append({"gpu": key[0], "N": key[1], "precision": key[2], "placement": key[3],
                        "library": "ThunderFFT", "batch": samples[0]["batch"],
                        "inside_repeats": samples[0]["inside_repeats"], "trials": len(samples),
                        "median_time_us": median, "min_time_us": min(times), "max_time_us": max(times),
                        "median_e2e_out_of_place_us": statistics.median(row["e2e_out_of_place_us"] for row in samples),
                        "max_rel_l2": max(row["rel_l2"] for row in samples), "passed": "true",
                        "paper_thunderfft_target_us": target if target is not None else "",
                        "relative_delta_from_paper": median / target - 1 if target is not None else "",
                        "cufftdx_comparison": "not_measured"})
    return results


def positive_env(name, default):
    value = os.environ.get(name, str(default))
    if not value.isdecimal() or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def selection_env(name, allowed, convert=str):
    """Keep split runs explicit; reject typos instead of silently skipping work."""
    text = os.environ.get(name)
    if text is None:
        return tuple(allowed)
    tokens = [value.strip() for value in text.split(",")]
    if not tokens or any(not value for value in tokens):
        raise ValueError(f"{name} must be a nonempty comma-separated list")
    values = tuple(convert(value) for value in tokens)
    if len(set(values)) != len(values) or not set(values) <= set(allowed):
        raise ValueError(f"{name} must contain unique values from {tuple(allowed)}")
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    settings = {"batch": positive_env("THUNDERFFT_TABLE3_BATCH", 1024 if args.profile == "smoke" else 65536),
                "repeats": positive_env("THUNDERFFT_TABLE3_REPEATS", 256 if args.profile == "smoke" else 1000),
                "trials": positive_env("THUNDERFFT_TABLE3_TRIALS", 2 if args.profile == "smoke" else 3),
                "timeout_seconds_per_configuration": positive_env("THUNDERFFT_TABLE3_TIMEOUT", 120)}
    if settings["batch"] % 16:
        parser.error("THUNDERFFT_TABLE3_BATCH must be a positive multiple of 16")
    if args.profile == "paper" and (settings["batch"] != 65536 or settings["repeats"] != 1000):
        parser.error("paper profile requires batch=65536 and R=1000; use smoke for custom checks")
    try:
        selection = {"N": selection_env("THUNDERFFT_TABLE3_N_LIST", SIZES, int),
                     "precisions": selection_env("THUNDERFFT_TABLE3_PRECISIONS", PRECISIONS),
                     "placements": selection_env("THUNDERFFT_TABLE3_PLACEMENTS", PLACEMENTS)}
    except ValueError as error:
        parser.error(str(error))
    selected_count = math.prod(len(values) for values in selection.values())
    metadata = {"schema_version": 1, "evaluation": "table3", "profile": args.profile,
                "started_at": now(), "status": "running", "settings": settings,
                "selection": selection,
                "scope": "ThunderFFT-only; cuFFTDx tuning and comparison excluded",
                "full_paper_comparison_complete": False,
                "coverage": {"expected_thunderfft_configurations": 28, "measured_thunderfft_configurations": 0,
                             "selected_thunderfft_configurations": selected_count,
                             "cufftdx_configurations": 0, "other_gpus": "not measured by this run"},
                "correctness": {"gate": "finite single-FFT output and aggregate relL2 <= 0.005",
                                "reference": "cuFFT FP64 on exact storage-rounded input",
                                "validation_batch": 64, "repeated_loop_output_used_for_correctness": False},
                "run_id": os.environ.get("THUNDERFFT_RUN_ID"),
                "requested_logical_device": os.environ.get("THUNDERFFT_DEVICE", "0"),
                "effective_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "previous_CUDA_VISIBLE_DEVICES": os.environ.get("THUNDERFFT_TABLE3_PREVIOUS_CUDA_VISIBLE_DEVICES")
                  if os.environ.get("THUNDERFFT_TABLE3_PREVIOUS_CUDA_VISIBLE_DEVICES_SET") else None,
                "git": capture(["git", "rev-parse", "HEAD"]),
                "worktree": capture(["git", "status", "--porcelain"]),
                "source_sha256": source_hashes(), "build_configuration": build_evidence(args.build_dir),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
                "gpu_inventory_before": capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": [], "scientific_validation": "bounded single-FFT gate; full paper comparison unavailable"}
    output_rows = []
    identity = None
    atomic_json(args.output / "metadata.json", metadata)
    try:
        for n in selection["N"]:
            for precision in selection["precisions"]:
                for placement in selection["placements"]:
                    label = f"n{n}-{precision}-{placement}"
                    command = [str(args.binary.resolve()), str(n), precision, placement,
                               str(settings["repeats"]), str(settings["trials"]), str(settings["batch"])]
                    csv_path, log_path = raw / f"{label}.csv", raw / f"{label}.log"
                    record = {"argv": command, "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "timeout_seconds": settings["timeout_seconds_per_configuration"],
                              "stdout": str(csv_path.relative_to(args.output)), "stderr": str(log_path.relative_to(args.output)),
                              "started_at": now()}
                    metadata["commands"].append(record)
                    atomic_json(args.output / "metadata.json", metadata)
                    with (args.output / "commands.txt").open("a") as stream:
                        stream.write(shlex.join(command) + "\n")
                    print(f"Table 3 [{label}] batch={settings['batch']} R={settings['repeats']}", flush=True)
                    with csv_path.open("w") as stdout, log_path.open("w") as stderr:
                        try:
                            process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                                     timeout=settings["timeout_seconds_per_configuration"], check=False)
                            record["exit_code"] = process.returncode
                        except subprocess.TimeoutExpired:
                            record["exit_code"] = 124
                            raise RuntimeError(f"{label}: timed out; see {log_path}")
                    record["finished_at"] = now()
                    if process.returncode:
                        raise RuntimeError(f"{label}: executable exited {process.returncode}; see {log_path}")
                    gpu, name, compute = gpu_name(log_path.read_text())
                    native_uuid = gpu_uuid(log_path.read_text())
                    this_identity = (gpu, name, compute, native_uuid)
                    if identity is not None and this_identity != identity:
                        raise ValueError("CUDA device identity changed during the matrix")
                    identity = this_identity
                    metadata.update(gpu=gpu, gpu_name=name, compute_capability=compute, gpu_uuid=native_uuid,
                                    paper_target_gpu_recognized=gpu in ("A100", "H100", "B200"))
                    record["native_metadata"] = log_path.read_text().splitlines()
                    output_rows += parse_raw(csv_path.read_text(), n=n, precision=precision, placement=placement,
                                             batch=settings["batch"], repeats=settings["repeats"], trials=settings["trials"],
                                             gpu=gpu, raw_csv=str(csv_path.relative_to(args.output)))
                    write_csv(args.output / "data.csv", output_rows, FIELDS)
                    summary = summarize(output_rows, args.profile)
                    write_csv(args.output / "table.csv", summary, tuple(summary[0]))
                    metadata["coverage"]["measured_thunderfft_configurations"] += 1
                    atomic_json(args.output / "metadata.json", metadata)
        summary = summarize(output_rows, args.profile)
        write_csv(args.output / "table.csv", summary, tuple(summary[0]))
        metadata.update(status="completed", numerical_and_timing_gates_passed=True)
    except (OSError, ValueError, RuntimeError) as error:
        metadata.update(status="failed", error=str(error), numerical_and_timing_gates_passed=False)
        print(f"Table 3 failed: {error}", file=sys.stderr)
        return 1
    finally:
        metadata["finished_at"] = now()
        atomic_json(args.output / "metadata.json", metadata)
    print(f"Completed {selected_count}/28 ThunderFFT configurations on this GPU; "
          f"cuFFTDx comparison excluded. Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
