#!/usr/bin/env python3
"""Run the Table 6 drivers, check their output, and write data.csv and table.csv.

Time per FFT batch = (t_2R - t_R) / R, median over trials.
"""

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
import statistics
import subprocess
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLE3 = module("table6_table3_measure", HERE.parent / "table3/measure.py")
N = 256
NATIVE_FIELDS = ("N", "precision", "input", "placement", "batch", "inside_repeats", "trial", "t_R_us",
                 "t_2R_us", "compute_us", "e2e_us", "e2e_mode", "single_fft_relL2", "shared_bytes",
                 "library", "candidate", "ffts_per_block", "threads_per_block")
FIELDS = ("gpu", "N", "batch", "library", "variant", "candidate", "revision", "precision", "input",
          "placement", "execution_model", "repeat", "inside_repeats", "t_R_us", "t_2R_us", "time_ms",
          "e2e_ms", "e2e_mode", "rel_l2", "passed", "shared_bytes", "ffts_per_block", "threads_per_block")
TABLE_FIELDS = ("gpu", "N", "batch", "library", "variant", "selected_candidate", "trials",
                "median_time_ms", "speedup_vs_cufftdx", "median_e2e_ms", "rel_l2", "paper_time_ms",
                "paper_speedup", "execution_model")
# Fastest cuFFTDx configuration for N=256 FP16 batch 65536 on A100 (Table 3 sweep).
CUFFTDX_CANDIDATE = "block_f16_e16"
# Paper p. 9, Table 6 (A100, N=256, batch 65,536).
PAPER = {"cuFFTDx": (0.023, 1.00), "tcFFT": (0.021, 1.11), "FlashFFTConv": (0.018, 1.28),
         "ThunderFFT": (0.016, 1.46)}
# (library, variant): description
VARIANTS = {
    ("cuFFTDx", "block_fft"): f"cuFFTDx block FFT {CUFFTDX_CANDIDATE}",
    ("tcFFT", "layer_256_0_A100"): "tcFFT layer_256_0_A100 with repeat loop",
    ("FlashFFTConv", "conv_kernel"): "FlashFFTConv N=256 conv kernel with repeat loop, complex FFT",
    ("ThunderFFT", "register"): "ThunderFFT register kernel (Table 3)",
}


def positive_int(raw, name):
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def check_timing(t_r, t_2r, compute, e2e, repeats):
    if not all(math.isfinite(value) and value > 0 for value in (t_r, t_2r, compute, e2e)):
        raise ValueError("all native timings must be positive and finite")
    if t_2r <= t_r or not math.isclose(compute, (t_2r - t_r) / repeats, rel_tol=1e-4, abs_tol=1e-6):
        raise ValueError("native differential timing is invalid or inconsistent")


def parse_native(text, *, library, batch, repeats, trials, gpu, revision):
    """Validate a Table 6 driver CSV (tcFFT or FlashFFTConv)."""
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != NATIVE_FIELDS:
        raise ValueError(f"native CSV header does not match the Table 6 {library} contract")
    label = {"tcfft": "tcFFT", "flashfftconv": "FlashFFTConv"}[library]
    rows, seen = [], {}
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("malformed native CSV row")
        actual = (int(raw["N"]), raw["precision"], int(raw["batch"]), int(raw["inside_repeats"]), raw["library"])
        if actual != (N, "fp16", batch, repeats, library):
            raise ValueError(f"native CSV configuration mismatch: {actual}")
        candidate = raw["candidate"]
        if library == "tcfft":
            if candidate != "layer_256_0_A100" or raw["input"] != "complex" or raw["e2e_mode"] != "in_place":
                raise ValueError("invalid tcFFT candidate/input/e2e mode")
            variant = candidate
        else:
            if candidate == "conv_kernel" and raw["input"] == "complex":
                variant = "conv_kernel"
            else:
                raise ValueError("invalid FlashFFTConv candidate/input")
            if raw["e2e_mode"] != "out_of_place":
                raise ValueError("invalid FlashFFTConv e2e mode")
        trial = int(raw["trial"])
        trials_seen = seen.setdefault(candidate, set())
        if trial in trials_seen or not 0 <= trial < trials:
            raise ValueError("duplicate or out-of-range native trial")
        trials_seen.add(trial)
        t_r, t_2r, compute, e2e = (float(raw[key]) for key in ("t_R_us", "t_2R_us", "compute_us", "e2e_us"))
        check_timing(t_r, t_2r, compute, e2e, repeats)
        rel_l2 = float(raw["single_fft_relL2"])
        if not math.isfinite(rel_l2) or not 0 <= rel_l2 <= 0.005:
            raise ValueError("native single-FFT relative L2 gate failed")
        rows.append({"gpu": gpu, "N": N, "batch": batch, "library": label, "variant": variant,
                     "candidate": candidate, "revision": revision, "precision": "fp16", "input": raw["input"],
                     "placement": raw["placement"], "execution_model": VARIANTS[(label, variant)],
                     "repeat": trial, "inside_repeats": repeats, "t_R_us": t_r, "t_2R_us": t_2r,
                     "time_ms": compute / 1000.0, "e2e_ms": e2e / 1000.0, "e2e_mode": raw["e2e_mode"],
                     "rel_l2": rel_l2, "passed": "true", "shared_bytes": int(raw["shared_bytes"]),
                     "ffts_per_block": positive_int(raw["ffts_per_block"], "ffts_per_block"),
                     "threads_per_block": positive_int(raw["threads_per_block"], "threads_per_block")})
    if not seen:
        raise ValueError("native CSV has no measurements")
    for candidate, trials_seen in seen.items():
        if trials_seen != set(range(trials)):
            raise ValueError(f"expected {trials} complete timing trials for {candidate}, got {len(trials_seen)}")
    return rows


def parse_paper_fft(text, *, library, batch, repeats, trials, gpu, revision):
    """ThunderFFT/cuFFTDx output of thff_paper_fft, checked by the Table 3 parser."""
    label = {"thunderfft": "ThunderFFT", "cufftdx": "cuFFTDx"}[library]
    parsed = TABLE3.parse_raw(text, n=N, precision="fp16", placement="register", batch=batch, repeats=repeats,
                              trials=trials, gpu=gpu, raw_csv="", library=label)
    rows = []
    for row in parsed:
        variant = ("register" if label == "ThunderFFT" else
                   "thread_fft" if row["candidate"].startswith("thread_") else "block_fft")
        rows.append({"gpu": gpu, "N": N, "batch": batch, "library": label, "variant": variant,
                     "candidate": row["candidate"] or "ThunderFFT_benchmark_reg", "revision": revision,
                     "precision": "fp16", "input": "complex", "placement": "register",
                     "execution_model": VARIANTS.get((label, variant), label),
                     "repeat": row["trial"], "inside_repeats": repeats, "t_R_us": row["t_R_us"],
                     "t_2R_us": row["t_2R_us"], "time_ms": row["time_us"] / 1000.0,
                     "e2e_ms": row["e2e_out_of_place_us"] / 1000.0, "e2e_mode": "out_of_place",
                     "rel_l2": row["rel_l2"], "passed": "true", "shared_bytes": row["shared_bytes"],
                     "ffts_per_block": row["ffts_per_block"], "threads_per_block": row["threads_per_block"]})
    return rows


def summarize(rows):
    """One row per (library, variant): the candidate with the smallest median
    compute time; speedup = selected cuFFTDx median / row median."""
    groups = {}
    for row in rows:
        groups.setdefault((row["gpu"], int(row["batch"]), row["library"], row["variant"]), {}) \
              .setdefault(row["candidate"], []).append(row)
    selected = {}
    for key, candidates in groups.items():
        best = min(candidates, key=lambda name: statistics.median(float(r["time_ms"]) for r in candidates[name]))
        selected[key] = (best, candidates[best])
    table = []
    order = list(VARIANTS)
    for (gpu, batch, library, variant), (best, measured) in sorted(
            selected.items(), key=lambda item: (item[0][0], item[0][1],
                                                order.index(item[0][2:]) if item[0][2:] in order else len(order))):
        median = statistics.median(float(r["time_ms"]) for r in measured)
        baseline = selected.get((gpu, batch, "cuFFTDx", "block_fft"))
        reference = statistics.median(float(r["time_ms"]) for r in baseline[1]) if baseline else None
        paper = PAPER.get(library) if gpu == "A100" and batch == 65536 else None
        table.append({"gpu": gpu, "N": N, "batch": batch, "library": library, "variant": variant,
                      "selected_candidate": best, "trials": len(measured),
                      "median_time_ms": round(median, 6),
                      "speedup_vs_cufftdx": round(reference / median, 3) if reference else "",
                      "median_e2e_ms": round(statistics.median(float(r["e2e_ms"]) for r in measured), 6),
                      "rel_l2": max(float(r["rel_l2"]) for r in measured),
                      "paper_time_ms": paper[0] if paper else "", "paper_speedup": paper[1] if paper else "",
                      "execution_model": measured[0]["execution_model"]})
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thunderfft-binary", type=Path, required=True)
    parser.add_argument("--cufftdx-binary", type=Path, required=True)
    parser.add_argument("--table6-build", type=Path, required=True, help="artifact_table6/build.json")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        settings = {"batch": TABLE3.positive_env("THUNDERFFT_TABLE6_BATCH", 4096 if args.profile == "smoke" else 65536),
                    "repeats": TABLE3.positive_env("THUNDERFFT_TABLE6_REPEATS", 64 if args.profile == "smoke" else 1000),
                    "trials": TABLE3.positive_env("THUNDERFFT_TABLE6_TRIALS", 2 if args.profile == "smoke" else 3),
                    "timeout_seconds_per_library": TABLE3.positive_env("THUNDERFFT_TABLE6_TIMEOUT", 600)}
    except ValueError as error:
        parser.error(str(error))
    if settings["batch"] % 512:
        parser.error("THUNDERFFT_TABLE6_BATCH must be a positive multiple of 512")
    if args.profile == "paper" and settings["batch"] != 65536:
        parser.error("paper profile requires batch=65536; use smoke for custom checks")
    build = json.loads(args.table6_build.read_text())
    cache = (args.build_dir / "CMakeCache.txt").read_text() if (args.build_dir / "CMakeCache.txt").is_file() else ""
    mathdx = re.search(r"^MATHDX_ROOT:\w+=(.*)$", cache, re.MULTILINE)
    git = TABLE3.capture(["git", "rev-parse", "HEAD"])
    thunderfft_revision = "ThunderFFT " + git.get("stdout", "unknown")[:12]
    revisions = {
        "thunderfft": thunderfft_revision,
        "cufftdx": None,  # filled from the native log (cuFFTDx=<version>)
        "tcfft": f"tcFFT {build['tcfft']['pinned_commit'][:12]} + {Path(build['tcfft']['patch']['path']).name}",
        "flashfftconv": f"FlashFFTConv {build['flashfftconv']['pinned_commit'][:12]} + {Path(build['flashfftconv']['patch']['path']).name}",
    }
    metadata = {"schema_version": 1, "evaluation": "table6", "profile": args.profile, "started_at": TABLE3.now(),
                "status": "running", "settings": settings, "N": N, "precision": "fp16", "direction": "forward",
                "timing": "(t_2R - t_R) / R, median over trials; speedup = cuFFTDx / implementation",
                "correctness": "FP64 DFT on 64 FFTs before timing, relL2 <= 0.005",
                "paper_reference": {library: {"time_ms": value[0], "speedup": value[1]}
                                    for library, value in PAPER.items()},
                "table6_build": build, "mathdx_root": mathdx.group(1) if mathdx else None,
                "run_id": os.environ.get("THUNDERFFT_RUN_ID"),
                "effective_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "git": git, "worktree": TABLE3.capture(["git", "status", "--porcelain"]),
                "source_sha256": dict(TABLE3.source_hashes(), **{
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(HERE.rglob("*")) if path.is_file() and "__pycache__" not in path.parts}),
                "build_configuration": TABLE3.build_evidence(args.build_dir),
                "binary_sha256": {"thunderfft": hashlib.sha256(args.thunderfft_binary.read_bytes()).hexdigest(),
                                  "cufftdx": hashlib.sha256(args.cufftdx_binary.read_bytes()).hexdigest()},
                "gpu_inventory_before": TABLE3.capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": []}
    common = [str(settings["repeats"]), str(settings["trials"]), str(settings["batch"])]
    paper_fft = ["256", "fp16", "register"]
    jobs = [("cufftdx", [str(args.cufftdx_binary.resolve()), "--library", "cufftdx",
                         "--candidate", CUFFTDX_CANDIDATE] + paper_fft + common),
            ("tcfft", [build["binaries"]["tcfft"]["path"]] + common),
            ("flashfftconv", [build["binaries"]["flashfftconv"]["path"]] + common),
            ("thunderfft", [str(args.thunderfft_binary.resolve())] + paper_fft + common)]
    rows, identity = [], None
    TABLE3.atomic_json(args.output / "metadata.json", metadata)
    try:
        for label, command in jobs:
            record = {"argv": command, "started_at": TABLE3.now()}
            metadata["commands"].append(record)
            print(f"Table 6 [{label}] batch={settings['batch']} R={settings['repeats']}", flush=True)
            try:
                process = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                         timeout=settings["timeout_seconds_per_library"], check=False)
            except subprocess.TimeoutExpired:
                record["exit_code"] = 124
                raise RuntimeError(f"{label}: timed out")
            record.update(exit_code=process.returncode, finished_at=TABLE3.now())
            log = process.stderr
            print(log, end="", flush=True)
            if process.returncode:
                raise RuntimeError(f"{label}: executable exited {process.returncode}")
            this_identity = TABLE3.gpu_name(log) + (TABLE3.gpu_uuid(log),)
            if identity is not None and this_identity != identity:
                raise ValueError("CUDA device identity changed during the run")
            identity = this_identity
            metadata.update(gpu=identity[0], gpu_name=identity[1], compute_capability=identity[2],
                            gpu_uuid=identity[3])
            library = label.split("-")[0]
            if library == "cufftdx":
                version = re.search(r"cuFFTDx=(\d+)", log)
                revisions["cufftdx"] = f"cuFFTDx {version.group(1) if version else 'unknown'}"
            options = dict(library=library, batch=settings["batch"], repeats=settings["repeats"],
                           trials=settings["trials"], gpu=identity[0], revision=revisions[library])
            parse = parse_paper_fft if library in ("thunderfft", "cufftdx") else parse_native
            rows += parse(process.stdout, **options)
            TABLE3.write_csv(args.output / "data.csv", rows, FIELDS)
            TABLE3.atomic_json(args.output / "metadata.json", metadata)
        table = summarize(rows)
        TABLE3.write_csv(args.output / "table.csv", table, TABLE_FIELDS)
        present = {(row["library"], row["variant"]) for row in table}
        metadata["missing"] = [f"{library}/{variant}" for (library, variant) in VARIANTS
                               if (library, variant) not in present]
        metadata["paper_configuration"] = identity[0] == "A100" and settings["batch"] == 65536
        metadata.update(status="completed", numerical_and_timing_gates_passed=True)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        metadata.update(status="failed", error=str(error), numerical_and_timing_gates_passed=False)
        print(f"Table 6 failed: {error}", file=sys.stderr)
        return 1
    finally:
        metadata["finished_at"] = TABLE3.now()
        TABLE3.atomic_json(args.output / "metadata.json", metadata)
    width = max(len(f"{row['library']} ({row['variant']})") for row in table)
    print(f"{'Library':{width}}  Time (ms)  Speedup  Paper")
    for row in table:
        name = f"{row['library']} ({row['variant']})"
        paper = f"{row['paper_time_ms']} / {row['paper_speedup']}" if row["paper_time_ms"] != "" else ""
        print(f"{name:{width}}  {row['median_time_ms']:9.4f}  {row['speedup_vs_cufftdx']:7}  {paper}")
    print(f"Completed Table 6 measurements. Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
