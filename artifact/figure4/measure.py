#!/usr/bin/env python3
"""Collect Figure 4: correctness-gated ThunderFFT FFT convolution and cuDNN over a
filter-length sweep, joined with separately produced backend CSVs (FlashFFTConv).

The sweep/collection helpers are shared with Figure 5 (artifact/figure5/measure.py).
"""

import argparse
import csv
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLE3 = module("conv_table3_measure", HERE.parent / "table3/measure.py")
# Backend adapters (cuDNN, FlashFFTConv, cuFFT, cuFFTDx) write these columns.
BACKEND_FIELDS = ("gpu", "filter", "library", "repeat", "time_ms", "rel_l2", "passed", "operation")
FIELDS = ("gpu", "shape", "filter", "library", "precision", "repeat", "time_ms", "filter_fft_included",
          "rel_l2", "validated", "passed", "operation", "source")
BUSY_RETRIES = 20
SUMMARY_FIELDS = ("gpu", "shape", "filter", "library", "repeats", "median_time_ms", "min_time_ms", "max_time_ms")

FIGURE4 = {
    "evaluation": "figure4", "title": "Figure 4", "env": "THUNDERFFT_FIGURE4",
    "shape": "L=4096,B=64,D=768",
    # Paper axis 3..51; the earlier A100 evaluation measured odd lengths.
    "filters": {"smoke": (3, 9, 53), "paper": tuple(range(3, 54, 2))},
    "allowed_filters": tuple(range(1, 4097)),
    "operation": "depthwise circular convolution y[n]=sum_k x[(n-k) mod L] h[k], filter not flipped, "
                 "full L outputs, real fp16 input/filter/output, seed 42 x,h~U[0,1]",
    "precision": "fp16",
}


def figure4_command(binary, filter_size, profile):
    batch, channels = (64, 768) if profile == "paper" else (4, 64)
    sample = os.environ.get("THUNDERFFT_FIGURE4_VALIDATE_CHANNELS", "1024")
    # N K device B D warmup runs include_filter_fft validate real_input real_filter validate_channels
    return [str(binary), "4096", str(filter_size), "0", str(batch), str(channels), "5", "30", "1", "1", "1", "1",
            sample], f"L=4096,B={batch},D={channels}"


def figure4_cudnn_command(binary, filter_size, profile):
    batch, channels = (64, 768) if profile == "paper" else (4, 64)
    sample = os.environ.get("THUNDERFFT_FIGURE4_VALIDATE_CHANNELS", "1024")
    return [str(binary), "--figure4", "4096", str(filter_size), str(batch), str(channels), "5", "30",
            sample], f"L=4096,B={batch},D={channels}"


def figure4_parse(stdout):
    """Return (time_ms, rel_l2, validated) from either native diagnostic output."""
    if "Circular convolution validation: PASS" not in stdout:
        raise ValueError("native circular-convolution correctness gate did not pass")
    rel = re.search(r"direct circular conv \((?:raw/N|cuDNN)\): L2_rel=(\S+)", stdout)
    checked = re.search(r"^Validation sequences: (\d+) of (\d+)$", stdout, re.MULTILINE)
    time = re.search(r"time: (\S+) ms$", stdout, re.MULTILINE)
    if not (rel and checked and time):
        raise ValueError("native output lacks validation or timing lines")
    return float(time.group(1)), float(rel.group(1)), f"{checked.group(1)}/{checked.group(2)} sequences"


def check_row(row):
    time_ms, rel_l2 = float(row["time_ms"]), float(row["rel_l2"])
    if not (math.isfinite(time_ms) and time_ms > 0):
        raise ValueError(f"{row['library']} f={row['filter']}: time must be positive and finite")
    if not (math.isfinite(rel_l2) and 0 <= rel_l2 <= 5e-3):
        raise ValueError(f"{row['library']} f={row['filter']}: relative L2 gate failed")
    return time_ms


def read_backends(paths, spec, gpu, shape, filters, measured=("ThunderFFT",)):
    """Join backend CSVs; reject rows whose operation or workload differs from ours."""
    rows, hashes = [], {}
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(BACKEND_FIELDS + ("shape",)) <= set(reader.fieldnames or ()):
                raise ValueError(f"{path}: backend CSV needs columns {BACKEND_FIELDS + ('shape',)}")
            for row in reader:
                if row["library"] in measured:
                    raise ValueError(f"{path}: backend rows cannot replace measured {row['library']}")
                if row["passed"].strip().lower() not in ("true", "1"):
                    raise ValueError(f"{path}: unvalidated {row['library']} f={row['filter']}")
                if (row["gpu"], row["shape"]) != (gpu, shape) or int(row["filter"]) not in filters:
                    raise ValueError(f"{path}: backend GPU/shape/filter does not match this run")
                if row["operation"] != spec["operation"]:
                    raise ValueError(f"{path}: {row['library']} operation differs: {row['operation']!r}")
                check_row(row)
                rows.append(dict({key: row.get(key, "") for key in FIELDS[:-1]}, source=str(path)))
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, hashes


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["gpu"], row["shape"], int(row["filter"]), row["library"]), []).append(float(row["time_ms"]))
    summary = [{"gpu": gpu, "shape": shape, "filter": size, "library": library, "repeats": len(times),
                "median_time_ms": statistics.median(times), "min_time_ms": min(times), "max_time_ms": max(times)}
               for (gpu, shape, size, library), times in sorted(groups.items())]
    # Per-library mean over the swept filters (the paper reports average runtimes).
    for library in sorted({row["library"] for row in summary}):
        medians = [row["median_time_ms"] for row in summary if row["library"] == library]
        summary.append({"gpu": summary[0]["gpu"], "shape": summary[0]["shape"], "filter": f"mean({len(medians)})",
                        "library": library, "repeats": "", "median_time_ms": statistics.mean(medians),
                        "min_time_ms": "", "max_time_ms": ""})
    return summary


def collect(spec, runners, build_dir, output, profile, sources):
    """Run one native invocation per (filter, library, trial); write data/summary/metadata.

    runners: (library, binary, command, parse, filter_fft_included) tuples.
    """
    env = spec["env"]
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    trials = TABLE3.positive_env(f"{env}_TRIALS", 1 if profile == "smoke" else 3)
    timeout = TABLE3.positive_env(f"{env}_TIMEOUT", 1800)
    filters = (TABLE3.selection_env(f"{env}_FILTERS", spec["allowed_filters"], int)
               if f"{env}_FILTERS" in os.environ else spec["filters"][profile])
    backends = [Path(path) for path in os.environ.get(f"{env}_BACKENDS", "").split(",") if path]
    metadata = {"schema_version": 1, "evaluation": spec["evaluation"], "profile": profile,
                "started_at": TABLE3.now(), "status": "running", "operation": spec["operation"],
                "notes": spec.get("notes"),
                "settings": {"trials": trials, "filters": list(filters), "timeout_seconds": timeout},
                "scope": f"{', '.join(r[0] for r in runners)} measured here; other libraries from {env}_BACKENDS",
                "full_paper_comparison_complete": False,
                "timing": {"metric": "native mean over its timed launches per trial",
                           "central_statistic": "median over trials",
                           "filter_fft_included": {r[0]: r[4] for r in runners}},
                "correctness": "native gate before the time is accepted: finite relL2 <= 5e-3 vs direct reference",
                "run_id": os.environ.get("THUNDERFFT_RUN_ID"),
                "effective_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "git": TABLE3.capture(["git", "rev-parse", "HEAD"]),
                "worktree": TABLE3.capture(["git", "status", "--porcelain"]),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(set(sources)) if path.is_file()},
                "build_cache": TABLE3.build_evidence(build_dir).get("CMakeCache.txt"),
                "binary_sha256": {r[0]: hashlib.sha256(r[1].read_bytes()).hexdigest() for r in runners},
                "gpu_inventory_before": TABLE3.capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": []}
    rows, identity, shape = [], None, None
    TABLE3.atomic_json(output / "metadata.json", metadata)
    try:
        for size in filters:
            for library, binary, command, parse, filter_fft in runners:
                for trial in range(trials):
                    argv, shape = command(binary.resolve(), size, profile)
                    label = f"f{size}-{library.lower()}-t{trial}"
                    out_path, log_path = raw / f"{label}.out", raw / f"{label}.log"
                    record = {"argv": argv, "stdout": str(out_path.relative_to(output)), "started_at": TABLE3.now()}
                    metadata["commands"].append(record)
                    with (output / "commands.txt").open("a") as stream:
                        stream.write(shlex.join(argv) + "\n")
                    print(f"{spec['title']} [{label}] {shape}", flush=True)
                    # Shared exclusive-process GPUs: another user may hold the device between
                    # runs. Only that start-up failure is retried; nothing is timed concurrently.
                    for attempt in range(BUSY_RETRIES + 1):
                        with out_path.open("w") as stdout, log_path.open("w") as stderr:
                            try:
                                process = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                                                         timeout=timeout, check=False)
                            except subprocess.TimeoutExpired:
                                record["exit_code"] = 124
                                raise RuntimeError(f"{label}: timed out; see {log_path}")
                        busy = "busy or unavailable" in out_path.read_text() + log_path.read_text()
                        if not (process.returncode and busy) or attempt == BUSY_RETRIES:
                            break
                        record["busy_retries"] = attempt + 1
                        time.sleep(30)
                    record.update(exit_code=process.returncode, finished_at=TABLE3.now())
                    if process.returncode:
                        raise RuntimeError(f"{label}: executable exited {process.returncode}; see {out_path}")
                    this_identity = TABLE3.gpu_name(log_path.read_text())
                    if identity is not None and this_identity != identity:
                        raise ValueError("CUDA device identity changed during the sweep")
                    identity = this_identity
                    metadata.update(gpu=identity[0], gpu_name=identity[1], compute_capability=identity[2])
                    time_ms, rel_l2, validated = parse(out_path.read_text())
                    row = {"gpu": identity[0], "shape": shape, "filter": size, "library": library,
                           "precision": spec["precision"], "repeat": trial, "time_ms": time_ms,
                           "filter_fft_included": filter_fft, "rel_l2": rel_l2, "validated": validated,
                           "passed": "true", "operation": spec["operation"], "source": record["stdout"]}
                    check_row(row)
                    rows.append(row)
                    TABLE3.write_csv(output / "data.csv", rows, FIELDS)
                    TABLE3.atomic_json(output / "metadata.json", metadata)
        backend_rows, metadata["backend_sha256"] = read_backends(backends, spec, identity[0], shape, filters,
                                                                 tuple(r[0] for r in runners))
        rows += backend_rows
        TABLE3.write_csv(output / "data.csv", rows, FIELDS)
        TABLE3.write_csv(output / "summary.csv", summarize(rows), SUMMARY_FIELDS)
        metadata["libraries"] = sorted({row["library"] for row in rows})
        metadata.update(status="completed", numerical_and_timing_gates_passed=True)
    except (OSError, ValueError, RuntimeError) as error:
        metadata.update(status="failed", error=str(error), numerical_and_timing_gates_passed=False)
        print(f"{spec['title']} failed: {error}", file=sys.stderr)
        return 1
    finally:
        metadata["finished_at"] = TABLE3.now()
        TABLE3.atomic_json(output / "metadata.json", metadata)
    print(f"Completed {spec['title']}: {len(filters)} filters, libraries={metadata['libraries']}. Results: {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cudnn-binary", type=Path, help="thff_conv1d_compare; omit to leave cuDNN pending")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    runners = [("ThunderFFT", args.binary, figure4_command, figure4_parse, "true")]
    if args.cudnn_binary:
        # Direct convolution: there is no filter transform to include.
        runners.append(("cuDNN", args.cudnn_binary, figure4_cudnn_command, figure4_parse, "n/a"))
    sources = [ROOT / "sample/conv1d/conv1d_fullfft_l4096_thunder.cu",
               ROOT / "sample/conv1d/conv1d_compare_cudnn_thunder.cu", *HERE.glob("*.py"), HERE / "run.sh"]
    try:
        return collect(FIGURE4, runners, args.build_dir, args.output, args.profile, sources)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
