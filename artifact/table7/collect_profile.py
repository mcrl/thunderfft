#!/usr/bin/env python3
"""Optional Nsight Compute collection, independent of ordinary timing trials."""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

DEFAULT_METRIC = "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active"


def parse_metric(text, metric):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if "Metric Name" in line and "Metric Value" in line:
            rows = list(csv.DictReader(io.StringIO("\n".join(lines[index:]))))
            matches = [r for r in rows if r.get("Metric Name") == metric]
            if len(matches) != 1:
                raise ValueError(f"Expected one measured {metric} value, found {len(matches)}")
            row = matches[0]
            value = float(row["Metric Value"].replace(",", ""))
            if row.get("Metric Unit") not in ("%", "pct") or not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError("Expected a finite percentage counter")
            return {"metric": metric, "unit": row["Metric Unit"], "value": value,
                    "kernel": row.get("Kernel Name", ""), "raw_row": row}
    raise ValueError("No Nsight Compute metric header found")


def collect(binary, variant, batch, output, ncu="ncu", metric=DEFAULT_METRIC):
    output.mkdir(parents=True, exist_ok=True)
    raw = (output / f"{variant}.ncu.csv").resolve()
    command = [ncu, "--clock-control", "none", "--cache-control", "none", "--csv",
               "--log-file", str(raw), "--kernel-name", "regex:.*fft_kernel_radix64_batch16.*",
               "--launch-skip", "2", "--launch-count", "1", "--metrics", metric,
               str(binary.resolve()), variant, str(batch), "32", "1", "--profile"]
    record = {"argv": command, "variant": variant, "batch": batch, "R": 32,
              "metric": metric, "unit": "percent of peak sustained active",
              "selection": "two R1 correctness launches skipped, then one R32 profiling launch",
              "paper_metric_identity": "unconfirmed", "clock_control": "none", "cache_control": "none",
              "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    try:
        version = subprocess.run([ncu, "--version"], capture_output=True, text=True, timeout=15)
        record["profiler_version"] = version.stdout + version.stderr
        with (output / f"{variant}.profile.log").open("w") as stream:
            run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, LC_ALL="C"), timeout=180)
        record["exit_code"] = run.returncode
        if run.returncode:
            raise ValueError(f"Nsight Compute exited {run.returncode}; see profiling log")
        measured = parse_metric(raw.read_text(), metric)
        record["measurement"] = measured
        return measured
    finally:
        (output / f"{variant}.profile.json").write_text(json.dumps(record, indent=2) + "\n")
        (output / f"{variant}.profile-command.txt").write_text(shlex.join(command) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--variant", choices=("register_branch", "register_uniform"), required=True)
    parser.add_argument("--batch", type=int, default=65536)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch <= 0 or args.batch % 16:
        parser.error("batch must be a positive multiple of 16")
    print(json.dumps(collect(args.binary, args.variant, args.batch, args.output,
                             os.environ.get("THUNDERFFT_NCU", "ncu"),
                             os.environ.get("THUNDERFFT_TABLE7_METRIC", DEFAULT_METRIC)), indent=2))


if __name__ == "__main__":
    main()
