#!/usr/bin/env python3
"""Summarize repeated measurements; never substitute targets for measurements."""
import argparse
import csv
import glob
import json
import math
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Measurement CSV paths or globs")
    parser.add_argument("--gpu", required=True, choices=("A100", "H100", "B200"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=0.10,
                        help="Descriptive relative tolerance; not a paper-defined criterion")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    targets = {}
    with (root / "paper-table3-xfft-targets.csv").open() as stream:
        for row in csv.DictReader(stream):
            if row["gpu"] == args.gpu:
                targets[(int(row["N"]), row["precision"], row["placement"])] = float(row["target_us"])
    groups = {}
    paths = sorted({path for pattern in args.inputs for path in glob.glob(pattern)})
    if not paths:
        parser.error("No input files matched")
    for path in paths:
        with open(path) as stream:
            for row in csv.DictReader(stream):
                if row.get("N") == "N":
                    continue  # Concatenated per-process headers.
                key = (int(row["N"]), row["precision"], row["placement"])
                groups.setdefault(key, []).append(row)
    fields = ["gpu", "N", "precision", "placement", "batch", "inside_repeats",
              "trials", "target_us", "median_compute_us", "min_compute_us",
              "max_compute_us", "relative_delta", "measured_over_paper",
              "within_descriptive_tolerance", "tolerance", "median_e2e_out_of_place_us",
              "single_fft_relL2_max", "positive_finite_slopes"]
    results = []
    for (n, precision, placement), rows in sorted(groups.items()):
        if any(int(r["batch"]) != 65536 for r in rows):
            raise ValueError("Paper comparisons require batch=65536")
        values = [float(r["compute_us"]) for r in rows]
        valid = all(math.isfinite(v) and v > 0 for v in values)
        target = targets[(n, precision, placement)]
        median = statistics.median(values)
        delta = median / target - 1
        results.append(dict(zip(fields, [
            args.gpu, n, precision, placement, 65536,
            ";".join(map(str, sorted({int(r["inside_repeats"]) for r in rows}))),
            len(values), target, median, min(values), max(values), delta,
            median / target, valid and abs(delta) <= args.tolerance, args.tolerance,
            statistics.median(float(r["e2e_out_of_place_us"]) for r in rows),
            max(float(r["single_fft_relL2"]) for r in rows), valid,
        ])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps({"gpu": args.gpu, "measured_configurations": len(results),
                      "paper_configurations_per_gpu": 28,
                      "within_descriptive_tolerance": sum(r["within_descriptive_tolerance"] for r in results),
                      "tolerance": args.tolerance, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
