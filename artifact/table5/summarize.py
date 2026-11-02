#!/usr/bin/env python3
"""Validate the collected rows and summarize measured Table 5 quantities."""

import argparse
import csv
import json
import math
from pathlib import Path


def load_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir
    metadata = json.loads((root / "metadata.json").read_text())
    sizes = metadata["N"]
    batch = metadata["batch"]
    errors = []
    accuracy = {}
    raw_data = []

    def retain_rows(filename, rows, phase, reference):
        for row in rows:
            raw_data.append({
                "phase": phase,
                "source_csv": f"{filename}.csv",
                "implementation": metadata["implementations"].get(row["precision"], "unknown"),
                "reference": reference,
                **row,
            })

    for name, precisions in (("accuracy_reduced", {"fp16", "tf32"}), ("accuracy_fp32_cufft", {"fp32"})):
        rows = load_rows(root / f"{name}.csv")
        retain_rows(name, rows, "accuracy", "cuFFT FP64")
        expected = {(n, p) for n in sizes for p in precisions}
        if len(rows) != len(expected):
            errors.append(f"{name}: expected {len(expected)} rows, found {len(rows)}")
        seen = set()
        for row in rows:
            key = (int(row["N"]), row["precision"])
            if key not in expected or key in seen:
                errors.append(f"{name}: unexpected or duplicate row {key}")
            seen.add(key)
            accuracy[key] = row
            value = float(row["rel_l2"])
            limit = 1e-5 if row["precision"] == "fp32" else 1e-2
            if int(row["batch"]) != batch or row["mode"] != "gaussian" or float(row["alpha"]) != 1.0:
                errors.append(f"{name}: wrong input configuration for {key}")
            if row["all_finite"] != "1" or not math.isfinite(value) or not (0 <= value <= limit):
                errors.append(f"{name}: {key} failed broad numerical sanity limit {limit} (rel_l2={value})")
        if seen != expected:
            errors.append(f"{name}: missing expected configurations")

    summary = []
    for precision, filename in (("fp16", "range_fp16"), ("tf32", "range_tf32"), ("fp32", "range_fp32_cufft")):
        lower, upper = metadata["range_exponents"][precision]
        rows = load_rows(root / f"{filename}.csv")
        retain_rows(filename, rows, "finite_scan", "none (finite-output check)")
        groups = {n: [] for n in sizes}
        for row in rows:
            n = int(row["N"])
            if n not in groups or row["precision"] != precision or int(row["batch"]) != batch or row["mode"] != "gaussian":
                errors.append(f"{filename}: unexpected configuration")
                continue
            alpha = float(row["alpha"])
            if not math.isfinite(alpha) or alpha <= 0:
                errors.append(f"{filename}: invalid alpha {alpha}")
                continue
            exponent = round(math.log2(alpha))
            if not math.isclose(alpha, math.ldexp(1.0, exponent), rel_tol=1e-7):
                errors.append(f"{filename}: alpha is not a power of two")
            if row["all_finite"] not in {"0", "1"}:
                errors.append(f"{filename}: invalid all_finite flag")
            groups[n].append((exponent, row["all_finite"] == "1"))
        for n, values in groups.items():
            values.sort()
            if [e for e, _ in values] != list(range(lower, upper + 1)):
                errors.append(f"{filename}: N={n} has missing, duplicate, or out-of-range exponents")
            finite = [e for e, ok in values if ok]
            nonfinite = [e for e, ok in values if not ok]
            monotone = not finite or not nonfinite or max(finite) < min(nonfinite)
            if not monotone:
                errors.append(f"{filename}: N={n} has a nonmonotonic finite-output boundary")
            accuracy_row = accuracy.get((n, precision), {})
            summary.append({
                "N": n,
                "batch": batch,
                "precision": precision,
                "implementation": metadata["implementations"][precision],
                "reference": "cuFFT FP64",
                "rel_l2_alpha1": accuracy_row.get("rel_l2", ""),
                "largest_finite_exponent_observed": max(finite) if finite else "",
                "largest_finite_alpha_observed": math.ldexp(1.0, max(finite)) if finite else "",
                "scan_exponent_min": lower,
                "scan_exponent_max": upper,
                "overflow_observed": bool(nonfinite),
                "right_censored": bool(finite) and max(finite) == upper,
                "finite_boundary_monotone": monotone,
            })

    if raw_data:
        with (root / "data.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(raw_data[0]))
            writer.writeheader()
            writer.writerows(raw_data)
    if summary:
        for filename in ("table5_summary.csv", "table.csv"):
            with (root / filename).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
    validation = {
        "numerical_sanity": "failed" if errors else "passed",
        "scope": "Row completeness, finite alpha=1 outputs, broad relative-L2 limits, and scanned finite-boundary consistency; this does not validate paper provenance or exact published numbers.",
        "limits": {"fp16": 1e-2, "tf32": 1e-2, "fp32_cufft": 1e-5},
        "scientific_validation": "not_assessed",
        "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"Numerical sanity checks passed; measured summary: {root / 'table5_summary.csv'}")


if __name__ == "__main__":
    main()
