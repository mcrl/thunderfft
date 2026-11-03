#!/usr/bin/env python3
"""Validate independently collected Table 5 phases and summarize their coverage."""

import argparse
import csv
import json
import math
from pathlib import Path

PHASES = ("accuracy_reduced", "accuracy_fp32_cufft", "range_fp16", "range_tf32", "range_fp32_cufft")
PRECISIONS = ("fp16", "tf32", "fp32")
RANGE_PHASE = dict(zip(PRECISIONS, PHASES[2:]))
ACCURACY_PHASE = {"fp16": "accuracy_reduced", "tf32": "accuracy_reduced", "fp32": "accuracy_fp32_cufft"}
RAW_FIELDS = ["N", "batch", "mode", "alpha", "precision", "rel_l2", "max_abs", "rel_linf", "all_finite"]
DATA_FIELDS = ["phase", "source_csv", "implementation", "reference", *RAW_FIELDS]
TABLE_FIELDS = ["N", "batch", "precision", "implementation", "reference", "rel_l2_alpha1",
                "largest_finite_exponent_observed", "largest_finite_alpha_observed",
                "scan_exponent_min", "scan_exponent_max", "overflow_observed", "left_censored",
                "right_censored", "finite_boundary_monotone", "accuracy_status", "range_status"]


def load_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RAW_FIELDS:
            raise ValueError(f"{path}: invalid measurement CSV header")
        return list(reader)


def selected_phases(metadata):
    phases = metadata.get("selected_phases", list(PHASES))
    if not isinstance(phases, list) or not phases or len(phases) != len(set(phases)) or set(phases) - set(PHASES):
        raise ValueError("selected_phases must contain unique supported Table 5 phase names")
    return phases


def validate_bundle(root, metadata):
    """Read and validate a bundle without modifying its metadata or measurements."""
    sizes, batch = metadata["N"], metadata["batch"]
    phases = selected_phases(metadata)
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - {64, 128, 256, 512, 1024, 2048, 4096}:
        raise ValueError("metadata N must contain unique supported FFT sizes")
    if not isinstance(batch, int) or batch <= 0 or batch % 16:
        raise ValueError("metadata batch must be a positive multiple of 16")
    errors, raw_data, environments = [], [], []
    accuracy, ranges, components = {}, {}, {}
    for name in phases:
        environment_path = root / f"{name}.environment.json"
        if not environment_path.exists():
            errors.append(f"{name}: missing selected-GPU/library environment record")
        else:
            environments.append(json.loads(environment_path.read_text()))
        try:
            rows = load_rows(root / f"{name}.csv")
        except (OSError, ValueError) as error:
            errors.append(str(error))
            continue
        phase = "accuracy" if name.startswith("accuracy_") else "finite_scan"
        reference = "cuFFT FP64" if phase == "accuracy" else "none (finite-output check)"
        for row in rows:
            raw_data.append({"phase": phase, "source_csv": f"{name}.csv",
                             "implementation": metadata["implementations"].get(row["precision"], "unknown"),
                             "reference": reference, **row})
        phase_errors_before = len(errors)
        if phase == "accuracy":
            precisions = {"fp16", "tf32"} if name == "accuracy_reduced" else {"fp32"}
            expected = {(n, p) for n in sizes for p in precisions}
            seen = set()
            if len(rows) != len(expected):
                errors.append(f"{name}: expected {len(expected)} rows, found {len(rows)}")
            for row in rows:
                try:
                    key = (int(row["N"]), row["precision"])
                    if key not in expected or key in seen:
                        errors.append(f"{name}: unexpected or duplicate row {key}")
                    seen.add(key)
                    value = float(row["rel_l2"])
                    limit = 1e-5 if row["precision"] == "fp32" else 1e-2
                    if int(row["batch"]) != batch or row["mode"] != "gaussian" or float(row["alpha"]) != 1.0:
                        errors.append(f"{name}: wrong input configuration for {key}")
                    if row["all_finite"] != "1" or not math.isfinite(value) or not 0 <= value <= limit:
                        errors.append(f"{name}: {key} failed broad numerical sanity limit {limit} (rel_l2={value})")
                    accuracy[key] = row
                except (TypeError, ValueError, OverflowError) as error:
                    errors.append(f"{name}: malformed measurement row: {error}")
            if seen != expected:
                errors.append(f"{name}: missing expected configurations")
        else:
            precision = next(p for p, phase_name in RANGE_PHASE.items() if phase_name == name)
            lower, upper = metadata["range_exponents"][precision]
            if not isinstance(lower, int) or not isinstance(upper, int) or not 0 <= lower <= upper <= 127:
                raise ValueError(f"{name}: invalid range_exponents")
            groups = {n: [] for n in sizes}
            for row in rows:
                try:
                    n = int(row["N"])
                    if n not in groups or row["precision"] != precision or int(row["batch"]) != batch or row["mode"] != "gaussian":
                        errors.append(f"{name}: unexpected configuration")
                        continue
                    alpha = float(row["alpha"])
                    if not math.isfinite(alpha) or alpha <= 0:
                        errors.append(f"{name}: invalid alpha {alpha}")
                        continue
                    exponent = round(math.log2(alpha))
                    if not math.isclose(alpha, math.ldexp(1.0, exponent), rel_tol=1e-7):
                        errors.append(f"{name}: alpha is not a power of two")
                        continue
                    if row["all_finite"] not in {"0", "1"}:
                        errors.append(f"{name}: invalid all_finite flag")
                        continue
                    groups[n].append((exponent, row["all_finite"] == "1"))
                except (TypeError, ValueError, OverflowError) as error:
                    errors.append(f"{name}: malformed measurement row: {error}")
            for n, values in groups.items():
                values.sort()
                complete = [e for e, _ in values] == list(range(lower, upper + 1))
                if not complete:
                    errors.append(f"{name}: N={n} has missing, duplicate, or out-of-range exponents")
                finite = [e for e, ok in values if ok]
                nonfinite = [e for e, ok in values if not ok]
                monotone = not finite or not nonfinite or max(finite) < min(nonfinite)
                if not monotone:
                    errors.append(f"{name}: N={n} has a nonmonotonic finite-output boundary")
                # An interrupted/incomplete scan must never become a boundary observation.
                if complete and monotone:
                    ranges[n, precision] = {
                        "largest_finite_exponent_observed": max(finite) if finite else "",
                        "largest_finite_alpha_observed": math.ldexp(1.0, max(finite)) if finite else "",
                        "scan_exponent_min": lower, "scan_exponent_max": upper,
                        "overflow_observed": bool(nonfinite), "left_censored": bool(nonfinite) and not finite,
                        "right_censored": bool(finite) and max(finite) == upper,
                        "finite_boundary_monotone": monotone,
                    }
        if len(errors) == phase_errors_before:
            for n in sizes:
                components[name, n] = [row for row in rows if int(row["N"]) == n]
    if environments and any(environment != environments[0] for environment in environments[1:]):
        errors.append("GPU/library environment changed between measurement invocations")
    return {"errors": errors, "raw_data": raw_data, "accuracy": accuracy, "ranges": ranges,
            "components": components, "environment": environments[0] if environments else None}


def summary_rows(sizes, metadata, accuracy, ranges):
    summary = []
    for n in sizes:
        for precision in PRECISIONS:
            row = {key: "" for key in TABLE_FIELDS}
            row.update(N=n, batch=metadata.get("batch", ""), precision=precision,
                       implementation=metadata.get("implementations", {}).get(precision, ""), reference="cuFFT FP64",
                       rel_l2_alpha1=accuracy.get((n, precision), {}).get("rel_l2", ""),
                       accuracy_status="complete" if (n, precision) in accuracy else "pending",
                       range_status="complete" if (n, precision) in ranges else "pending")
            row.update(ranges.get((n, precision), {}))
            summary.append(row)
    return summary


def missing_components(sizes, components):
    return [{"item": f"{name}:N={n}", "reason": "validated phase CSV has not been collected"}
            for n in sizes for name in PHASES if (name, n) not in components]


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(root, metadata, raw_data, summary):
    write_csv(root / "data.csv", DATA_FIELDS, raw_data)
    for filename in ("table5_summary.csv", "table.csv"):
        write_csv(root / filename, TABLE_FIELDS, summary)
    reference_path = Path(__file__).with_name("table5_accuracy_range_summary_with_fp32.csv")
    with reference_path.open(newline="") as stream:
        references = {int(row["N"]): row for row in csv.DictReader(stream)}
    comparisons = []
    for row in summary:
        reference = references.get(row["N"], {})
        precision = row["precision"]
        expected_error = reference.get(f"rel_l2_{precision}_alpha1", "")
        expected_exponent = reference.get(f"largest_finite_exp_{precision}", "")
        comparisons.append({
            "N": row["N"], "precision": precision, "implementation": row["implementation"],
            "measured_rel_l2_alpha1": row["rel_l2_alpha1"], "retained_reference_rel_l2_alpha1": expected_error,
            "relative_error_ratio_to_reference": float(row["rel_l2_alpha1"]) / float(expected_error)
                if row["rel_l2_alpha1"] != "" and expected_error else "",
            "measured_largest_finite_exponent": row["largest_finite_exponent_observed"],
            "retained_reference_finite_exponent": expected_exponent,
            "finite_exponent_delta": row["largest_finite_exponent_observed"] - int(expected_exponent)
                if row["largest_finite_exponent_observed"] != "" and expected_exponent else "",
            "boundary_fully_observed": row["range_status"] == "complete" and not row["left_censored"] and not row["right_censored"],
            "comparison_status": "informational; paper population/provenance unestablished",
        })
    if comparisons:
        write_csv(root / "comparison.csv", list(comparisons[0]), comparisons)
    lines = ["# Measured Table 5 quantities", "",
             f"Profile: `{metadata['profile']}`; batch: {metadata.get('batch', 'pending')}; seed: {metadata.get('seed', 'pending')}.", "",
             "Accuracy reference: cuFFT FP64 over the generated FP32 input. FP32 measures cuFFT C2C.", "",
             "| N | Precision / implementation | Relative L2 at alpha=1 | Largest finite exponent observed | Boundary |",
             "|---:|---|---:|---:|---|"]
    for row in summary:
        boundary = ("pending" if row["range_status"] != "complete" else
                    "below scan" if row["left_censored"] else
                    "at/above scan maximum" if row["right_censored"] else "observed")
        error_text = f"{float(row['rel_l2_alpha1']):.6e}" if row["rel_l2_alpha1"] != "" else "pending"
        exponent = row["largest_finite_exponent_observed"] if row["range_status"] == "complete" else "pending"
        lines.append(f"| {row['N']} | {row['precision']} / {row['implementation']} | {error_text} | {exponent} | {boundary} |")
    lines += ["", "Finite exponents refer to alpha=2^k; finite-only scan errors are intentionally not measured.", "",
              "Uncollected phases remain pending. Only a complete validated scan contributes a range observation.", "",
              "`comparison.csv` compares retained source reference values for context. "
              "The paper's population and FP32 provenance are not established, so this is not an exact-paper acceptance test.", ""]
    (root / "table.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    root = parser.parse_args().result_dir
    metadata = json.loads((root / "metadata.json").read_text())
    result = validate_bundle(root, metadata)
    metadata["measured_environment"] = result["environment"]
    metadata["missing_components"] = missing_components(metadata["N"], result["components"])
    metadata["full_paper_comparison"] = False
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summary_rows(metadata["N"], metadata, result["accuracy"], result["ranges"])
    write_outputs(root, metadata, result["raw_data"], summary)
    errors = result["errors"]
    validation = {
        "numerical_sanity": "failed" if errors else "passed",
        "scope": "Completeness of each selected phase, finite alpha=1 outputs, broad relative-L2 limits, and scanned finite-boundary consistency; this does not validate paper provenance or exact published numbers.",
        "limits": {"fp16": 1e-2, "tf32": 1e-2, "fp32_cufft": 1e-5},
        "scientific_validation": "not_assessed", "selected_phases": selected_phases(metadata),
        "measured_rows": len(result["raw_data"]), "summary_rows": len(summary),
        "paper_size_set_complete": set(metadata["N"]) == {64, 256, 1024, 4096},
        "missing_components": metadata["missing_components"], "errors": errors,
    }
    (root / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        raise SystemExit(1)
    print(f"Selected phases passed numerical sanity checks; summary: {root / 'table5_summary.csv'}")


if __name__ == "__main__":
    main()
