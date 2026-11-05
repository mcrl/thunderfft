#!/usr/bin/env python3
"""Validate independently collected Table 5 phases and summarize their coverage."""

import argparse
import csv
import json
import math
from pathlib import Path

# The first five phases need only CUDA Toolkit; accuracy_fp16_cufftdx needs a
# MathDX build. A legacy bundle without selected_phases selected the first five.
DEFAULT_PHASES = ("accuracy_reduced", "accuracy_fp32_cufft", "range_fp16", "range_tf32", "range_fp32_cufft")
CUFFTDX_PHASES = ("accuracy_fp16_cufftdx",)
PHASES = DEFAULT_PHASES + CUFFTDX_PHASES
RANGE_PRECISIONS = ("fp16", "tf32", "fp32")
PRECISIONS = RANGE_PRECISIONS + ("fp16_cufftdx",)
RANGE_PHASE = {"fp16": "range_fp16", "tf32": "range_tf32", "fp32": "range_fp32_cufft"}
ACCURACY_PRECISIONS = {"accuracy_reduced": {"fp16", "tf32"}, "accuracy_fp32_cufft": {"fp32"},
                       "accuracy_fp16_cufftdx": {"fp16_cufftdx"}}
ACCURACY_PHASE = {p: name for name, precisions in ACCURACY_PRECISIONS.items() for p in precisions}
IMPLEMENTATIONS = {"fp16": "ThunderFFT", "tf32": "ThunderFFT", "fp32": "cuFFT C2C",
                   "fp16_cufftdx": "cuFFTDx (ElementsPerThread 8, FFTsPerBlock 2)"}
RAW_FIELDS = ["N", "batch", "mode", "alpha", "precision", "rel_l2", "max_abs", "rel_linf", "all_finite"]
REFERENCE = "cuFFT Z2Z FP64 of the FP32-generated input"
NO_REFERENCE = "none (finite-output check)"
IMPLEMENTATION_INPUT = {
    "fp16": "FP32-generated input rounded to FP16 (round-to-nearest)",
    "tf32": "FP32-generated input (float2 storage)",
    "fp32": "FP32-generated input",
    "fp16_cufftdx": "FP32-generated input rounded to FP16 (round-to-nearest) inside the kernel",
}
DATA_FIELDS = ["phase", "source_csv", "implementation", "implementation_input", "reference", *RAW_FIELDS]
TABLE_FIELDS = ["N", "batch", "precision", "implementation", "implementation_input", "reference", "rel_l2_alpha1",
                "largest_finite_exponent_observed", "largest_finite_alpha_observed",
                "first_nonfinite_exponent_observed", "scan_exponent_min", "scan_exponent_max",
                "overflow_observed", "left_censored", "right_censored", "finite_boundary_monotone",
                "boundary_status", "accuracy_status", "range_status"]


def data_row(phase_name, source_csv, metadata, row):
    """Label one raw measurement with its implementation, input and reference."""
    accuracy = phase_name.startswith("accuracy_")
    return {"phase": "accuracy" if accuracy else "finite_scan", "source_csv": source_csv,
            "implementation": metadata["implementations"].get(row["precision"], "unknown"),
            "implementation_input": IMPLEMENTATION_INPUT.get(row["precision"], "unknown"),
            "reference": REFERENCE if accuracy else NO_REFERENCE, **row}


def load_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != RAW_FIELDS:
            raise ValueError(f"{path}: invalid measurement CSV header")
        return list(reader)


def selected_phases(metadata):
    phases = metadata.get("selected_phases", list(DEFAULT_PHASES))
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
        for row in rows:
            raw_data.append(data_row(name, f"{name}.csv", metadata, row))
        phase_errors_before = len(errors)
        if phase == "accuracy":
            precisions = ACCURACY_PRECISIONS[name]
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
                    status = ("observed" if finite and nonfinite else
                              "left_censored" if nonfinite else "right_censored")
                    ranges[n, precision] = {
                        "largest_finite_exponent_observed": max(finite) if finite else "",
                        "largest_finite_alpha_observed": math.ldexp(1.0, max(finite)) if finite else "",
                        "first_nonfinite_exponent_observed": min(nonfinite) if nonfinite else "",
                        "scan_exponent_min": lower, "scan_exponent_max": upper,
                        "overflow_observed": bool(nonfinite), "left_censored": status == "left_censored",
                        "right_censored": status == "right_censored",
                        "finite_boundary_monotone": monotone, "boundary_status": status,
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
                       implementation=metadata.get("implementations", {}).get(precision, ""),
                       implementation_input=IMPLEMENTATION_INPUT[precision], reference=REFERENCE,
                       rel_l2_alpha1=accuracy.get((n, precision), {}).get("rel_l2", ""),
                       boundary_status="pending" if precision in RANGE_PHASE else "not_measured",
                       accuracy_status="complete" if (n, precision) in accuracy else "pending",
                       range_status="complete" if (n, precision) in ranges else
                                    "pending" if precision in RANGE_PHASE else "not_measured")
            row.update(ranges.get((n, precision), {}))
            summary.append(row)
    return summary


def missing_components(sizes, components):
    return [{"item": f"{name}:N={n}",
             "reason": ("validated cuFFTDx phase CSV has not been collected (requires a MATHDX_ROOT build)"
                        if name in CUFFTDX_PHASES else "validated phase CSV has not been collected")}
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
    here = Path(__file__).parent
    with (here / "table5_accuracy_range_summary_with_fp32.csv").open(newline="") as stream:
        references = {int(row["N"]): row for row in csv.DictReader(stream)}
    retained_batches = {}
    for name in ("alpha1_rel_l2_N64_256_1024_4096.csv", "alpha1_rel_l2_fp32_N64_256_1024_4096.csv"):
        with (here / name).open(newline="") as stream:
            retained_batches.update({(int(row["N"]), row["precision"]): row["batch"] for row in csv.DictReader(stream)})
    comparisons = []
    for row in summary:
        reference = references.get(row["N"], {})
        precision = row["precision"]
        expected_error = reference.get(f"rel_l2_{precision}_alpha1", "")
        expected_exponent = reference.get(f"largest_finite_exp_{precision}", "")
        observed = row["boundary_status"] == "observed"
        comparisons.append({
            "N": row["N"], "precision": precision, "implementation": row["implementation"],
            "measured_batch": row["batch"], "retained_reference_batch": retained_batches.get((row["N"], precision), ""),
            "measured_rel_l2_alpha1": row["rel_l2_alpha1"], "retained_reference_rel_l2_alpha1": expected_error,
            "relative_error_ratio_to_reference": float(row["rel_l2_alpha1"]) / float(expected_error)
                if row["rel_l2_alpha1"] != "" and expected_error else "",
            "measured_largest_finite_exponent": row["largest_finite_exponent_observed"],
            "measured_boundary_status": row["boundary_status"],
            "retained_reference_finite_exponent": expected_exponent,
            "finite_exponent_delta": row["largest_finite_exponent_observed"] - int(expected_exponent)
                if observed and expected_exponent else "",
            "boundary_fully_observed": observed,
            "comparison_status": "informational; Table 5 seed and input details are not stated in the paper",
        })
    if comparisons:
        write_csv(root / "comparison.csv", list(comparisons[0]), comparisons)
    lines = ["# Measured Table 5 quantities", "",
             f"Profile: `{metadata['profile']}`; batch: {metadata.get('batch', 'pending')}; seed: {metadata.get('seed', 'pending')}.", "",
             "Accuracy reference: cuFFT Z2Z FP64 of the FP32-generated input. "
             "FP16 receives that input rounded to half, so its error includes input rounding. "
             "FP16/TF32 measure ThunderFFT; FP32 measures cuFFT C2C; fp16_cufftdx measures cuFFTDx FP16 "
             "(ElementsPerThread 8, FFTsPerBlock 2) with the same input and reference.", "",
             "| N | Precision / implementation | Relative L2 at alpha=1 | Largest finite exponent observed | Boundary |",
             "|---:|---|---:|---:|---|"]
    for row in summary:
        status = row["boundary_status"]
        boundary = {"pending": "pending", "not_measured": "not measured",
                    "left_censored": f"non-finite at scan minimum 2^{row['scan_exponent_min']}",
                    "right_censored": f"no overflow up to scan maximum 2^{row['scan_exponent_max']}",
                    "observed": f"first non-finite at 2^{row['first_nonfinite_exponent_observed']}"}[status]
        error_text = f"{float(row['rel_l2_alpha1']):.6e}" if row["rel_l2_alpha1"] != "" else "pending"
        exponent = {"pending": "pending", "not_measured": "not measured", "left_censored": f"< {row['scan_exponent_min']}",
                    "right_censored": f">= {row['scan_exponent_max']}"}.get(
                        status, row["largest_finite_exponent_observed"])
        lines.append(f"| {row['N']} | {row['precision']} / {row['implementation']} | {error_text} | {exponent} | {boundary} |")
    lines += ["", "Finite exponents refer to alpha=2^k; finite-only scan errors are intentionally not measured.", "",
              "Uncollected phases remain pending. Only a complete validated scan contributes a range observation; "
              "a censored scan reports a bound, not an overflow boundary.", "",
              "`comparison.csv` compares retained source reference values for context. "
              "The paper does not state Table 5's seed or several input details, so this is not an exact-paper acceptance test.", ""]
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
        "scope": "Completeness of each selected phase, finite alpha=1 outputs, broad relative-L2 limits, and scanned finite-boundary consistency; this does not validate unstated paper conditions or exact published numbers.",
        "limits": {"fp16": 1e-2, "tf32": 1e-2, "fp32_cufft": 1e-5, "fp16_cufftdx": 1e-2},
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
