#!/usr/bin/env python3
"""Validate independently collected Table 5 phases and summarize their coverage."""

import argparse
import csv
import json
import math
from pathlib import Path

# The first three phases need only CUDA Toolkit; the four cuFFTDx phases need a
# MathDX build. A bundle without selected_phases selected the first three.
DEFAULT_PHASES = ("accuracy_reduced", "range_fp16", "range_tf32")
CUFFTDX_PHASES = ("accuracy_fp32_cufftdx", "accuracy_fp16_cufftdx", "range_fp32_cufftdx", "range_fp16_cufftdx")
PHASES = DEFAULT_PHASES + CUFFTDX_PHASES
RANGE_PRECISIONS = ("fp16", "tf32", "fp32_cufftdx", "fp16_cufftdx")
CUFFTDX_PRECISIONS = ("fp32_cufftdx", "fp16_cufftdx")
PRECISIONS = RANGE_PRECISIONS
RANGE_PHASE = {"fp16": "range_fp16", "tf32": "range_tf32", "fp32_cufftdx": "range_fp32_cufftdx",
               "fp16_cufftdx": "range_fp16_cufftdx"}
ACCURACY_PRECISIONS = {"accuracy_reduced": {"fp16", "tf32"}, "accuracy_fp32_cufftdx": {"fp32_cufftdx"},
                       "accuracy_fp16_cufftdx": {"fp16_cufftdx"}}
ACCURACY_PHASE = {p: name for name, precisions in ACCURACY_PRECISIONS.items() for p in precisions}
IMPLEMENTATIONS = {"fp16": "ThunderFFT", "tf32": "ThunderFFT",
                   "fp32_cufftdx": "cuFFTDx (ElementsPerThread 8, FFTsPerBlock 1)",
                   "fp16_cufftdx": "cuFFTDx (ElementsPerThread 8, FFTsPerBlock 2)"}
ACCURACY_LIMITS = {"fp16": 1e-2, "tf32": 1e-2, "fp32_cufftdx": 1e-5, "fp16_cufftdx": 1e-2}
RAW_FIELDS = ["N", "batch", "mode", "alpha", "precision", "rel_l2", "max_abs", "rel_linf", "all_finite"]
REFERENCE = "cuFFT Z2Z FP64 of the FP32-generated input"
NO_REFERENCE = "none (finite-output check)"
IMPLEMENTATION_INPUT = {
    "fp16": "FP32-generated input rounded to FP16 (round-to-nearest)",
    "tf32": "FP32-generated input (float2 storage)",
    "fp32_cufftdx": "FP32-generated input",
    "fp16_cufftdx": "FP32-generated input rounded to FP16 (round-to-nearest) inside the kernel",
}
DATA_FIELDS = ["phase", "source_csv", "implementation", "implementation_input", "reference", *RAW_FIELDS]
TABLE_FIELDS = ["N", "batch", "precision", "implementation", "implementation_input", "reference", "rel_l2_alpha1",
                "largest_finite_exponent_observed", "largest_finite_alpha_observed",
                "first_nonfinite_exponent_observed", "scan_exponent_min", "scan_exponent_max",
                "overflow_observed", "left_censored", "right_censored", "finite_boundary_monotone",
                "boundary_status", "accuracy_status", "range_status", "range_coverage", "full_range_scan",
                "expected_last_finite_exponent", "boundary_matches_preset"]


def range_mode(metadata):
    mode = metadata.get("range_mode", "full")
    if mode not in ("boundary", "full"):
        raise ValueError("range_mode must be boundary or full")
    return mode


def profile_ranges(profile):
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported Table 5 profile")
    return {precision: ([8, 13] if precision in ("fp16", "fp16_cufftdx") else [120, 127])
            if profile == "smoke" else [0, 127] for precision in RANGE_PRECISIONS}


def validate_base_ranges(metadata):
    expected = profile_ranges(metadata["profile"])
    actual = metadata.get("range_exponents", {})
    selected = selected_phases(metadata)
    legacy = "range_mode" not in metadata
    required = set(expected) - {p for p in CUFFTDX_PRECISIONS if legacy and RANGE_PHASE[p] not in selected}
    if not isinstance(actual, dict) or not required <= set(actual) <= set(expected) \
            or any(actual[key] != expected[key] for key in actual):
        raise ValueError(f"finite scan ranges do not match the {metadata['profile']} profile")
    return expected


def evidence_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("boundary attempt evidence must use relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError(f"missing or escaping boundary attempt evidence: {relative}")
    return path


def window(value, bounds):
    if not isinstance(value, list) or len(value) != 2 or any(type(k) is not int for k in value) \
            or not bounds[0] <= value[0] <= value[1] <= bounds[1]:
        raise ValueError("invalid boundary window within profile exponent grid")
    return value


def attempt_values(rows, n, precision, batch, bounds):
    values = []
    for row in rows:
        if int(row["N"]) != n or row["precision"] != precision or int(row["batch"]) != batch \
                or row["mode"] != "gaussian" or row["all_finite"] not in ("0", "1"):
            raise ValueError("boundary attempt has inconsistent input configuration")
        alpha = float(row["alpha"])
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("boundary attempt has invalid alpha")
        exponent = round(math.log2(alpha))
        if not math.isclose(alpha, math.ldexp(1.0, exponent), rel_tol=1e-7):
            raise ValueError("boundary attempt alpha is not a power of two")
        values.append((exponent, row["all_finite"] == "1"))
    values.sort()
    if [k for k, _ in values] != list(range(bounds[0], bounds[1] + 1)):
        raise ValueError("boundary attempt has missing, duplicate, or out-of-range exponents")
    finite, nonfinite = [k for k, ok in values if ok], [k for k, ok in values if not ok]
    if finite and nonfinite and max(finite) >= min(nonfinite):
        raise ValueError("boundary attempt has a nonmonotonic finite-output boundary")
    return dict(values)


def boundary_check(root, metadata, precision, n, rows, environment):
    """Revalidate retained adaptive attempts and the final combined CSV slice."""
    check = metadata.get("range_checks", {}).get(precision, {}).get(str(n))
    if not isinstance(check, dict):
        raise ValueError(f"missing boundary range_checks for {precision}:N={n}")
    bounds = profile_ranges(metadata["profile"])[precision]
    initial, final = window(check.get("initial_window"), bounds), window(check.get("window"), bounds)
    expected = check.get("expected_last_finite")
    if expected is not None and (type(expected) is not int or not bounds[0] <= expected <= bounds[1]):
        raise ValueError("invalid expected_last_finite boundary provenance")
    if not isinstance(check.get("preset_source"), str) or not check["preset_source"]:
        raise ValueError("boundary preset_source provenance is required")
    attempts = check.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError("boundary attempts must be nonempty")
    previous, evidence, observed = None, set(), {}
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict) or attempt.get("exit_code") != 0:
            raise ValueError("boundary attempt did not complete successfully")
        current = window([attempt.get("min"), attempt.get("max")], bounds)
        if index == 0 and current != initial or previous and not (current[0] <= previous[0] and current[1] >= previous[1]):
            raise ValueError("boundary attempts must start at initial_window and expand contiguously")
        paths = {key: evidence_file(root, attempt.get(key)) for key in ("csv", "environment", "log")}
        if json.loads(paths["environment"].read_text()) != environment:
            raise ValueError("GPU/library environment changed between boundary attempts")
        attempted_rows = load_rows(paths["csv"])
        values = attempt_values(attempted_rows, n, precision, metadata["batch"], current)
        if any(k in observed and observed[k] != finite for k, finite in values.items()):
            raise ValueError("boundary attempts disagree on repeated finite-output measurements")
        observed.update(values)
        evidence.update(attempt[key] for key in paths)
        previous = current
    if previous != final:
        raise ValueError("final boundary attempt does not match window")
    canonical = lambda values: sorted(values, key=lambda row: float(row["alpha"]))
    if canonical(attempted_rows) != canonical(rows):
        raise ValueError("combined phase CSV disagrees with final boundary attempt")
    return check, evidence


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
    mode = range_mode(metadata)
    base_ranges = validate_base_ranges(metadata)
    if not sizes or len(set(sizes)) != len(sizes) or set(sizes) - {64, 128, 256, 512, 1024, 2048, 4096}:
        raise ValueError("metadata N must contain unique supported FFT sizes")
    if not isinstance(batch, int) or batch <= 0 or batch % 16:
        raise ValueError("metadata batch must be a positive multiple of 16")
    errors, raw_data, environments = [], [], []
    accuracy, ranges, components, evidence = {}, {}, {}, set()
    for name in phases:
        environment_path = root / f"{name}.environment.json"
        phase_environment = None
        if not environment_path.exists():
            errors.append(f"{name}: missing selected-GPU/library environment record")
        else:
            phase_environment = json.loads(environment_path.read_text())
            environments.append(phase_environment)
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
                    limit = ACCURACY_LIMITS[row["precision"]]
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
                lower, upper = base_ranges[precision]
                check = None
                if mode == "boundary":
                    try:
                        check, attempt_files = boundary_check(root, metadata, precision, n,
                            [row for row in rows if int(row["N"]) == n], phase_environment)
                        lower, upper = check["window"]
                        evidence.update(attempt_files)
                    except (OSError, KeyError, TypeError, ValueError, OverflowError) as error:
                        errors.append(f"{name}: N={n}: {error}")
                        continue
                values.sort()
                complete = [e for e, _ in values] == list(range(lower, upper + 1))
                if not complete:
                    errors.append(f"{name}: N={n} has missing, duplicate, or out-of-range exponents")
                finite = [e for e, ok in values if ok]
                nonfinite = [e for e, ok in values if not ok]
                monotone = not finite or not nonfinite or max(finite) < min(nonfinite)
                if not monotone:
                    errors.append(f"{name}: N={n} has a nonmonotonic finite-output boundary")
                bracketed = bool(finite and nonfinite and max(finite) + 1 == min(nonfinite))
                if mode == "boundary" and not bracketed:
                    errors.append(f"{name}: N={n} boundary recheck requires adjacent finite/nonfinite exponents")
                # An interrupted/incomplete scan must never become a boundary observation.
                if complete and monotone and (mode == "full" or bracketed):
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
                        "range_coverage": "boundary_recheck" if mode == "boundary" else "full_profile_scan",
                        "full_range_scan": mode == "full",
                        "expected_last_finite_exponent": check["expected_last_finite"]
                            if check and check["expected_last_finite"] is not None else "",
                        "boundary_matches_preset": max(finite) == check["expected_last_finite"]
                            if check and finite and check["expected_last_finite"] is not None else "",
                    }
        if len(errors) == phase_errors_before:
            for n in sizes:
                components[name, n] = [row for row in rows if int(row["N"]) == n]
    if environments and any(environment != environments[0] for environment in environments[1:]):
        errors.append("GPU/library environment changed between measurement invocations")
    return {"errors": errors, "raw_data": raw_data, "accuracy": accuracy, "ranges": ranges,
            "components": components, "environment": environments[0] if environments else None,
            "evidence_files": sorted(evidence)}


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
                                    "pending" if precision in RANGE_PHASE else "not_measured",
                       range_coverage="pending", full_range_scan=False)
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
    with (here / "alpha1_rel_l2_N64_256_1024_4096.csv").open(newline="") as stream:
        retained_batches = {(int(row["N"]), row["precision"]): row["batch"] for row in csv.DictReader(stream)}
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
            "range_coverage": row["range_coverage"], "full_range_scan": row["full_range_scan"],
            "retained_reference_finite_exponent": expected_exponent,
            "finite_exponent_delta": row["largest_finite_exponent_observed"] - int(expected_exponent)
                if observed and expected_exponent else "",
            "boundary_fully_observed": observed and row["full_range_scan"],
            "boundary_bracketed": observed,
            "comparison_status": "informational; Table 5 seed and input details are not stated in the paper",
        })
    if comparisons:
        write_csv(root / "comparison.csv", list(comparisons[0]), comparisons)
    lines = ["# Measured Table 5 quantities", "",
             f"Profile: `{metadata['profile']}`; batch: {metadata.get('batch', 'pending')}; seed: {metadata.get('seed', 'pending')}.", "",
             "Accuracy reference: cuFFT Z2Z FP64 of the FP32-generated input. "
             "FP16 receives that input rounded to half, so its error includes input rounding. "
             "FP16/TF32 measure xFFT (ThunderFFT); fp32_cufftdx measures cuFFTDx FP32 (ElementsPerThread 8, "
             "FFTsPerBlock 1) and fp16_cufftdx measures cuFFTDx FP16 (ElementsPerThread 8, FFTsPerBlock 2), "
             "with the same input and reference.", "",
             "| N | Precision / implementation | Relative L2 at alpha=1 | Largest finite exponent observed | Boundary | Measured exponent window / coverage |",
             "|---:|---|---:|---:|---|---|"]
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
        coverage = (f"{row['scan_exponent_min']}..{row['scan_exponent_max']} / {row['range_coverage']}"
                    if row["range_status"] == "complete" else "pending")
        implementation = "xFFT (ThunderFFT)" if row["implementation"] == "ThunderFFT" else row["implementation"]
        lines.append(f"| {row['N']} | {row['precision']} / {implementation} | {error_text} | {exponent} | {boundary} | {coverage} |")
    lines += ["", "Finite exponents refer to alpha=2^k; finite-only scan errors are intentionally not measured.", "",
              "Uncollected phases remain pending. A boundary_recheck measures every exponent in its displayed window "
              "and requires adjacent finite/nonfinite outputs; it does not claim a full-grid scan or a global maximum. "
              "A full_profile_scan measures the profile exponent grid. A censored full sweep reports a bound.", "",
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
    metadata["range_exponents"] = validate_base_ranges(metadata)
    metadata["range_mode"] = range_mode(metadata)
    metadata["range_coverage"] = "boundary_recheck" if metadata["range_mode"] == "boundary" else "full_profile_scan"
    metadata["full_range_scan"] = metadata["range_mode"] == "full" and all(
        (phase, n) in result["components"] for n in metadata["N"] for phase in RANGE_PHASE.values())
    metadata["full_paper_comparison"] = False
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    summary = summary_rows(metadata["N"], metadata, result["accuracy"], result["ranges"])
    write_outputs(root, metadata, result["raw_data"], summary)
    errors = result["errors"]
    validation = {
        "numerical_sanity": "failed" if errors else "passed",
        "scope": "Completeness of each selected phase, finite alpha=1 outputs, broad relative-L2 limits, and scanned finite-boundary consistency; this does not validate unstated paper conditions or exact published numbers.",
        "limits": dict(ACCURACY_LIMITS),
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
