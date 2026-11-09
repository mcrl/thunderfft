"""Paper-shaped presentation of validated Tables 5, 6 and 7.

These adapters read assembled measurements and explicitly identified estimates.
Retained ``paper_*`` comparison fields never supply a presentation cell.
"""
import csv
import json
import math
from pathlib import Path


MISSING = "—"
SIZES = (64, 256, 1024, 4096)
SERIES = (("fp16", "FP16\nxFFT"), ("fp16_cufftdx", "FP16\ncuFFTDx"),
          ("tf32", "TF32\nxFFT"), ("fp32_cufftdx", "FP32\ncuFFTDx"))
ROLES = (("radix16", "radix-16 (tcFFT)"),
         ("radix4_compute", "radix-4 + reducing computation"),
         ("register_branch", "+ reducing communication"),
         ("register_uniform", "+ reducing branch divergence"))
SELECTED_BRANCH_SOURCE = "historical_e4aa1b0"
SELECTED_BRANCH_COMMIT = "e4aa1b0bf7d38493fcb8d5ff7c1ffd599c7d6375"


def _read(directory):
    directory = Path(directory)
    with (directory / "table.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    return rows, metadata


def _index(rows, fields):
    result = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in fields)
        if key in result:
            raise ValueError("duplicate table presentation key: " + repr(key))
        result[key] = row
    return result


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _decimal(value, places):
    number = _number(value)
    return f"{number:.{places}f}" if number is not None else MISSING


def _scientific(value):
    number = _number(value)
    if number is None or number < 0:
        return MISSING
    if number == 0:
        return "$0$"
    mantissa, exponent = f"{number:.2e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def _power(value, relation=""):
    exponent = _number(value)
    if exponent is None or not exponent.is_integer():
        return MISSING
    return rf"${relation}2^{{{int(exponent)}}}$"


def _finite_range(row):
    if row.get("range_status") != "complete":
        return MISSING
    status = row.get("boundary_status")
    if status == "observed":
        return _power(row.get("largest_finite_exponent_observed"))
    if status == "right_censored":
        return _power(row.get("scan_exponent_max"), r"\geq ")
    if status == "left_censored":
        return _power(row.get("scan_exponent_min"), "<")
    return MISSING


def _rules(count, header=1):
    return [{"row": 0, "start": 0, "end": count, "weight": 1.0},
            {"row": header, "start": 0, "end": count, "weight": 0.6}]


def _table5(rows, metadata):
    indexed = _index(rows, ("N", "precision"))
    displayed = [["", r"Rel-$L_2$ at $\alpha=1$", "", "", "",
                  r"Largest finite $\alpha$", "", "", ""],
                 ["$N$", *[label for _, label in SERIES], *[label for _, label in SERIES]]]
    for size in SIZES:
        selected = [indexed.get((str(size), precision), {}) for precision, _ in SERIES]
        displayed.append([str(size),
                          *[_scientific(row.get("rel_l2_alpha1"))
                            if row.get("accuracy_status") == "complete" else MISSING for row in selected],
                          *[_finite_range(row) for row in selected]])
    notes = ["Relative error uses a double-precision FFT reference. The two FP16 columns distinguish xFFT and cuFFTDx.",
             "— denotes an uncollected measurement; inequalities denote a censored finite-range scan."]
    if any(row.get("range_coverage") == "boundary_recheck" for row in rows):
        notes.append("Boundary recheck validates adjacent finite/nonfinite outputs within the measured exponent window; it does not perform a full-range scan.")
    if metadata.get("batch") is not None:
        notes.append("Measured batch size: " + str(metadata["batch"]) + ".")
    rules = _rules(9, 2) + [
        {"row": 1, "start": 1, "end": 5, "weight": 0.5},
        {"row": 1, "start": 5, "end": 9, "weight": 0.5},
        {"row": len(displayed), "start": 0, "end": 9, "weight": 1.0}]
    return {"number": 5, "caption": "Numerical accuracy and dynamic range relative to a double-precision FFT reference.",
            "width": 8.6, "font_size": 9,
            "panels": [{"header_rows": 2, "widths": [0.65, 1.25, 1.25, 1.25, 1.25, 0.95, 0.95, 0.95, 0.95],
                        "rows": displayed, "spans": [{"row": 0, "col": 1, "rowspan": 1, "colspan": 4},
                                                       {"row": 0, "col": 5, "rowspan": 1, "colspan": 4}],
                        "rules": rules}], "notes": notes}


def _table6(rows, metadata):
    a100 = [row for row in rows if row.get("gpu", "A100") == "A100"]
    indexed = _index(a100, ("library",))
    displayed = [["Library", "Time (ms)", "Speedup"]]
    baseline = indexed.get(("cuFFTDx",), {})
    baseline_time = _number(baseline.get("median_time_ms"))
    baseline_ok = (baseline.get("status", "collected") == "collected"
                   and baseline_time is not None and baseline_time > 0)
    for library, label in (("cuFFTDx", "cuFFTDx"), ("tcFFT", "tcFFT"),
                           ("FlashFFTConv", "FlashFFTConv"), ("ThunderFFT", "xFFT")):
        row = indexed.get((library,), {})
        collected = row.get("status", "collected") == "collected"
        timing = _decimal(row.get("median_time_ms"), 3) if collected else MISSING
        speedup = _decimal(row.get("speedup_vs_cufftdx"), 2) if collected and baseline_ok else MISSING
        displayed.append([label, timing, speedup])
    return {"number": 6, "caption": "Computation time comparison across different Tensor Core-based FFT libraries.",
            "width": 4.2, "font_size": 10,
            "panels": [{"header_rows": 1, "bold_header": True, "widths": [1.8, 1.1, 1.0], "rows": displayed,
                        "align": {0: "left"}, "rules": _rules(3) + [
                            {"row": len(displayed), "start": 0, "end": 3, "weight": 1.0}]}],
            "notes": ["A100, N = 256. Speedup is relative to the measured cuFFTDx baseline.",
                      "— denotes an uncollected measurement or unavailable baseline."]}


def _table7_selected_sources(indexed, metadata):
    """Accept explicit role assignments only when the row retains native scope."""
    selections = metadata.get("paper_row_sources", {})
    roles = {variant for variant, _ in ROLES}
    if not isinstance(selections, dict) or set(selections) - roles:
        raise ValueError("Invalid Table 7 paper-row source mapping")
    selected = set()
    for role, selection in selections.items():
        if not isinstance(selection, dict):
            raise ValueError("Invalid Table 7 paper-row source selection")
        source = selection.get("source_variant")
        if source == role:
            continue
        row = indexed.get((role,), {})
        if (role != "register_branch" or source != SELECTED_BRANCH_SOURCE
                or selection.get("selection_status") != "artifact_selected"
                or not isinstance(selection.get("selection_basis"), str)
                or not selection["selection_basis"].strip()
                or row.get("source_variant") != source
                or row.get("source_commit") != SELECTED_BRANCH_COMMIT
                or row.get("implementation_kind") != "historical_partial_kernel"
                or row.get("operation_coverage") != "partial_fft_operand_setup_omitted"
                or row.get("paper_identity") != "unconfirmed"
                or metadata.get("full_paper_comparison") is True):
            raise ValueError("Selected Table 7 source disagrees with the displayed row or its function scope")
        status = row.get("timing_status")
        accuracy = row.get("accuracy_status")
        if ((status == "missing" and accuracy != "not_collected")
                or (status == "collected_unvalidated" and accuracy != "unchecked")
                or status not in ("missing", "collected_unvalidated")):
            raise ValueError("Selected Table 7 partial kernel requires unchecked timing or a missing measurement")
        selected.add(role)
    for role in roles:
        source = indexed.get((role,), {}).get("source_variant")
        if source and source != role and role not in selected:
            raise ValueError("Remapped Table 7 row requires matching paper-row source metadata")
    return selected


def _table7(rows, metadata):
    indexed = _index(rows, ("variant",))
    selected_sources = _table7_selected_sources(indexed, metadata)
    baseline = indexed.get(("radix16",), {})
    baseline_time = _number(baseline.get("median_time_ms"))
    baseline_ok = (baseline.get("timing_status") == "collected"
                   and baseline.get("accuracy_status") == "passed"
                   and baseline_time is not None and baseline_time > 0)
    estimated_time = _number(baseline.get("estimated_time_ms"))
    estimated_utilization = _number(baseline.get("estimated_utilization_pct"))
    estimated_baseline = (baseline.get("timing_status") == "estimated"
                          and baseline.get("utilization_status") == "estimated"
                          and baseline.get("accuracy_status") == "not_applicable_projection"
                          and estimated_time is not None and estimated_time > 0
                          and estimated_utilization is not None and estimated_utilization > 0)
    displayed = [["Configuration", "Time (ms)", "Utilization",
                  "Speedup*" if estimated_baseline else "Speedup"]]
    unchecked = False
    for variant, label in ROLES:
        row = indexed.get((variant,), {})
        selected_partial = variant in selected_sources
        if selected_partial:
            label += " (e4aa1b0)"
        if variant == "radix16" and estimated_baseline:
            displayed.append([label + "*", _decimal(estimated_time, 4),
                              _decimal(estimated_utilization, 0) + "%", "1.00"])
            continue
        status = row.get("timing_status")
        collected = status in ("collected", "collected_unvalidated")
        unvalidated = collected and (status == "collected_unvalidated" or row.get("accuracy_status") != "passed")
        unchecked = unchecked or unvalidated
        timing = _decimal(row.get("median_time_ms"), 4) if collected else MISSING
        utilization = (_decimal(row.get("utilization_pct"), 0)
                       if collected and row.get("utilization_status") in ("calculated", "calculated_unvalidated")
                       else MISSING)
        if utilization != MISSING:
            utilization += "%"
        speedup_field = "speedup_vs_estimated_radix16" if estimated_baseline else "radix16_speedup"
        speedup_value = _number(row.get(speedup_field))
        measured_time = _number(row.get("median_time_ms"))
        selected_ratio = (selected_partial and estimated_baseline
                          and status == "collected_unvalidated"
                          and row.get("accuracy_status") == "unchecked"
                          and row.get("utilization_status") == "calculated_unvalidated"
                          and measured_time is not None and measured_time > 0
                          and speedup_value is not None and speedup_value > 0
                          and math.isclose(speedup_value, estimated_time / measured_time,
                                           rel_tol=1e-8))
        speedup = (_decimal(speedup_value, 2)
                   if (baseline_ok or estimated_baseline) and collected and (not unvalidated or selected_ratio)
                   and measured_time is not None and measured_time > 0
                   and speedup_value is not None and speedup_value > 0 else MISSING)
        displayed.append([label + ("†" if unvalidated else ""), timing, utilization, speedup])
    notes = ["— denotes an unavailable measurement. Missing original stages are not replaced by retained paper numbers.",
             "Recovered candidates are displayed under their proposed paper roles; exact original-stage identities remain unconfirmed."]
    if estimated_baseline:
        notes.append("* Assumes the tcFFT utilization measured in Table 6 is retained for N=64 TF32; the first-row time is estimated. Speedups use this estimated baseline.")
    elif not baseline_ok:
        notes.append("The original radix-16 baseline is unavailable; speedups relative to that baseline remain unreported.")
    if unchecked:
        notes.append("† Timing-only measurement: FFT numerical correctness was not checked.")
    if selected_sources:
        notes.append("The selected e4aa1b0 kernel omits MMA B-fragment initialization and includes a shared-memory round trip each repeat; its FFT outputs are undefined.")
        if estimated_baseline:
            notes.append("The e4aa1b0 ratio compares unit-kernel timings with the estimated baseline; it does not represent complete-FFT speedup.")
    if any(row.get("variant") not in {variant for variant, _ in ROLES} for row in rows):
        notes.append("Additional native variants remain in kernel_comparison.csv; only the explicit paper-row source mapping selects a displayed stage.")
    return {"number": 7, "caption": "Ablation study of the optimization steps.",
            "width": 6.2, "font_size": 9,
            "panels": [{"header_rows": 1, "bold_header": True, "widths": [3.3, 1.0, 1.1, 0.9], "rows": displayed,
                        "align": {0: "left"}, "rules": _rules(4) + [
                            {"row": len(displayed), "start": 0, "end": 4, "weight": 1.0}]}],
            "notes": notes}


def build(evaluation, directory):
    """Return a renderer-independent paper layout without changing any inputs."""
    builders = {"table5": _table5, "table6": _table6, "table7": _table7}
    if evaluation not in builders:
        raise ValueError("unsupported accuracy/comparison table: " + str(evaluation))
    rows, metadata = _read(directory)
    return builders[evaluation](rows, metadata)
