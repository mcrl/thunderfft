"""Paper-shaped presentation of validated Tables 5, 6 and 7.

These adapters only read assembled measurements.  In particular, retained
``paper_*`` comparison fields never supply an unmeasured presentation cell.
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


def _table7(rows, metadata):
    indexed = _index(rows, ("variant",))
    displayed = [["Configuration", "Time (ms)", "Utilization", "Speedup"]]
    baseline = indexed.get(("radix16",), {})
    baseline_time = _number(baseline.get("median_time_ms"))
    baseline_ok = (baseline.get("timing_status") == "collected"
                   and baseline.get("accuracy_status") == "passed"
                   and baseline_time is not None and baseline_time > 0)
    unchecked = False
    for variant, label in ROLES:
        row = indexed.get((variant,), {})
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
        speedup = (_decimal(row.get("radix16_speedup"), 2)
                   if baseline_ok and collected and not unvalidated else MISSING)
        displayed.append([label + ("†" if unvalidated else ""), timing, utilization, speedup])
    notes = ["— denotes an unavailable measurement. Missing original stages are not replaced by retained paper numbers.",
             "Recovered candidates are displayed under their proposed paper roles; exact original-stage identities remain unconfirmed."]
    if not baseline_ok:
        notes.append("The original radix-16 baseline is unavailable; speedups relative to that baseline remain unreported.")
    if unchecked:
        notes.append("† Timing-only measurement: FFT numerical correctness was not checked.")
    if any(row.get("variant") not in {variant for variant, _ in ROLES} for row in rows):
        notes.append("Supplemental partial-kernel microbenchmarks remain in kernel_comparison.csv and do not replace the four paper stages.")
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
