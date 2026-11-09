"""Paper-shaped views of the assembled Table 2, 3 and 4 results.

These builders never launch measurements or substitute published timing targets
for missing results. The machine-readable CSVs remain the source of truth.
"""
import csv
import json
import math
from pathlib import Path
import re
import statistics


MISSING = "—"
GPUS = ("A100", "H100", "B200")
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _metadata(directory):
    path = directory / "metadata.json"
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _number(value, positive=False):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or (positive and number <= 0):
        return None
    return number


def _format(value, digits=2, comma=False):
    number = _number(value)
    return MISSING if number is None else format(number, ("," if comma else "") + f".{digits}f")


def _index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row.get(field, "") for field in keys)
        if key in result:
            raise ValueError("ambiguous paper table row: " + repr(key))
        result[key] = row
    return result


def _has_preset(value):
    if isinstance(value, dict):
        return any((key in ("cufftdx_mode", "mode") and item == "preset")
                   or _has_preset(item) for key, item in value.items())
    if isinstance(value, list):
        return any(_has_preset(item) for item in value)
    return False


def _scope_notes(metadata):
    notes = []
    if _has_preset(metadata):
        notes.append("cuFFTDx uses remeasured saved presets; parameter search is not included.")
    if metadata.get("profile") == "smoke":
        notes.append("Smoke-profile results use reduced settings and are not a full paper comparison.")
    if metadata.get("status") in ("partial", "pending"):
        notes.append("This result is partial; uncollected measurements remain unavailable.")
    return notes


def _rule(row, start, end, weight=0.6):
    return {"row": row, "start": start, "end": end, "weight": weight}


def _span(row, col, rowspan=1, colspan=1):
    return {"row": row, "col": col, "rowspan": rowspan, "colspan": colspan}


def _table2(directory):
    metadata = _metadata(directory)
    by_design = _index(_csv(directory / "table.csv"), ("design",))
    rows = [["Engine", "TFLOPS", "Design", "MMA Shape", "Load", "Max Util", "Rel. Time"]]
    ideal = by_design.get(("Ideal FFT",), {})
    for design in ("Ideal FFT", "cuFFTDx", "Radix-4", "Radix-8", "Radix-16"):
        source = by_design.get((design,), {})
        measured = design == "cuFFTDx"
        # Load=5 is the CUDA FFT arithmetic model, including the cuFFTDx row;
        # it is not an unmeasured timing or utilization estimate.
        load = source.get("arithmetic_coefficient", "")
        if measured and _number(load) is None:
            load = ideal.get("arithmetic_coefficient", "")
        utilization = _number(source.get("utilization_ceiling"))
        util = MISSING if utilization is None else (
            f"{utilization * 100:.1f}%" if measured else f"{utilization * 100:g}%")
        time = _format(source.get("relative_time"))
        if measured:
            if util != MISSING:
                util += "†"
            if time != MISSING:
                time += "†"
        shape = MISSING
        if design.startswith("Radix-"):
            match = re.fullmatch(r"m\d+n(\d+)k(\d+)", source.get("mma_shape", ""))
            if match:
                shape = "$m\\cdot n" + match[1] + "k" + match[2] + "$"
        engine = ("CUDA\nCore" if design == "Ideal FFT" else
                  "Tensor\nCore" if design == "Radix-4" else "")
        peak = _format(source.get("peak_tflops"), digits=0) if engine else ""
        rows.append([engine, peak, design, shape, _format(load, digits=1), util, time])
    notes = ["Load is the coefficient of $N\\log_2 N$. Calculated rows use the paper's A100 FP16 model assumptions.",
             "† cuFFTDx utilization and relative time are measured from Table 3; — means unavailable."]
    if not by_design.get(("cuFFTDx",)):
        notes.append("The cuFFTDx measurement has not been collected; its Load uses the ideal CUDA FFT model.")
    notes.extend(_scope_notes(metadata))
    return {"number": 2,
            "caption": "Theoretical FFT performance model on NVIDIA A100 (FP16).",
            "width": 8.1, "font_size": 10,
            "panels": [{"widths": [1.0, 0.85, 1.35, 1.55, 0.7, 1.0, 1.05], "rows": rows,
                        "header_rows": 1,
                        "spans": [_span(1, 0, rowspan=2), _span(1, 1, rowspan=2),
                                  _span(3, 0, rowspan=3), _span(3, 1, rowspan=3)],
                        "rules": [_rule(1, 0, 7), _rule(3, 0, 7)], "align": {2: "left"}}],
            "notes": notes}


def _time(row):
    if row.get("status") in ("pending", "failed", "excluded", "not_measured"):
        return None
    if str(row.get("passed", "true")).lower() not in ("true", "1"):
        return None
    return _number(row.get("median_time_us"), positive=True)


def _table3(directory):
    metadata = _metadata(directory)
    source = _index(_csv(directory / "table.csv"), ("gpu", "N", "precision", "placement", "library"))
    speedup_path = directory / "speedup.csv"
    speedups = _index(_csv(speedup_path), ("gpu", "N", "thunderfft_precision", "placement")) \
        if speedup_path.is_file() else {}
    panels = []
    for placement, title in (("register", "(a) Inputs in register"), ("shared", "(b) Inputs in shared memory")):
        rows = [["N"] + [""] * 18, [""] * 19, [""] + list(GPUS) * 6]
        rows[0][1], rows[0][10] = "FP16", "TF32 / FP32"
        spans = [_span(0, 0, rowspan=3), _span(0, 1, colspan=9), _span(0, 10, colspan=9)]
        rules = [_rule(1, 1, 10), _rule(1, 10, 19), _rule(3, 0, 19), _rule(10, 0, 19)]
        for offset in (1, 10):
            for group, label in enumerate(("cuFFTDx", "xFFT", "Speedup")):
                col = offset + 3 * group
                rows[1][col] = label
                spans.append(_span(1, col, colspan=3))
                rules.append(_rule(2, col, col + 3))
        ratios = {}
        for n in SIZES:
            row = [str(n)]
            for precision, baseline_precision in (("fp16", "fp16"), ("tf32", "fp32")):
                dx_values, xfft_values, speed_values = [], [], []
                for gpu in GPUS:
                    dx = _time(source.get((gpu, str(n), baseline_precision, placement, "cuFFTDx"), {}))
                    xfft = _time(source.get((gpu, str(n), precision, placement, "ThunderFFT"), {}))
                    ratio = dx / xfft if dx is not None and xfft is not None else None
                    saved = speedups.get((gpu, str(n), precision, placement))
                    if ratio is not None and saved is not None:
                        recorded = _number(saved.get("speedup"), positive=True)
                        if recorded is None or not math.isclose(ratio, recorded, rel_tol=1e-9):
                            raise ValueError("Table 3 speedup disagrees with measured median times")
                        if saved.get("cufftdx_precision") != baseline_precision:
                            raise ValueError("Table 3 speedup precision pairing disagrees with table")
                    if ratio is not None:
                        ratios.setdefault((gpu, precision), []).append(ratio)
                    dx_values.append(_format(dx))
                    xfft_values.append(_format(xfft))
                    speed_values.append(_format(ratio))
                row.extend(dx_values + xfft_values + speed_values)
            rows.append(row)
        average = ["Avg."] + [""] * 18
        for offset, precision in ((7, "fp16"), (16, "tf32")):
            for gpu_index, gpu in enumerate(GPUS):
                values = ratios.get((gpu, precision), [])
                mean = statistics.geometric_mean(values) if len(values) == len(SIZES) else None
                saved = speedups.get((gpu, "geomean(7 sizes)", precision, placement))
                if mean is not None and saved is not None:
                    recorded = _number(saved.get("speedup"), positive=True)
                    if recorded is None or not math.isclose(mean, recorded, rel_tol=1e-9):
                        raise ValueError("Table 3 geometric mean disagrees with seven measured sizes")
                average[offset + gpu_index] = _format(mean)
        rows.append(average)
        vrules = [{"col": col, "row": 0.1, "end": 10.9, "weight": 0.5} for col in (1, 10)]
        vrules += [{"col": col, "row": 2.1, "end": 9.9, "weight": 0.35}
                   for col in (2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18)]
        vrules += [{"col": col, "row": 10.1, "end": 10.9, "weight": 0.35}
                   for col in (8, 9, 17, 18)]
        panels.append({"title": title, "widths": [0.8] + [1.0] * 18,
                       "rows": rows, "spans": spans, "rules": rules,
                       "vrules": vrules, "header_rows": 3, "row_height": 0.23})
    return {"number": 3,
            "caption": "Performance comparison between cuFFTDx and xFFT across FFT sizes and GPU architectures (in $\\mu$s).",
            "width": 13.2, "font_size": 8.5, "panels": panels,
            "notes": ["xFFT denotes ThunderFFT. TF32 xFFT is compared with FP32 cuFFTDx; speedup = cuFFTDx / xFFT.",
                      "Avg. is the geometric mean across all seven FFT sizes; incomplete groups and missing measurements are —."]
                     + _scope_notes(metadata)}


def _table4(directory):
    metadata = _metadata(directory)
    source = _index(_csv(directory / "table.csv"), ("gpu",))
    rows = [[""] + list(GPUS)]
    for label, field in (("Hardware Peak FLOPS", "hardware_peak_tflops"),
                         ("Ideal $m16n8k8$ FLOPS", "ideal_mma_tflops"),
                         ("Unit kernel ($N=64$)", "unit_tflops")):
        row = [label]
        for gpu in GPUS:
            result = source.get((gpu,), {})
            value = result.get(field)
            if field == "ideal_mma_tflops" and result.get("ideal_mma_status") != "measured":
                value = None
            if field == "unit_tflops" and result.get("status") in ("pending", "failed"):
                value = None
            row.append(_format(value, comma=True))
        rows.append(row)
    return {"number": 4,
            "caption": "Reported chip-level peak throughput, measured ideal $m16n8k8$ throughput, and achieved throughput of the 64-point unit kernel (TFLOPS).",
            "width": 6.9, "font_size": 10,
            "panels": [{"widths": [3.2, 1.0, 1.0, 1.25], "rows": rows,
                        "header_rows": 1,
                        "rules": [_rule(1, 0, 4)], "align": {0: "left"}}],
            "notes": ["Hardware peaks are published reference values. Ideal MMA values are measured; unit throughput is derived from Table 3 timings.",
                      "— means unavailable; published ideal/unit throughput targets do not fill missing measurements."]
                     + _scope_notes(metadata)}


def build(evaluation, directory):
    """Return a renderer-independent layout using existing assembled results."""
    builders = {"table2": _table2, "table3": _table3, "table4": _table4}
    if evaluation not in builders:
        raise ValueError("unsupported performance table: " + evaluation)
    return builders[evaluation](Path(directory))
