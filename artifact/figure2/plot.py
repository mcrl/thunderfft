#!/usr/bin/env python3
"""Generate Figure 2 (A100, register inputs) from Table 3 output.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys


SIZES = (64, 128, 256, 512, 1024, 2048, 4096)
PRECISIONS = ("TF32", "FP16")
CUFFTDX_PRECISIONS = ("FP32", "FP16")
MATCHED = {"TF32": "FP32", "FP16": "FP16"}
MATCHED_INV = {dx: thunder for thunder, dx in MATCHED.items()}
SERIES = (("ThunderFFT", "TF32"), ("ThunderFFT", "FP16"), ("cuFFTDx", "FP32"), ("cuFFTDx", "FP16"))
LIBRARY_COLORS = {"ThunderFFT": "#ff4d00", "cuFFTDx": "#186b24"}
COLORS = {series: LIBRARY_COLORS[series[0]] for series in SERIES}
Y_TICKS_MS = tuple(0.002 * 4 ** k for k in range(6))
Y_RANGE_MS = (Y_TICKS_MS[0], Y_TICKS_MS[-1])
REQUIRED = {"gpu", "N", "batch", "placement", "library", "precision", "trial",
            "time_us", "e2e_out_of_place_us", "rel_l2", "passed", "inside_repeats"}


class Pending(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_directory(explicit):
    if explicit:
        return explicit.resolve()
    dependencies = json.loads(os.environ.get("THUNDERFFT_DEPENDENCY_RESULTS", "{}"))
    if not isinstance(dependencies, dict):
        raise ValueError("THUNDERFFT_DEPENDENCY_RESULTS must be a JSON object")
    source = dependencies.get("table3")
    if not source:
        raise Pending("missing completed Table 3 dependency; use ./run.sh figure2")
    if not isinstance(source, str):
        raise ValueError("Table 3 dependency must be a result-directory path")
    return Path(source).resolve()


def check_row(row, profile):
    """Validate one A100/register row; return (size, precision, trial, time_us)."""
    if any(row.get(field) is None for field in REQUIRED):
        raise ValueError("Table 3 CSV contains a row with missing field values")
    size, trial = int(row["N"]), int(row["trial"])
    precision = row["precision"].upper()
    allowed = PRECISIONS if row["library"] == "ThunderFFT" else CUFFTDX_PRECISIONS
    if size not in SIZES or precision not in allowed:
        raise ValueError(f"unexpected A100/register {row['library']} configuration: N={size}, precision={precision}")
    if trial < 0:
        raise ValueError("trial must be nonnegative")
    batch, repeat = int(row["batch"]), int(row["inside_repeats"])
    if batch <= 0 or repeat <= 0:
        raise ValueError("batch and inside_repeats must be positive")
    if profile == "paper" and (batch != 65536 or repeat != 1000):
        raise ValueError("paper Figure 2 requires batch 65536 and R=1000")
    if row["passed"].strip().lower() not in ("true", "1"):
        raise ValueError(f"unvalidated source row: {row['library']} N={size}, precision={precision}, trial={trial}")
    time_us = float(row["time_us"])
    e2e_us = float(row["e2e_out_of_place_us"])
    rel_l2 = float(row["rel_l2"])
    if not math.isfinite(time_us) or time_us <= 0:
        raise ValueError("source compute time must be finite and positive")
    if not math.isfinite(e2e_us) or e2e_us <= 0:
        raise ValueError("source end-to-end time must be finite and positive")
    if not math.isfinite(rel_l2) or not 0 <= rel_l2 <= 0.005:
        raise ValueError("source relative L2 must be finite and at most 0.005")
    return size, precision, trial, time_us


def read_source(directory, profile):
    status_path = directory / "status.json"
    data_path = directory / "data.csv"
    if not status_path.is_file() or not data_path.is_file():
        raise Pending(f"Table 3 status.json or data.csv is missing: {directory}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("evaluation") != "table3" or status.get("status") != "completed":
        raise ValueError("source must be a completed Table 3 evaluation")
    if status.get("profile") != profile:
        raise ValueError("Table 3 source profile does not match requested Figure 2 profile")
    if status.get("exit_code") != 0:
        raise ValueError("completed Table 3 source must have exit_code 0")
    with data_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError("Table 3 CSV has duplicate column names")
        missing = REQUIRED - set(fields)
        if missing:
            raise ValueError("missing Table 3 fields: " + ", ".join(sorted(missing)))
        all_rows = list(reader)
    if not all_rows:
        raise ValueError("Table 3 data.csv is empty")
    selected_rows = [row for row in all_rows if row["gpu"] == "A100" and row["placement"] == "register"]
    thunder = [row for row in selected_rows if row["library"] == "ThunderFFT"]
    dx = [row for row in selected_rows if row["library"] == "cuFFTDx"]
    if not thunder:
        raise Pending("Figure 2 requires measured A100/register ThunderFFT rows; this source has none")
    if dx and "candidate" not in fields:
        raise ValueError("Table 3 cuFFTDx rows need a candidate column")
    groups, candidates = {}, {}
    batches, repeats = set(), set()
    for row in thunder + dx:
        size, precision, trial, time_us = check_row(row, profile)
        batches.add(int(row["batch"]))
        repeats.add(int(row["inside_repeats"]))
        if row["library"] == "ThunderFFT":
            trials = groups.setdefault(("ThunderFFT", size, precision), {})
        else:
            if not row["candidate"]:
                raise ValueError("cuFFTDx row without a candidate label")
            trials = candidates.setdefault((size, precision), {}).setdefault(row["candidate"], {})
        if trial in trials:
            raise ValueError(f"duplicate source trial: {row['library']} N={size}, precision={precision}, trial={trial}")
        trials[trial] = time_us
    expected = {("ThunderFFT", size, precision) for size in SIZES for precision in PRECISIONS}
    if set(groups) != expected:
        raise ValueError("incomplete A100/register coverage; missing " + str(sorted(expected - set(groups))))
    selected = {}
    if dx:
        expected_dx = {(size, precision) for size in SIZES for precision in CUFFTDX_PRECISIONS}
        if set(candidates) != expected_dx:
            raise ValueError("incomplete A100/register cuFFTDx coverage; missing "
                             + str(sorted(expected_dx - set(candidates))))
        for (size, precision), by_candidate in sorted(candidates.items()):
            # Table 3's rule: smallest median compute time; ties keep source order.
            best = min(by_candidate, key=lambda name: statistics.median(by_candidate[name].values()))
            selected[(size, precision)] = best
            groups[("cuFFTDx", size, precision)] = by_candidate[best]
            for name, trials in by_candidate.items():
                if tuple(sorted(trials)) != tuple(sorted(groups[("ThunderFFT", size, MATCHED_INV[precision])])):
                    raise ValueError(f"cuFFTDx candidate {name} N={size} {precision} has different trials")
        check_table3_selection(directory, selected)
    trial_sets = {tuple(sorted(values)) for values in groups.values()}
    if len(trial_sets) != 1:
        raise ValueError("inconsistent trial coverage across Figure 2 configurations")
    if len(batches) != 1 or len(repeats) != 1:
        raise ValueError("Figure 2 requires one common batch and inside_repeats value")
    rows = thunder + [row for row in dx
                      if row["candidate"] == selected[(int(row["N"]), row["precision"].upper())]]
    order = {series: index for index, series in enumerate(SERIES)}
    rows.sort(key=lambda row: (int(row["N"]), order[(row["library"], row["precision"].upper())], int(row["trial"])))
    provenance = {"source_run": directory.name, "source_result_dir": str(directory),
                  "source_file": str(data_path), "source_sha256": sha256(data_path),
                  "source_status_sha256": sha256(status_path)}
    metadata_path = directory / "metadata.json"
    if metadata_path.is_file():
        provenance["source_metadata_file"] = str(metadata_path)
        provenance["source_metadata_sha256"] = sha256(metadata_path)
    return fields, rows, groups, provenance, selected


def check_table3_selection(directory, selected):
    """When Table 3 wrote its candidate table, its A100/register choice must agree."""
    path = directory / "cufftdx_candidates.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["gpu"] != "A100" or row["placement"] != "register" or row["selected"] != "true":
                continue
            key = int(row["N"]), row["precision"].upper()
            if selected.get(key) != row["candidate"]:
                raise ValueError(f"cuFFTDx selection for N={key[0]} {key[1]} disagrees with Table 3")


def make_plot(groups, output, with_cufftdx, allow_partial=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Pending("matplotlib is required: python3 -m pip install -r artifact/figure2/requirements.txt") from error
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(8.6, 5.0))
    plotted = []
    for library, precision in SERIES:
        if library == "cuFFTDx" and not with_cufftdx:
            continue
        values = [list(groups.get((library, size, precision), {}).values()) if allow_partial
                  else list(groups[(library, size, precision)].values()) for size in SIZES]
        if not any(values):
            continue
        # NaNs are intentional gaps: missing measurements are neither zero nor interpolated.
        medians = [statistics.median(value) / 1000.0 if value else math.nan for value in values]
        errors = [[mid - min(value) / 1000.0 if value else math.nan for mid, value in zip(medians, values)],
                  [max(value) / 1000.0 - mid if value else math.nan for mid, value in zip(medians, values)]]
        plotted += [time / 1000.0 for value in values for time in value]
        wide = precision in ("TF32", "FP32")
        color = COLORS[(library, precision)]
        axis.errorbar(range(len(SIZES)), medians, yerr=errors, label=f"{library} ({precision})", color=color,
                      marker="o" if wide else "s", markersize=5,
                      markerfacecolor=color, linestyle="-" if wide else "--",
                      linewidth=1.8, capsize=3)
    axis.set_yscale("log")
    fixed = Y_RANGE_MS[0] <= min(plotted) and max(plotted) <= Y_RANGE_MS[1]
    if fixed:
        from matplotlib.ticker import NullLocator
        axis.set_ylim(*Y_RANGE_MS)
        axis.set_yticks(Y_TICKS_MS, [f"{tick:g}" for tick in Y_TICKS_MS])
        axis.yaxis.set_minor_locator(NullLocator())
    axis.set_xticks(range(len(SIZES)), [str(size) for size in SIZES])
    axis.set_xlabel("FFT size")
    axis.set_ylabel("Compute time (ms, logarithmic)")
    axis.grid(axis="y", which="major" if fixed else "both", color="#d7d7d7", linewidth=0.6, alpha=0.8)
    axis.set_axisbelow(True)
    figure.legend(*axis.get_legend_handles_labels(), loc="upper right", ncol=2,
                  bbox_to_anchor=(0.98, 0.995), frameon=False, fontsize=10)
    figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.90))
    # No visible title; the PDF metadata still distinguishes the ThunderFFT-only subset.
    title = "Figure 2: A100, register inputs" if with_cufftdx else "Figure 2 subset: A100, register inputs"
    figure.savefig(output, metadata={"CreationDate": None, "ModDate": None, "Title": title})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path,
                        help="completed Table 3 result directory (normally supplied by the runner)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    try:
        directory = source_directory(args.source_dir)
        fields, rows, groups, provenance, selected = read_source(directory, args.profile)
        with_cufftdx = bool(selected)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=True)
        extra_fields = list(provenance) + ["time_ms"]
        if set(fields) & set(extra_fields):
            raise ValueError("Table 3 input has reserved provenance column names")
        batch, repeats = int(rows[0]["batch"]), int(rows[0]["inside_repeats"])
        make_plot(groups, output / "figure.pdf", with_cufftdx)
        with (output / "data.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields + extra_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({**row, **provenance, "time_ms": float(row["time_us"]) / 1000.0})
        with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["gpu", "N", "batch", "placement", "library",
                "precision", "candidate", "trials", "median_us", "min_us", "max_us", "median_ms",
                "speedup_vs_cufftdx", "source_run", "source_sha256"])
            writer.writeheader()
            for size in SIZES:
                for library, precision in SERIES:
                    if (library, size, precision) not in groups:
                        continue
                    values = list(groups[(library, size, precision)].values())
                    median = statistics.median(values)
                    speedup = ""
                    if library == "ThunderFFT" and with_cufftdx:
                        speedup = statistics.median(groups[("cuFFTDx", size, MATCHED[precision])].values()) / median
                    writer.writerow({"gpu": "A100", "N": size, "batch": batch, "placement": "register",
                        "library": library, "precision": precision,
                        "candidate": selected.get((size, precision), "") if library == "cuFFTDx" else "",
                        "trials": len(values), "median_us": median, "min_us": min(values), "max_us": max(values),
                        "median_ms": median / 1000.0, "speedup_vs_cufftdx": speedup,
                        "source_run": provenance["source_run"], "source_sha256": provenance["source_sha256"]})
        metadata = {"schema_version": 2, "evaluation": "figure2", "profile": args.profile,
                    "scope": ("A100/register Figure 2: ThunderFFT and cuFFTDx" if with_cufftdx
                              else "ThunderFFT-only A100/register Figure 2 subset"),
                    "scientific_comparison": ("cuFFTDx and ThunderFFT from the same Table 3 run" if with_cufftdx
                                              else "incomplete: cuFFTDx measurements are not included"),
                    "not_included": [] if with_cufftdx else ["cuFFTDx FP16", "cuFFTDx FP32"],
                    "cufftdx_selection": ({f"N{size}-{precision}": name
                                           for (size, precision), name in sorted(selected.items())}
                                          if with_cufftdx else None),
                    "cufftdx_selection_rule": "smallest median compute time among Table 3 candidates",
                    "pairing": MATCHED, "speedup": "cuFFTDx median / ThunderFFT median",
                    "source": provenance, "raw_rows": len(rows), "configurations": len(groups),
                    "batch": batch, "inside_repeats": repeats,
                    "timing_metric": "Table 3 compute_us = (t_2R_us - t_R_us) / R",
                    "central_statistic": "median of trial compute_us",
                    "error_bars": "minimum and maximum of trial compute_us",
                    "unit_conversion": "time_ms = time_us / 1000",
                    "series_colors": {f"{library} {precision}": COLORS[(library, precision)]
                                      for library, precision in SERIES},
                    "plot_command": [sys.executable, *sys.argv],
                    "plot_script_sha256": sha256(Path(__file__)),
                    "outputs": {name: sha256(output / name)
                                for name in ("data.csv", "summary.csv", "figure.pdf")}}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        kind = "ThunderFFT and cuFFTDx" if with_cufftdx else "ThunderFFT-only subset"
        print(f"Figure 2 {kind}: {len(groups)} configurations, {len(rows)} validated source rows")
        print(f"Source: {directory}; sha256={provenance['source_sha256']}")
        print(f"Outputs: {output}" + ("" if with_cufftdx else "; cuFFTDx comparison remains pending"))
        return 0
    except Pending as error:
        print(f"PENDING [figure2]: {error}", file=sys.stderr)
        return 78
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAILED [figure2]: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main())
