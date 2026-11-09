#!/usr/bin/env python3
"""Plot runtime versus filter size from summary.csv (Figure 4; reused by Figure 5)."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


SERIES = ("cuDNN", "FlashFFTConv", "cuFFT", "cuFFTDx", "ThunderFFT")
COLORS = {"cuDNN": "#7f7f7f", "FlashFFTConv": "#6a3d9a", "cuFFT": "#1f5fa8", "cuFFTDx": "#186b24",
          "ThunderFFT": "#ff4d00"}
LABELS = {"figure4": ("Filter size (f)", "Figure 4: 1D convolution"),
          "figure5": ("Filter size (f x f)", "Figure 5: 2D convolution")}


class Pending(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path):
    if not path.is_file():
        raise Pending(f"missing {path}; run measure.py first")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["filter"].isdecimal()]
    if not rows:
        raise Pending("summary.csv has no per-filter rows")
    return rows


def make_plot(rows, output, evaluation, policy_skipped=()):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Pending("matplotlib is required: python3 -m pip install -r artifact/requirements.txt") from error
    xlabel, title = LABELS[evaluation]
    values = {(row["library"], int(row["filter"])): float(row["median_time_ms"]) for row in rows}
    for item in policy_skipped:
        if "item" in item:
            library, separator, size = item["item"].partition(":filter=")
            if not separator:
                raise ValueError("invalid skipped filter cell")
        else:
            library, size = item["library"], item["filter"]
        # Both native bundles and assembled coverage retain the skipped x axis
        # cells, even when no other measured library spans the tail.
        values.setdefault((library, int(size)), math.nan)
    filters = sorted({size for _, size in values})
    libraries = [library for library in SERIES if any(key[0] == library for key in values)]
    libraries += sorted({key[0] for key in values} - set(SERIES))
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axis = plt.subplots(figsize=(7.0, 3.6))
    for library in libraries:
        # Unmeasured points are gaps, not zeros or interpolated values.
        axis.plot(filters, [values.get((library, size), math.nan) for size in filters], marker="o", markersize=3,
                  linewidth=1.6, label=library, color=COLORS.get(library))
    axis.set_xticks(filters[::2] if len(filters) > 14 else filters)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Time (ms)")
    finite = [v for v in values.values() if math.isfinite(v) and v > 0]
    if evaluation == "figure5" and finite:
        # Direct convolution grows quadratically with the filter; a log axis keeps the FFT paths readable.
        from matplotlib.ticker import NullLocator, ScalarFormatter
        axis.set_yscale("log")
        axis.set_yticks([y for y in (0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
                         if min(finite) / 1.5 <= y <= max(finite) * 1.5])
        axis.yaxis.set_major_formatter(ScalarFormatter())
        axis.yaxis.set_minor_locator(NullLocator())
    else:
        axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.6)
    axis.set_axisbelow(True)
    figure.legend(*axis.get_legend_handles_labels(), loc="upper center", ncol=len(libraries), frameon=False)
    if policy_skipped:
        figure.text(0.5, 0.015, "Fast mode: cuDNN tail omitted; use --full for all filters.",
                    ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0.065 if policy_skipped else 0, 1, 0.88))
    figure.savefig(output, metadata={"CreationDate": None, "ModDate": None, "Title": title})
    plt.close(figure)
    return libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation", choices=tuple(LABELS), default="figure4")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        rows = read_summary(output / "summary.csv")
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        policy_skipped = metadata.get("policy_skipped", [])
        libraries = make_plot(rows, output / "figure.pdf", args.evaluation, policy_skipped=policy_skipped)
        metadata["plot"] = {"command": [sys.executable, *sys.argv], "script_sha256": sha256(Path(__file__)),
                            "series": libraries, "statistic": "median over trials",
                            "policy_skipped": policy_skipped,
                            "outputs": {name: sha256(output / name) for name in ("data.csv", "summary.csv", "figure.pdf")
                                        if (output / name).is_file()}}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"{LABELS[args.evaluation][1]}: {len(rows)} points; outputs: {output}")
        return 0
    except Pending as error:
        print(f"PENDING [{args.evaluation}]: {error}", file=sys.stderr)
        return 78
    except (OSError, ValueError, KeyError) as error:
        print(f"FAILED [{args.evaluation}]: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main())
