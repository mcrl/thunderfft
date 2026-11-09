#!/usr/bin/env python3
"""Plot runtime versus filter size from summary.csv (Figure 5; reused by Figure 4)."""

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
LABELS = {"figure5": ("Filter size (f)", "Figure 5: 1D convolution"),
          "figure4": ("Filter size (f x f)", "Figure 4: 2D convolution")}
PAPER_GPUS = {"figure5": "A100", "figure4": "H100"}


class Pending(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path, evaluation=None):
    if not path.is_file():
        raise Pending(f"missing {path}; run measure.py first")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = [row for row in reader if row["filter"].isdecimal()]
        fields = reader.fieldnames or []
    if evaluation and ("library" in fields) != (evaluation == "figure5"):
        raise ValueError("summary layout does not match the current figure's convolution dimension; "
                         "old Figure 4/5 bundles keep their original numbering")
    if "library" in fields:
        rows = [{"library": row["library"], "filter": row["filter"], "time_ms": row["median_time_ms"]} for row in rows]
    else:
        # Wide per-filter mean table (Figure 4): one lower-case column per library.
        names = {name.lower(): name for name in SERIES}
        rows = [{"library": names.get(column, column), "filter": row["filter"], "time_ms": row[column] or "nan"}
                for row in rows for column in fields if column != "filter"]
    if not rows:
        raise Pending("summary.csv has no per-filter rows")
    return rows


def statistic(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return "median over trials" if "library" in next(csv.reader(stream), []) else "mean over trials"


def caption(evaluation, measured_gpu):
    """Keep the paper's figure identity separate from the measured hardware."""
    title = LABELS[evaluation][1]
    paper_gpu = PAPER_GPUS[evaluation]
    if not measured_gpu:
        return f"{title}; measured GPU unspecified (paper: {paper_gpu})"
    if measured_gpu == paper_gpu:
        return f"{title} on {measured_gpu}"
    return f"{title} on {measured_gpu} (paper: {paper_gpu})"


def make_plot(rows, output, evaluation, policy_skipped=(), measured_gpu=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Pending("matplotlib is required: python3 -m pip install -r artifact/requirements.txt") from error
    xlabel, _ = LABELS[evaluation]
    title = caption(evaluation, measured_gpu)
    values = {(row["library"], int(row["filter"])): float(row["time_ms"]) for row in rows}
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
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    for library in libraries:
        # Unmeasured points are gaps, not zeros or interpolated values.
        axis.plot(filters, [values.get((library, size), math.nan) for size in filters], marker="o", markersize=3,
                  linewidth=1.6, label=library, color=COLORS.get(library))
    axis.set_xticks(filters[::2] if len(filters) > 14 else filters)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Time (ms)")
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#d7d7d7", linewidth=0.6)
    axis.set_axisbelow(True)
    figure.legend(*axis.get_legend_handles_labels(), loc="upper center", ncol=len(libraries), frameon=False)
    figure.text(0.5, 0.015, title, ha="center", va="bottom", fontsize=9)
    if policy_skipped:
        figure.text(0.5, 0.055, "Fast mode: cuDNN tail omitted; use --full for all filters.",
                    ha="center", va="bottom", fontsize=8)
    figure.tight_layout(rect=(0, 0.12 if policy_skipped else 0.08, 1, 0.88))
    figure.savefig(output, metadata={"CreationDate": None, "ModDate": None, "Title": title})
    plt.close(figure)
    return libraries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation", choices=tuple(LABELS), default="figure5")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        rows = read_summary(output / "summary.csv", args.evaluation)
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        if metadata.get("evaluation", args.evaluation) != args.evaluation:
            raise ValueError("result evaluation ID disagrees with the requested figure")
        policy_skipped = metadata.get("policy_skipped", [])
        measured_gpu = metadata.get("gpu") or (metadata.get("identity") or {}).get("gpu")
        libraries = make_plot(rows, output / "figure.pdf", args.evaluation, policy_skipped=policy_skipped,
                              measured_gpu=measured_gpu)
        metadata["plot"] = {"command": [sys.executable, *sys.argv], "script_sha256": sha256(Path(__file__)),
                            "series": libraries, "statistic": statistic(output / "summary.csv"),
                            "caption": caption(args.evaluation, measured_gpu),
                            "measured_gpu": measured_gpu, "paper_gpu": PAPER_GPUS[args.evaluation],
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
