#!/usr/bin/env python3
"""Generate Figure 3 (normalized end-to-end FFT performance) from summary.csv."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


SERIES = ("Comm. only", "cuFFT", "cuFFTDx", "ThunderFFT")
COLORS = {"Comm. only": "#7f7f7f", "cuFFT": "#1f5fa8", "cuFFTDx": "#186b24", "ThunderFFT": "#ff4d00"}
# Bars group by storage width; cuFFT/cuFFTDx FP32 share TF32's float2 storage.
STORAGE = {"tf32": "FP32 storage", "fp32": "FP32 storage", "fp16": "FP16 storage"}


class Pending(Exception):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_summary(path):
    if not path.is_file():
        raise Pending(f"missing {path}; run measure.py first")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["library"] in SERIES and row["normalized"]]
    if not rows:
        raise Pending("summary.csv has no normalized rows")
    return rows


def make_plot(rows, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise Pending("matplotlib is required: python3 -m pip install -r artifact/requirements.txt") from error
    panels = sorted({(row["gpu"], STORAGE[row["precision"]]) for row in rows}) or [("", "Measurements pending")]
    sizes = sorted({64, 256, 1024, 4096} | {int(row["N"]) for row in rows})
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
    figure, axes = plt.subplots(1, len(panels), figsize=(7.0 * len(panels), 3.6), squeeze=False)
    width = 0.8 / len(SERIES)
    for axis, (gpu, panel) in zip(axes[0], panels):
        values = {(row["library"], int(row["N"])): float(row["normalized"])
                  for row in rows if row["gpu"] == gpu and STORAGE[row["precision"]] == panel}
        for index, library in enumerate(SERIES):
            # Missing baselines are gaps (NaN), never zero bars.
            heights = [values.get((library, size), math.nan) for size in sizes]
            axis.bar([x + (index - 1.5) * width for x in range(len(sizes))], heights, width,
                     label=library, color=COLORS[library])
        axis.set_xticks(range(len(sizes)), [str(size) for size in sizes])
        axis.set_xlabel("FFT size")
        axis.set_ylabel("Normalized Performance")
        finite_values = [value for value in values.values() if math.isfinite(value)]
        axis.set_ylim(0, max(1.05, max(finite_values, default=1) * 1.05))
        axis.set_title(f"{gpu} {panel}".strip(), fontsize=10)
        if not values:
            axis.text(0.5, 0.5, "Collect communication and FFT CSVs to normalize", ha="center", va="center", transform=axis.transAxes)
        axis.grid(axis="y", color="#d7d7d7", linewidth=0.6)
        axis.set_axisbelow(True)
    figure.legend(*axes[0][0].get_legend_handles_labels(), loc="upper right", ncol=len(SERIES), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output, metadata={"CreationDate": None, "ModDate": None,
                                     "Title": "Figure 3: end-to-end FFT normalized by communication-only"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true", help="Write a pending placeholder when communication rows are not collected")
    args = parser.parse_args()
    try:
        output = args.output_dir.resolve()
        try:
            rows = read_summary(output / "summary.csv")
        except Pending:
            if not args.allow_pending:
                raise
            rows = []
        make_plot(rows, output / "figure.pdf")
        metadata_path = output / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
        metadata["plot"] = {"command": [sys.executable, *sys.argv], "script_sha256": sha256(Path(__file__)),
                            "series": [library for library in SERIES if any(row["library"] == library for row in rows)],
                            "outputs": {name: sha256(output / name) for name in ("data.csv", "summary.csv", "figure.pdf")
                                        if (output / name).is_file()}}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"Figure 3: {len(rows)} normalized bars; outputs: {output}")
        return 0
    except Pending as error:
        print(f"PENDING [figure3]: {error}", file=sys.stderr)
        return 78
    except (OSError, ValueError, KeyError) as error:
        print(f"FAILED [figure3]: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main())
