#!/usr/bin/env python3
"""Collect Figure 5: correctness-gated 2D convolution (cuDNN, cuFFT full image,
tiled cuFFTDx) over a filter-size sweep, joined with backend CSVs (tiled ThunderFFT).

Uses the shared sweep collector in artifact/figure4/measure.py.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import re


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SWEEP = module("figure5_sweep", HERE.parent / "figure4/measure.py")
FIGURE5 = {
    "evaluation": "figure5", "title": "Figure 5", "env": "THUNDERFFT_FIGURE5",
    # The earlier H100 evaluation used a 16384 x 16384 image and odd f = 3..53.
    "filters": {"smoke": (3, 33, 53), "paper": tuple(range(3, 54, 2))},
    "allowed_filters": tuple(range(3, 60, 2)),
    "operation": "complex fp32 valid 2D cross-correlation y[i][j]=sum_{a,b<f} x[i+a][j+b] h[a][b], "
                 "(N-f+1)^2 outputs, U[0,1] parts (x: minstd seed 42, h: seed 42+f), filter FFT timed, "
                 "tile 64 for tiled paths",
    "precision": "fp32",
    "truncate": ("cuDNN",),
    "libraries": ("cuFFT", "cuFFTDx", "ThunderFFT", "cuDNN"),
    "notes": {"ThunderFFT": "artifact implementation: complex FP32 storage, current public N=64 TF32 tensor-core FFT arithmetic, cuFFT filter FFT timed; original paper binary identity is unconfirmed", "cuDNN": "as in the earlier evaluation: real FP32 single channel (real parts of x and h), "
                       "CUDNN_CROSS_CORRELATION, fastest of the top three cudnnGetConvolutionForwardAlgorithm_v7 candidates (cudnnFind exhausts host memory at f>=31); no filter transform"},
}


def size(profile):
    return int(os.environ.get("THUNDERFFT_FIGURE5_N", 16384 if profile == "paper" else 2048))


def command(library):
    def build(binary, sizes, profile, trials):
        n = size(profile)
        return [str(binary), str(n), ",".join(map(str, sizes)), library, "5", "20", str(trials)], f"N={n}"
    return build


def gate(block):
    if "Convolution validation: PASS" not in block:
        raise ValueError("native 2D convolution correctness gate did not pass")
    rel = re.search(r"vs direct reference: L2_rel=(\S+) samples=(\d+)", block)
    if not rel:
        raise ValueError("native output lacks validation lines")
    return float(rel.group(1)), f"{rel.group(2)} sampled outputs"


def parse(stdout):
    """Per-filter (size, trial, time_ms, rel_l2, validated) from the thff_conv output."""
    return SWEEP.parse_blocks(stdout, gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="thff_conv")
    parser.add_argument("--available", default="cuFFT,ThunderFFT", help="compiled backend names")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    if args.profile == "paper" and size("paper") != 16384:
        parser.error("paper profile requires N=16384; use smoke for custom sizes")
    available = tuple(args.available.split(","))
    runners = [(library, args.binary, command(library.lower()), parse, "n/a" if library == "cuDNN" else "true")
               for library in FIGURE5["libraries"] if library in available]
    sources = [*ROOT.glob("include/**/*.cuh"), ROOT / "sample/utils.h", ROOT / "sample/conv2d/main.cu", ROOT / "sample/conv2d/cufftdx.cu", ROOT / "sample/conv2d/thunder_tiled64.cu",
               ROOT / "sample/conv2d/CMakeLists.txt", *HERE.glob("*.py"), HERE / "run.sh",
               HERE.parent / "figure4/measure.py"]
    try:
        return SWEEP.collect(FIGURE5, runners, args.build_dir, args.output, args.profile, sources)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
