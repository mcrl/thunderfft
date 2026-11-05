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
import sys


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
                 "(N-f+1)^2 outputs, seed 42 U[0,1] parts, filter FFT timed, tile 64 for tiled paths",
    "precision": "fp32",
    "notes": {"cuDNN": "as in the earlier evaluation: real FP32 single channel (real parts of x and h), "
                       "CUDNN_CROSS_CORRELATION, fastest of the top three cudnnGetConvolutionForwardAlgorithm_v7 candidates (cudnnFind exhausts host memory at f>=31); no filter transform"},
}


def size(profile):
    return int(os.environ.get("THUNDERFFT_FIGURE5_N", 16384 if profile == "paper" else 2048))


def command(library):
    def build(binary, filter_size, profile):
        n = size(profile)
        return [str(binary), str(n), str(filter_size), library, "5", "20"], f"N={n}"
    return build


def parse(stdout):
    """Return (time_ms, rel_l2, validated) from the thff_conv output."""
    if "Convolution validation: PASS" not in stdout:
        raise ValueError("native 2D convolution correctness gate did not pass")
    rel = re.search(r"vs direct reference: L2_rel=(\S+) samples=(\d+)", stdout)
    time = re.search(r"time: (\S+) ms$", stdout, re.MULTILINE)
    if not (rel and time):
        raise ValueError("native output lacks validation or timing lines")
    return float(time.group(1)), float(rel.group(1)), f"{rel.group(2)} sampled outputs"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True, help="thff_conv")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    if args.profile == "paper" and size("paper") != 16384:
        parser.error("paper profile requires N=16384; use smoke for custom sizes")
    runners = [(library, args.binary, command(library.lower()), parse, "true") for library in ("cuFFT", "cuFFTDx")]
    runners.append(("cuDNN", args.binary, command("cudnn"), parse, "n/a"))
    sources = [ROOT / "sample/conv2d/main.cu", ROOT / "sample/conv2d/cufftdx.cu",
               ROOT / "sample/conv2d/CMakeLists.txt", *HERE.glob("*.py"), HERE / "run.sh",
               HERE.parent / "figure4/measure.py"]
    try:
        return SWEEP.collect(FIGURE5, runners, args.build_dir, args.output, args.profile, sources)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
