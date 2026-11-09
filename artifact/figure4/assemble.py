#!/usr/bin/env python3
"""Assemble validated Figure 4 library/filter parts without rerunning GPUs."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


SHARED = module("figure4_assembly_shared", HERE.parent / "figure5/assemble.py")
MEASURE = module("figure4_assembly_measure", HERE / "measure.py")


def assemble(parts, output, profile):
    return SHARED.assemble_sweep(parts, output, profile, MEASURE.FIGURE4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parts", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    print(json.dumps(assemble(args.parts, args.output, args.profile), indent=2))
