#!/usr/bin/env python3
"""Select a logical CUDA device without escaping an inherited visibility mask."""

import os
import sys


def resolve(ordinal, environment):
    if not ordinal.isdecimal():
        raise ValueError("device must be a nonnegative CUDA ordinal")
    index = int(ordinal)
    if "CUDA_VISIBLE_DEVICES" not in environment:
        return str(index)
    visible = [token.strip() for token in environment["CUDA_VISIBLE_DEVICES"].split(",")]
    if any(not token or token == "-1" for token in visible):
        raise ValueError("inherited CUDA_VISIBLE_DEVICES is empty or disables GPU visibility")
    if index >= len(visible):
        raise ValueError(f"logical device {index} is outside the {len(visible)} inherited visible device(s)")
    return visible[index]


if __name__ == "__main__":
    try:
        print(resolve(sys.argv[1], os.environ))
    except (IndexError, ValueError) as error:
        print(f"Cannot select Table 5 CUDA device: {error}", file=sys.stderr)
        raise SystemExit(78)
