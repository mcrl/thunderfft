#!/usr/bin/env python3
"""Normalize Figure 4 library selectors before optional build targets are chosen."""
import sys

LIBRARIES = ("ThunderFFT", "cuDNN", "FlashFFTConv")


def normalize(value, allowed=LIBRARIES):
    selected = tuple(token.strip() for token in value.split(","))
    if selected == ("auto",):
        return "auto"
    if not selected or len(selected) != len(set(selected)) or any(token not in allowed for token in selected):
        raise ValueError("libraries must be auto or unique comma-separated names: " + ",".join(allowed))
    return ",".join(selected)


if __name__ == "__main__":
    try:
        print(normalize(sys.argv[1] if len(sys.argv) == 2 else "auto"))
    except ValueError as error:
        raise SystemExit(str(error))
