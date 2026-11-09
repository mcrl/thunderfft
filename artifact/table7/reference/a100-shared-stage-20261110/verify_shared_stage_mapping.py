#!/usr/bin/env python3
"""Independent FP64 model of the proposed natural shared-stage MMA indexing.

This proves the FFT64 index/coefficient mapping. It does not emulate TF32
rounding, GPU memory ordering, compiled instructions, or the actual CUDA file.
Native R=1 validation and sanitizer evidence remain necessary.
"""
import json
import numpy as np


def reverse_base4_digits(index):
    return ((index & 3) << 4) | (index & 12) | ((index >> 4) & 3)


def fft64_shared_stage(value):
    data = np.asarray([value[reverse_base4_digits(i)] for i in range(64)])
    for stride in (1, 4, 16):
        next_data = np.empty_like(data)
        for butterfly in range(16):
            offset = butterfly % stride
            base = (butterfly // stride) * (4 * stride) + offset
            values = data[base + stride * np.arange(4)]
            # m16n8k8 A: a0/a2 at K=q/q+4; a1/a3 are the other row.
            # q<2 uses real components; q>=2 uses imaginary components.
            a = np.asarray([
                values[0].real, values[2].real,
                values[0].imag, values[2].imag,
                values[1].real, values[3].real,
                values[1].imag, values[3].imag,
            ])
            b = np.empty((8, 8))
            for raw_k in range(8):
                # B b0/b1 at K=q/q+4. The historical helper starts at
                # j=2*q and 2*q+1, then applies its scalar embedding map.
                helper_j = 2 * (raw_k % 4) + raw_k // 4
                j = (helper_j % 4) * 2 + helper_j // 4
                for raw_n in range(8):
                    i = (raw_n % 4) * 2 + raw_n // 4
                    index = ((j // 2) * (offset + stride * (i // 2))
                             + stride * (i & 1) - stride * (j & 1))
                    b[raw_k, raw_n] = np.cos(
                        -2 * np.pi * (index % (4 * stride)) / (4 * stride))
            d = a @ b
            # D: each lane q owns columns 2*q, 2*q+1. This yields
            # [Re0,Re1,Re2,Re3,Im0,Im1,Im2,Im3].
            next_data[base + stride * np.arange(4)] = d[:4] + 1j * d[4:]
        data = next_data
    return data


def main():
    rng = np.random.default_rng(123)
    cases = [row.astype(complex) for row in np.eye(64)]
    cases += [1j * row for row in np.eye(64)]
    cases += [rng.normal(size=64) + 1j * rng.normal(size=64)]
    errors = []
    for value in cases:
        expected = np.fft.fft(value)
        actual = fft64_shared_stage(value)
        errors.append((float(np.max(np.abs(actual - expected))),
                       float(np.linalg.norm(actual - expected)
                             / np.linalg.norm(expected))))
    result = {
        "cases": len(cases),
        "real_basis_cases": 64,
        "imaginary_basis_cases": 64,
        "random_cases": 1,
        "max_absolute_error": max(x[0] for x in errors),
        "max_relative_l2": max(x[1] for x in errors),
        "threshold_relative_l2": 1e-14,
        "scope": "Independent FP64 index/coefficient model; no TF32 or GPU simulation",
    }
    result["passed"] = result["max_relative_l2"] < 1e-14
    print(json.dumps(result, indent=2))
    assert result["passed"]


if __name__ == "__main__":
    main()
