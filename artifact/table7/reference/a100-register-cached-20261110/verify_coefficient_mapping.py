#!/usr/bin/env python3
"""Exhaustive symbolic proof of the 28-pair branch FFT coefficient cache.

The original 51e346c fill_reg_b<64> uses one complex W64 entry, with a
lane-dependent component order and sign. Comparing those symbolic expressions
proves equality for any supplied LUT, including its exact FP32 rounding.
No GPU, CUDA library, or numerical tolerance is involved.
"""

import json
from itertools import permutations


def original_expression(stage, butterfly, lane):
    """Original fill_reg_b<64> index and the two component/sign expressions."""
    stride = 1 << (2 * stage)
    i_perm = (butterfly // stride) % 4
    j_perm = ((butterfly // (stride // 4)) % 4) if stride >= 4 else 0
    k = butterfly % stride
    i = (lane // 8 - i_perm) & 3
    j = (lane % 4 - j_perm) & 3
    index = (1 << (4 - 2 * stage)) * (
        (j * (k + stride * i)) & (4 * stride - 1)
    )
    assert 0 <= index < 64
    if (lane // 4) & 1:
        return ((index, "y", 1), (index, "x", 1))
    return ((index, "x", 1), (index, "y", -1))


def cache_slot(stage, butterfly):
    """All stage-1 pairs plus stage-2 pairs whose j is not divisible by 4."""
    if stage == 0:
        return 4 * (butterfly % 4)
    if stage == 1:
        return butterfly
    if butterfly % 4 == 0:
        return butterfly // 4
    return 16 + 3 * (butterfly // 4) + butterfly % 4 - 1


def cache_source(slot):
    if slot < 16:
        return (1, slot)
    offset = slot - 16
    return (2, 4 * (offset // 3) + offset % 3 + 1)


def original_switch_permutation(values, pattern):
    """The four original 51e346c switch arms, represented symbolically."""
    a, b, c, d = values
    return ((a, d, c, b), (b, a, d, c),
            (c, b, a, d), (d, c, b, a))[pattern & 3]


def checkpoint_three_if_permutation(values, pattern):
    """The restored 652808b helper; callers constrain pattern to lane & 3."""
    a, b, c, d = values
    if pattern == 1 or pattern == 3:
        a, b = b, a
        c, d = d, c
    if pattern == 0 or pattern == 3:
        b, d = d, b
    if pattern == 2 or pattern == 3:
        a, c = c, a
    return a, b, c, d


def main():
    permutation_cases = 0
    # Distinct symbolic labels establish the exact index permutation, independent
    # of data values, scalar type, or floating-point representation. Also exercise
    # all 24 input orders for all four allowed lane patterns.
    for values in permutations(range(4)):
        for pattern in range(4):
            assert checkpoint_three_if_permutation(values, pattern) == (
                original_switch_permutation(values, pattern)
            ), (values, pattern)
            permutation_cases += 1
    assert permutation_cases == 96

    signatures = {}
    cases = 0
    cache_slots_used = set()
    stage_unique_counts = []
    for stage in range(3):
        stage_signatures = set()
        for butterfly in range(16):
            signature = tuple(original_expression(stage, butterfly, lane)
                              for lane in range(32))
            signatures.setdefault(signature, []).append((stage, butterfly))
            stage_signatures.add(signature)
            slot = cache_slot(stage, butterfly)
            assert 0 <= slot < 28
            cache_slots_used.add(slot)
            cached_stage, cached_butterfly = cache_source(slot)
            for lane in range(32):
                assert original_expression(stage, butterfly, lane) == (
                    original_expression(cached_stage, cached_butterfly, lane)
                ), (stage, butterfly, lane, slot)
                cases += 1
        stage_unique_counts.append(len(stage_signatures))

    assert cases == 3 * 16 * 32 == 1536
    assert len(signatures) == 28
    assert cache_slots_used == set(range(28))
    assert len({tuple(original_expression(*cache_source(slot), lane)
                      for lane in range(32)) for slot in range(28)}) == 28

    print(json.dumps({
        "status": "PASS",
        "proof": "Exact LUT index, component order, and sign equality for all lanes and MMA positions",
        "lanes": 32,
        "stages": 3,
        "butterflies_per_stage": 16,
        "pair_cases_checked": cases,
        "scalar_component_cases_checked": 2 * cases,
        "original_pair_positions": 48,
        "unique_pairs_within_each_stage": stage_unique_counts,
        "unique_pairs_across_stages": len(signatures),
        "cached_fp32_scalar_count": 56,
        "arbitrary_lut_bitwise_equivalence": True,
        "numerical_trig_symmetry_assumed": False,
        "checkpoint_permutation_proof": {
            "original": "51e346c four-way switch",
            "candidate": "652808b three conditional swap groups",
            "pattern_domain": [0, 1, 2, 3],
            "pattern_domain_reason": "Caller supplies lane & 3",
            "all_symbolic_input_orders": 24,
            "cases_checked": permutation_cases,
            "exact_index_permutations": [
                list(checkpoint_three_if_permutation(range(4), pattern))
                for pattern in range(4)
            ],
            "status": "PASS",
        },
        "stage_slot_mapping": [
            [cache_slot(stage, butterfly) for butterfly in range(16)]
            for stage in range(3)
        ],
        "cache_sources": [
            {"slot": slot, "stage": cache_source(slot)[0],
             "butterfly": cache_source(slot)[1]}
            for slot in range(28)
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
