#!/usr/bin/env python3
"""Prove stage-wide grouping preserves the register permutation; no GPU work."""
import argparse
import itertools
import json
from pathlib import Path


def quartets(stride):
    return [tuple(j + k + t * stride for t in range(4))
            for j in range(0, 32, 4 * stride) for k in range(stride)]


def swap(values, a, b):
    values[a], values[b] = values[b], values[a]


def original_per_quartet(values, groups, pattern):
    out = list(values)
    for a, b, c, d in groups:
        if pattern == 1 or pattern == 3:
            swap(out, a, b)
            swap(out, c, d)
        if pattern == 0 or pattern == 3:
            swap(out, b, d)
        if pattern == 2 or pattern == 3:
            swap(out, a, c)
    return out


def grouped_conditions(values, groups, pattern):
    out = list(values)
    if pattern == 1 or pattern == 3:
        for a, b, c, d in groups:
            swap(out, a, b)
            swap(out, c, d)
    if pattern == 0 or pattern == 3:
        for a, b, c, d in groups:
            swap(out, b, d)
    if pattern == 2 or pattern == 3:
        for a, b, c, d in groups:
            swap(out, a, c)
    return out


def grouped_switch(values, groups, pattern):
    out = list(values)
    for a, b, c, d in groups:
        if pattern == 0:
            swap(out, b, d)
        elif pattern == 1:
            swap(out, a, b)
            swap(out, c, d)
        elif pattern == 2:
            swap(out, a, c)
        else:
            swap(out, a, d)
            swap(out, b, c)
    return out


def prove():
    expected_permutations = ((0, 3, 2, 1), (1, 0, 3, 2),
                             (2, 1, 0, 3), (3, 2, 1, 0))
    cases = 0
    for stride in (1, 4):
        groups = quartets(stride)
        covered = [index for group in groups for index in group]
        assert len(groups) == 8 and sorted(covered) == list(range(32))
        for pattern in range(4):
            for order in itertools.permutations(range(4)):
                # Distinct symbols represent arbitrary complete complex values;
                # this proof needs neither finite arithmetic nor FFT input data.
                values = [None] * 32
                for group_id, group in enumerate(groups):
                    for index, component in zip(group, order):
                        values[index] = (group_id, component)
                reference = original_per_quartet(values, groups, pattern)
                assert grouped_conditions(values, groups, pattern) == reference
                assert grouped_switch(values, groups, pattern) == reference
                for group in groups:
                    before = [values[index] for index in group]
                    assert [reference[index] for index in group] == [
                        before[index] for index in expected_permutations[pattern]]
                cases += 1
    return {
        "status": "PASS",
        "claim": "Grouping each conditional swap across disjoint quartets preserves all register values exactly",
        "strides": [1, 4],
        "quartets_per_stage": 8,
        "register_values_per_lane": 32,
        "lane_pattern_domain": [0, 1, 2, 3],
        "pattern_domain_reason": "Native code uses lane & 3",
        "symbolic_orders_per_quartet": 24,
        "stage_pattern_order_cases": cases,
        "register_value_comparisons_per_candidate": cases * 32,
        "quartets_are_disjoint_and_cover_all_registers": True,
        "three_group_conditions_equivalent": True,
        "one_switch_diagnostic_equivalent": True,
        "arbitrary_symbolic_input_equivalence": True,
        "floating_point_operations_in_permutation": 0,
        "gpu_work_performed": False,
        "limitation": "Mathematical permutation proof only; does not execute the CUDA source or validate compiled scheduling, FFT arithmetic, or performance"
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    text = json.dumps(prove(), indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
