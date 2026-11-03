#!/usr/bin/env python3
"""Reject corrupt correctness evidence, without running a GPU."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import validate


def fixture():
    rows = []
    for precision in ("fp16", "tf32"):
        for n in (64, 128, 256, 512, 1024, 2048, 4096):
            for direction in ("forward", "inverse"):
                for path, batch in (("direct_core_guarded", 16), ("public_global", 16), ("public_global", 1)):
                    rows.append({"precision": precision, "N": str(n), "direction": direction,
                                 "path": path, "batch": str(batch), "input": "gaussian_rms1", "relative_l2": "0.001",
                                 "pass": "1", "nonfinite_outputs": "0",
                                 "shared_guard_changed_max": "0", "output_guard_changed": "0"})
    rechecks = [r for r in rows if r["N"] == "64" and r["path"] == "public_global" and r["batch"] == "1"]
    return rows + copy.deepcopy(rechecks)


class CorrectnessEvidenceChecks(unittest.TestCase):
    def test_complete_matrix(self):
        self.assertEqual(len(validate(fixture(), "smoke")), 28)

    def test_missing_case_fails(self):
        with self.assertRaises(ValueError):
            validate(fixture()[:-1], "smoke")

    def test_numerical_and_guard_failures(self):
        for field, value in (("relative_l2", "nan"), ("relative_l2", "0.28"),
                             ("pass", "0"), ("nonfinite_outputs", "1"),
                             ("shared_guard_changed_max", "1"), ("output_guard_changed", "1")):
            with self.subTest(field=field, value=value):
                rows = fixture()
                rows[0][field] = value
                with self.assertRaises(ValueError):
                    validate(rows, "smoke")

    def test_missing_direction_coverage_fails(self):
        rows = fixture()
        for row in rows:
            row["direction"] = "forward"
        with self.assertRaises(ValueError):
            validate(rows, "smoke")

    def test_equal_count_with_replaced_input_fails(self):
        rows = fixture()
        rows[0]["input"] = "constant"
        with self.assertRaises(ValueError):
            validate(rows, "smoke")


if __name__ == "__main__":
    unittest.main()
