#!/usr/bin/env python3
"""CPU-only checks that incomplete or invalid numerical results cannot pass."""

import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


FIELDS = ["N", "batch", "mode", "alpha", "precision", "rel_l2", "max_abs", "rel_linf", "all_finite"]
NAMES = ["accuracy_reduced", "accuracy_fp32_cufft", "range_fp16", "range_tf32", "range_fp32_cufft",
         "accuracy_fp16_cufftdx", "range_fp16_cufftdx"]


def fixture():
    metadata = {
        "profile": "smoke", "N": [64], "batch": 64, "seed": 1234,
        "implementations": {"fp16": "ThunderFFT", "tf32": "ThunderFFT", "fp32": "cuFFT C2C", "fp16_cufftdx": "cuFFTDx"},
        "selected_phases": NAMES, "range_mode": "full",
        "range_exponents": {"fp16": [8, 13], "tf32": [120, 127], "fp32": [120, 127], "fp16_cufftdx": [8, 13]},
    }

    def row(precision, alpha=1, error=1e-4, finite=1):
        return dict(zip(FIELDS, [64, 64, "gaussian", alpha, precision, error, error, error, finite]))

    rows = {
        "accuracy_reduced": [row("fp16"), row("tf32")],
        "accuracy_fp32_cufft": [row("fp32", error=1e-7)],
        "accuracy_fp16_cufftdx": [row("fp16_cufftdx", error=1e-3)],
    }
    for precision, name, boundary in (("fp16", "range_fp16", 11), ("tf32", "range_tf32", 123),
                                       ("fp32", "range_fp32_cufft", 123), ("fp16_cufftdx", "range_fp16_cufftdx", 11)):
        lower, upper = metadata["range_exponents"][precision]
        rows[name] = [row(precision, alpha=2.0**exponent, error="nan", finite=int(exponent <= boundary))
                      for exponent in range(lower, upper + 1)]
    environments = {name: {"gpu_name": "synthetic", "cufft_version": 11303} for name in NAMES}
    return metadata, rows, environments


class SummaryChecks(unittest.TestCase):
    def run_fixture(self, mutate=None, passes=False):
        metadata, rows, environments = copy.deepcopy(fixture())
        if mutate:
            mutate(rows, environments)
        with tempfile.TemporaryDirectory(prefix="thunderfft-table5-check-") as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps(metadata))
            for name in NAMES:
                with (root / f"{name}.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows[name])
                if name in environments:
                    (root / f"{name}.environment.json").write_text(json.dumps(environments[name]))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(root)],
                                    text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0 if passes else 1, result.stdout + result.stderr)
            validation = json.loads((root / "validation.json").read_text())
            self.assertEqual(validation["numerical_sanity"], "passed" if passes else "failed")
            for filename in ("data.csv", "table.csv", "table.md", "comparison.csv"):
                self.assertTrue((root / filename).is_file(), filename)

    def test_complete_data_passes(self):
        self.run_fixture(passes=True)

    def test_missing_accuracy_row_fails(self):
        self.run_fixture(lambda rows, _: rows["accuracy_reduced"].pop())

    def test_missing_cufftdx_range_row_fails(self):
        self.run_fixture(lambda rows, _: rows["range_fp16_cufftdx"].pop())

    def test_duplicate_scan_row_fails(self):
        self.run_fixture(lambda rows, _: rows["range_fp16"].append(rows["range_fp16"][0]))

    def test_large_accuracy_error_fails(self):
        self.run_fixture(lambda rows, _: rows["accuracy_reduced"][0].update(rel_l2=0.28))

    def test_nonmonotonic_boundary_fails(self):
        self.run_fixture(lambda rows, _: rows["range_tf32"][-1].update(all_finite=1))

    def test_changed_environment_fails(self):
        self.run_fixture(lambda _, environments: environments["range_tf32"].update(gpu_name="different GPU"))

    def test_missing_environment_fails(self):
        self.run_fixture(lambda _, environments: environments.pop("range_tf32"))


if __name__ == "__main__":
    unittest.main()
