"""Measured-value mapping and honest missing/bounded cells in paper tables."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import paper_table_accuracy as tables


class PaperTableAccuracyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write(self, rows, metadata=None):
        fields = list(dict.fromkeys(key for row in rows for key in row)) or ["N"]
        with (self.directory / "table.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        if metadata is not None:
            (self.directory / "metadata.json").write_text(json.dumps(metadata))

    def test_table5_four_distinct_series_in_paper_order(self):
        rows = []
        for index, precision in enumerate(("tf32", "fp32_cufftdx", "fp16_cufftdx", "fp16")):
            rows.append({"N": 64, "precision": precision, "rel_l2_alpha1": (index + 1) * 1e-4,
                         "accuracy_status": "complete", "range_status": "complete",
                         "boundary_status": "observed", "largest_finite_exponent_observed": index + 8})
        self.write(rows)
        layout = tables.build("table5", self.directory)
        panel = layout["panels"][0]
        self.assertEqual(panel["rows"][1][1:5], ["FP16\nxFFT", "FP16\ncuFFTDx", "TF32\nxFFT", "FP32\ncuFFTDx"])
        self.assertEqual(panel["rows"][2][1:5],
                         [r"$4.00\times 10^{-4}$", r"$3.00\times 10^{-4}$",
                          r"$1.00\times 10^{-4}$", r"$2.00\times 10^{-4}$"])
        self.assertEqual(panel["rows"][2][5:], ["$2^{11}$", "$2^{10}$", "$2^{8}$", "$2^{9}$"])
        self.assertEqual([row[0] for row in panel["rows"][2:]], ["64", "256", "1024", "4096"])
        self.assertEqual(panel["rows"][3][1:], [tables.MISSING] * 8)

    def test_table5_censored_scans_are_bounds_and_boundary_recheck_is_disclosed(self):
        self.write([
            {"N": 64, "precision": "fp16", "range_status": "complete", "boundary_status": "right_censored",
             "scan_exponent_max": 10, "largest_finite_exponent_observed": 10},
            {"N": 64, "precision": "fp16_cufftdx", "range_status": "complete", "boundary_status": "left_censored",
             "scan_exponent_min": 8},
            {"N": 64, "precision": "tf32", "range_status": "pending", "boundary_status": "observed",
             "largest_finite_exponent_observed": 123, "range_coverage": "boundary_recheck"}])
        layout = tables.build("table5", self.directory)
        self.assertEqual(layout["panels"][0]["rows"][2][5:], [r"$\geq 2^{10}$", "$<2^{8}$", "—", "—"])
        self.assertIn("measured exponent window", " ".join(layout["notes"]))

    def test_table5_only_completed_accuracy_and_finite_numbers_are_shown(self):
        self.write([
            {"N": 64, "precision": "fp16", "accuracy_status": "pending", "rel_l2_alpha1": 0.005},
            {"N": 64, "precision": "fp16_cufftdx", "accuracy_status": "complete", "rel_l2_alpha1": "nan"},
            {"N": 64, "precision": "tf32", "accuracy_status": "complete", "rel_l2_alpha1": 0},
            {"N": 64, "precision": "fp32_cufftdx", "accuracy_status": "complete", "rel_l2_alpha1": 9.999e-8}])
        row = tables.build("table5", self.directory)["panels"][0]["rows"][2]
        self.assertEqual(row[1:5], ["—", "—", "$0$", r"$1.00\times 10^{-7}$"])

    def test_table6_uses_a100_in_paper_library_order_and_measurement_units(self):
        self.write([
            {"gpu": "B200", "library": "ThunderFFT", "median_time_ms": 99, "speedup_vs_cufftdx": 99},
            {"gpu": "A100", "library": "ThunderFFT", "median_time_ms": 0.016432, "speedup_vs_cufftdx": 1.456},
            {"gpu": "A100", "library": "cuFFTDx", "median_time_ms": 0.023, "speedup_vs_cufftdx": 1},
            {"gpu": "A100", "library": "tcFFT", "median_time_ms": 0.021, "speedup_vs_cufftdx": 1.11}])
        rows = tables.build("table6", self.directory)["panels"][0]["rows"]
        self.assertEqual([row[0] for row in rows[1:]], ["cuFFTDx", "tcFFT", "FlashFFTConv", "xFFT"])
        self.assertEqual(rows[-1], ["xFFT", "0.016", "1.46"])
        self.assertEqual(rows[3], ["FlashFFTConv", "—", "—"])

    def test_table6_does_not_use_paper_reference_or_speedup_without_baseline(self):
        self.write([{"gpu": "A100", "library": "cuFFTDx", "status": "pending", "paper_time_ms": .023},
                    {"gpu": "A100", "library": "ThunderFFT", "status": "collected", "median_time_ms": .016,
                     "speedup_vs_cufftdx": 1.46, "paper_speedup": 1.46}])
        rows = tables.build("table6", self.directory)["panels"][0]["rows"]
        self.assertEqual(rows[1][1:], ["—", "—"])
        self.assertEqual(rows[-1][1:], ["0.016", "—"])

    def test_table7_missing_baseline_never_creates_speedups_or_promotes_microbenchmark(self):
        self.write([
            {"variant": "register_branch", "median_time_ms": .007912, "utilization_pct": 33.2,
             "timing_status": "collected", "accuracy_status": "passed", "utilization_status": "calculated",
             "radix16_speedup": .95},
            {"variant": "historical_e4aa1b0", "median_time_ms": .000123,
             "timing_status": "collected_unvalidated", "accuracy_status": "unchecked"}])
        layout = tables.build("table7", self.directory)
        rows = layout["panels"][0]["rows"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[1][1:], ["—", "—", "—"])
        self.assertEqual(rows[2][1:], ["—", "—", "—"])
        self.assertEqual(rows[3][1:], ["0.0079", "33%", "—"])
        self.assertIn("kernel_comparison.csv", " ".join(layout["notes"]))
        self.assertIn("identities remain unconfirmed", " ".join(layout["notes"]))

    def test_table7_unchecked_timing_is_marked_and_cannot_supply_speedup(self):
        self.write([
            {"variant": "radix16", "median_time_ms": .0074, "timing_status": "collected",
             "accuracy_status": "passed", "radix16_speedup": 1},
            {"variant": "register_uniform", "median_time_ms": .003512, "utilization_pct": 75.2,
             "timing_status": "collected_unvalidated", "accuracy_status": "unchecked",
             "utilization_status": "calculated_unvalidated", "radix16_speedup": 2.18}])
        layout = tables.build("table7", self.directory)
        row = layout["panels"][0]["rows"][-1]
        self.assertTrue(row[0].endswith("†"))
        self.assertEqual(row[1:], ["0.0035", "75%", "—"])
        self.assertIn("correctness was not checked", " ".join(layout["notes"]))

    def test_render_spec_is_deterministic_and_does_not_mutate_csv(self):
        self.write([{"N": 64, "precision": "fp16", "accuracy_status": "complete", "rel_l2_alpha1": .000434}])
        before = (self.directory / "table.csv").read_bytes()
        first = tables.build("table5", self.directory)
        self.assertEqual(first, tables.build("table5", self.directory))
        self.assertEqual(before, (self.directory / "table.csv").read_bytes())

    def test_duplicate_measurement_keys_rejected_instead_of_silently_overwritten(self):
        self.write([{"N": 64, "precision": "fp16"}, {"N": 64, "precision": "fp16"}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            tables.build("table5", self.directory)


if __name__ == "__main__":
    unittest.main()
