"""Measured-value mapping and honest missing/bounded cells in paper tables."""
import csv
import hashlib
import importlib.util
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
        self.assertEqual(len(layout["panels"]), 2)
        panel = layout["panels"][0]
        ranges = layout["panels"][1]
        self.assertEqual(panel["title"], r"(a) Relative $L_2$ error at $\alpha=1$.")
        self.assertEqual(ranges["title"], r"(b) Largest $\alpha=2^k$ with all finite outputs.")
        for part in (panel, ranges):
            self.assertEqual(part["rows"][0], ["", "FP16", "", "TF32 / FP32", ""])
            self.assertEqual(part["rows"][1], ["$N$", "xFFT", "cuFFTDx", "xFFT", "cuFFTDx"])
            self.assertEqual(part["spans"], [
                {"row": 0, "col": 1, "rowspan": 1, "colspan": 2},
                {"row": 0, "col": 3, "rowspan": 1, "colspan": 2}])
            self.assertEqual([row[0] for row in part["rows"][2:]], ["64", "256", "1024", "4096"])
            self.assertEqual(part["rows"][3][1:], [tables.MISSING] * 4)
        self.assertEqual(panel["rows"][2][1:5],
                         [r"$4.00\times 10^{-4}$", r"$3.00\times 10^{-4}$",
                          r"$1.00\times 10^{-4}$", r"$2.00\times 10^{-4}$"])
        self.assertEqual(ranges["rows"][2][1:], ["$2^{11}$", "$2^{10}$", "$2^{8}$", "$2^{9}$"])

    def test_table5_censored_scans_are_bounds_and_boundary_recheck_is_disclosed(self):
        self.write([
            {"N": 64, "precision": "fp16", "range_status": "complete", "boundary_status": "right_censored",
             "scan_exponent_max": 10, "largest_finite_exponent_observed": 10},
            {"N": 64, "precision": "fp16_cufftdx", "range_status": "complete", "boundary_status": "left_censored",
             "scan_exponent_min": 8},
            {"N": 64, "precision": "tf32", "range_status": "pending", "boundary_status": "observed",
             "largest_finite_exponent_observed": 123, "range_coverage": "boundary_recheck"}])
        layout = tables.build("table5", self.directory)
        self.assertEqual(layout["panels"][1]["rows"][2][1:], [r"$\geq 2^{10}$", "$<2^{8}$", "—", "—"])
        self.assertIn("measured exponent window", " ".join(layout["notes"]))

    def test_table5_only_completed_accuracy_and_finite_numbers_are_shown(self):
        self.write([
            {"N": 64, "precision": "fp16", "accuracy_status": "pending", "rel_l2_alpha1": 0.005},
            {"N": 64, "precision": "fp16_cufftdx", "accuracy_status": "complete", "rel_l2_alpha1": "nan"},
            {"N": 64, "precision": "tf32", "accuracy_status": "complete", "rel_l2_alpha1": 0},
            {"N": 64, "precision": "fp32_cufftdx", "accuracy_status": "complete", "rel_l2_alpha1": 9.999e-8}])
        row = tables.build("table5", self.directory)["panels"][0]["rows"][2]
        self.assertEqual(row[1:5], ["—", "—", "$0$", r"$1.00\times 10^{-7}$"])

    def comparisons(self, evaluation, panel=0):
        return {(item["row"], item["col"]): item
                for item in tables.build(evaluation, self.directory)["panels"][panel]["comparisons"]}

    def test_table5_latest_paper_references_include_four_series_and_actual_alpha(self):
        self.write([
            {"N": 64, "precision": "fp16", "accuracy_status": "complete", "rel_l2_alpha1": .000434123,
             "range_status": "complete", "boundary_status": "observed", "largest_finite_exponent_observed": 10},
            {"N": 64, "precision": "fp16_cufftdx", "accuracy_status": "complete", "rel_l2_alpha1": .00051,
             "range_status": "complete", "boundary_status": "observed", "largest_finite_exponent_observed": 11}])
        cells = self.comparisons("table5")
        ranges = self.comparisons("table5", panel=1)
        self.assertEqual(len(cells), 16)
        self.assertEqual(len(ranges), 16)
        self.assertEqual(cells[2, 1]["paper"], 4.34e-4)
        self.assertEqual(cells[2, 1]["current"], .000434123)
        self.assertEqual(ranges[2, 1]["paper"], 2048)
        self.assertEqual(ranges[2, 1]["current"], 1024)
        self.assertEqual(ranges[2, 1]["paper_text"], "$2^{11}$")
        self.assertEqual(ranges[5, 4]["paper"], 2 ** 120)
        self.assertEqual(cells[2, 2]["paper"], 8.36e-4)
        self.assertEqual(cells[2, 2]["paper_text"], r"$8.36\times 10^{-4}$")
        self.assertEqual(ranges[2, 2]["paper"], 2048)
        for panel in (cells, ranges):
            self.assertIsNotNone(panel[2, 2]["current"])
            self.assertEqual(panel[2, 2]["status"], "measured")
        self.assertIsNone(cells[3, 1]["current"])
        self.assertEqual(cells[3, 1]["status"], "missing_measurement")

    def test_table5_reference_grid_matches_latest_printed_values(self):
        self.write([])
        errors = self.comparisons("table5")
        ranges = self.comparisons("table5", panel=1)
        # Independently transcribed from Table 5(a)/(b), PDF (4), page 9.
        expected = (
            (4.34e-4, 8.36e-4, 1.35e-3, 1.24e-7),
            (6.03e-4, 1.13e-3, 1.36e-3, 1.78e-7),
            (6.09e-4, 1.32e-3, 2.12e-3, 1.91e-7),
            (6.54e-4, 1.51e-3, 2.64e-3, 2.06e-7))
        for index, row in enumerate(expected):
            for column, value in enumerate(row, start=1):
                self.assertEqual(errors[index + 2, column]["paper"], value)
                self.assertEqual(ranges[index + 2, column]["paper"],
                                 2 ** ((11 if column <= 2 else 123) - index))
                for panel in (errors, ranges):
                    self.assertIsNone(panel[index + 2, column]["current"])
                    self.assertEqual(panel[index + 2, column]["status"], "missing_measurement")

    def test_table5_censored_comparisons_preserve_bounds_without_exact_values(self):
        self.write([
            {"N": 64, "precision": "fp16", "range_status": "complete", "boundary_status": "right_censored",
             "scan_exponent_max": 10},
            {"N": 64, "precision": "tf32", "range_status": "complete", "boundary_status": "left_censored",
             "scan_exponent_min": 122}])
        cells = self.comparisons("table5", panel=1)
        for column, status, text in ((1, "censored_right", r"$\geq 2^{10}$"),
                                     (3, "censored_left", "$<2^{122}$")):
            self.assertIsNone(cells[2, column]["current"])
            self.assertIsNotNone(cells[2, column]["paper"])
            self.assertEqual(cells[2, column]["current_text"], text)
            self.assertEqual(cells[2, column]["status"], status)

    def test_table5_comparison_does_not_promote_pending_or_invalid_accuracy(self):
        self.write([
            {"N": 64, "precision": "fp16", "accuracy_status": "pending", "rel_l2_alpha1": .0004},
            {"N": 64, "precision": "tf32", "accuracy_status": "complete", "rel_l2_alpha1": -1}])
        cells = self.comparisons("table5")
        for column in (1, 3):
            self.assertIsNone(cells[2, column]["current"])
            self.assertEqual(cells[2, column]["status"], "missing_measurement")

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

    def test_table6_comparison_preserves_raw_timings_and_printed_paper_speedups(self):
        self.write([
            {"gpu": "A100", "library": "cuFFTDx", "status": "collected", "median_time_ms": .023123,
             "speedup_vs_cufftdx": 1},
            {"gpu": "A100", "library": "tcFFT", "status": "collected", "median_time_ms": .021,
             "speedup_vs_cufftdx": 1.101095238095238},
            {"gpu": "A100", "library": "ThunderFFT", "status": "collected", "median_time_ms": .016432,
             "speedup_vs_cufftdx": 1.4071920642648492}])
        cells = self.comparisons("table6")
        self.assertEqual(len(cells), 8)
        self.assertEqual(cells[4, 1]["paper"], .016)
        self.assertEqual(cells[4, 1]["current"], .016432)
        self.assertEqual(cells[4, 1]["current_text"], "0.016")
        self.assertEqual(cells[2, 2]["paper"], 1.11)
        self.assertNotAlmostEqual(cells[2, 2]["paper"], .023 / .021)
        self.assertEqual(cells[3, 1]["paper"], .018)
        self.assertIsNone(cells[3, 1]["current"])
        self.assertEqual(cells[3, 1]["status"], "missing_measurement")

    def test_table6_comparison_without_measured_baseline_has_no_current_speedup(self):
        self.write([{"gpu": "A100", "library": "ThunderFFT", "status": "collected",
                     "median_time_ms": .016432, "speedup_vs_cufftdx": 1.46}])
        cells = self.comparisons("table6")
        self.assertEqual(cells[4, 2]["paper"], 1.46)
        self.assertIsNone(cells[4, 2]["current"])
        self.assertEqual(cells[4, 2]["current_text"], tables.MISSING)

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

    def test_table7_estimated_tcfft_baseline_is_labeled_and_uses_projection_speedups(self):
        self.write([
            {"variant": "radix16", "timing_status": "estimated", "utilization_status": "estimated",
             "accuracy_status": "not_applicable_projection", "estimated_time_ms": .007875,
             "estimated_utilization_pct": 35.6, "median_time_ms": "", "utilization_pct": ""},
            {"variant": "register_branch", "median_time_ms": .004,
             "timing_status": "collected", "accuracy_status": "passed",
             "utilization_status": "calculated", "utilization_pct": 70.1,
             "speedup_vs_estimated_radix16": 1.96875, "radix16_speedup": 99},
            {"variant": "register_uniform", "median_time_ms": .0035,
             "timing_status": "collected_unvalidated", "accuracy_status": "unchecked",
             "speedup_vs_estimated_radix16": 2.25},
            {"variant": "historical_e4aa1b0", "median_time_ms": .0001,
             "timing_status": "collected", "accuracy_status": "passed",
             "speedup_vs_estimated_radix16": 78.75}])
        layout = tables.build("table7", self.directory)
        rows = layout["panels"][0]["rows"]
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0][-1], "Speedup*")
        self.assertEqual(rows[1], ["radix-16 (tcFFT)*", "0.0079", "36%", "1.00"])
        self.assertEqual(rows[2][1:], ["—", "—", "—"])
        self.assertEqual(rows[3][1:], ["0.0040", "70%", "1.97"])
        self.assertEqual(rows[4][-1], "—")
        notes = " ".join(layout["notes"])
        self.assertIn("utilization measured in Table 6", notes)
        self.assertIn("first-row time is estimated", notes)
        self.assertIn("identities remain unconfirmed", notes)
        self.assertIn("kernel_comparison.csv", notes)

    def test_table7_invalid_projection_does_not_create_baseline_cells_or_speedups(self):
        baseline = {"variant": "radix16", "timing_status": "estimated",
                    "utilization_status": "estimated", "accuracy_status": "not_applicable_projection",
                    "estimated_time_ms": .007875, "estimated_utilization_pct": 35.6}
        for field, value in (("timing_status", "pending"), ("utilization_status", "calculated"),
                             ("accuracy_status", "passed"), ("estimated_time_ms", ""),
                             ("estimated_time_ms", "nan"), ("estimated_time_ms", "inf"),
                             ("estimated_time_ms", -1), ("estimated_time_ms", 0),
                             ("estimated_utilization_pct", ""), ("estimated_utilization_pct", "nan"),
                             ("estimated_utilization_pct", "inf"), ("estimated_utilization_pct", 0),
                             ("estimated_utilization_pct", -1)):
            with self.subTest(field=field, value=value):
                self.write([dict(baseline, **{field: value}),
                            {"variant": "register_branch", "median_time_ms": .004,
                             "timing_status": "collected", "accuracy_status": "passed",
                             "speedup_vs_estimated_radix16": 1.96875, "radix16_speedup": 99}])
                rows = tables.build("table7", self.directory)["panels"][0]["rows"]
                self.assertEqual(rows[0][-1], "Speedup")
                self.assertEqual(rows[1], ["radix-16 (tcFFT)", "—", "—", "—"])
                self.assertEqual(rows[3][-1], "—")

    def test_table7_projection_speedups_require_validated_positive_measurements(self):
        baseline = {"variant": "radix16", "timing_status": "estimated",
                    "utilization_status": "estimated", "accuracy_status": "not_applicable_projection",
                    "estimated_time_ms": .007875, "estimated_utilization_pct": 35.6}
        measured = {"variant": "register_branch", "median_time_ms": .004,
                    "timing_status": "collected", "accuracy_status": "passed",
                    "speedup_vs_estimated_radix16": 1.96875}
        for field, value in (("timing_status", "pending"), ("accuracy_status", "unchecked"),
                             ("median_time_ms", ""), ("median_time_ms", "nan"),
                             ("median_time_ms", "inf"), ("median_time_ms", 0),
                             ("speedup_vs_estimated_radix16", ""),
                             ("speedup_vs_estimated_radix16", "inf"),
                             ("speedup_vs_estimated_radix16", -1)):
            with self.subTest(field=field, value=value):
                self.write([baseline, dict(measured, **{field: value})])
                rows = tables.build("table7", self.directory)["panels"][0]["rows"]
                self.assertEqual(rows[3][-1], "—")

    def selected_table7(self):
        baseline = {"variant": "radix16", "timing_status": "estimated",
                    "utilization_status": "estimated", "accuracy_status": "not_applicable_projection",
                    "estimated_time_ms": .008161125, "estimated_utilization_pct": 63.25378}
        selected = {"variant": "register_branch", "source_variant": "historical_e4aa1b0",
                    "source_commit": "e4aa1b0bf7d38493fcb8d5ff7c1ffd599c7d6375",
                    "implementation_kind": "historical_partial_kernel", "paper_identity": "unconfirmed",
                    "operation_coverage": "partial_fft_operand_setup_omitted",
                    "timing_status": "collected_unvalidated", "accuracy_status": "unchecked",
                    "utilization_status": "calculated_unvalidated", "utilization_pct": 32.718268,
                    "median_time_ms": .00788889598846,
                    "speedup_vs_estimated_radix16": .008161125 / .00788889598846}
        metadata = {"full_paper_comparison": False, "paper_row_sources": {
            "register_branch": {"source_variant": "historical_e4aa1b0",
                                "selection_status": "artifact_selected",
                                "selection_basis": "Historical function-limited kernel for this artifact row."}}}
        return baseline, selected, metadata

    def test_table7_selected_partial_source_has_commit_scope_and_timing_ratio(self):
        baseline, selected, metadata = self.selected_table7()
        self.write([baseline, selected], metadata)
        layout = tables.build("table7", self.directory)
        rows = layout["panels"][0]["rows"]
        self.assertEqual(rows[0][-1], "Speedup*")
        self.assertEqual(rows[3], ["+ reducing communication (e4aa1b0)†", "0.0079", "33%", "1.03"])
        notes = " ".join(layout["notes"])
        for text in ("MMA B-fragment initialization", "shared-memory round trip", "FFT outputs are undefined",
                     "unit-kernel timings", "complete-FFT speedup", "identities remain unconfirmed"):
            self.assertIn(text, notes)

    def test_table7_comparison_keeps_estimation_and_partial_fft_status(self):
        baseline, selected, metadata = self.selected_table7()
        self.write([baseline, selected], metadata)
        cells = self.comparisons("table7")
        self.assertEqual(len(cells), 12)
        self.assertEqual([cells[1, col]["paper"] for col in (1, 2, 3)], [.0074, 70, 1])
        self.assertEqual(cells[1, 1]["current"], .008161125)
        self.assertEqual(cells[1, 2]["current"], 63.25378)
        self.assertEqual(cells[1, 1]["status"], "estimated")
        self.assertEqual(cells[3, 1]["status"], "timing_only_partial_fft")
        self.assertEqual(cells[3, 1]["current"], .00788889598846)
        self.assertEqual(cells[3, 3]["status"], "timing_only_partial_fft_estimated_baseline")
        self.assertEqual(cells[3, 3]["current"], baseline["estimated_time_ms"] / selected["median_time_ms"])
        self.assertEqual(cells[3, 3]["paper"], .95)
        self.assertEqual(cells[2, 1]["paper"], .0120)
        self.assertIsNone(cells[2, 1]["current"])
        self.assertEqual(cells[2, 1]["status"], "missing_measurement")

    def test_table7_comparison_unvalidated_timing_cannot_create_full_fft_speedup(self):
        self.write([
            {"variant": "radix16", "median_time_ms": .0074, "timing_status": "collected",
             "accuracy_status": "passed", "radix16_speedup": 1},
            {"variant": "register_uniform", "median_time_ms": .003512, "utilization_pct": 75.2,
             "timing_status": "collected_unvalidated", "accuracy_status": "unchecked",
             "utilization_status": "calculated_unvalidated", "radix16_speedup": 2.18}])
        cells = self.comparisons("table7")
        self.assertEqual(cells[4, 1]["status"], "timing_only_unvalidated")
        self.assertEqual(cells[4, 1]["current"], .003512)
        self.assertEqual(cells[4, 2]["paper"], 75)
        self.assertEqual(cells[4, 2]["current"], 75.2)
        self.assertEqual(cells[4, 3]["paper"], 2.18)
        self.assertIsNone(cells[4, 3]["current"])

    def test_table7_selected_source_rejects_mismatched_mapping(self):
        mutations = (("source_variant", "register_branch"), ("source_variant", "historical_652808b"),
                     ("selection_status", "unchanged_candidate"), ("selection_basis", ""))
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                baseline, selected, metadata = self.selected_table7()
                metadata["paper_row_sources"]["register_branch"][key] = value
                self.write([baseline, selected], metadata)
                with self.assertRaisesRegex(ValueError, "Table 7"):
                    tables.build("table7", self.directory)
        baseline, selected, metadata = self.selected_table7()
        for mapping in ([], {"unknown_role": {}}, {"register_branch": "historical_e4aa1b0"}):
            with self.subTest(mapping=mapping):
                metadata["paper_row_sources"] = mapping
                self.write([baseline, selected], metadata)
                with self.assertRaisesRegex(ValueError, "Table 7"):
                    tables.build("table7", self.directory)

    def test_table7_selected_source_rejects_old_branch_fallback_and_false_correctness(self):
        for key, value in (("source_commit", "51e346c0fea2767c79075c2798fa2598876415a6"),
                           ("source_variant", "register_branch"), ("implementation_kind", "recovered"),
                           ("operation_coverage", "complete_fft_candidate"), ("paper_identity", "confirmed"),
                           ("accuracy_status", "passed"), ("timing_status", "collected")):
            with self.subTest(key=key, value=value):
                baseline, selected, metadata = self.selected_table7()
                selected[key] = value
                self.write([baseline, selected], metadata)
                with self.assertRaisesRegex(ValueError, "Table 7"):
                    tables.build("table7", self.directory)
        baseline, selected, metadata = self.selected_table7()
        metadata["full_paper_comparison"] = True
        self.write([baseline, selected], metadata)
        with self.assertRaisesRegex(ValueError, "Table 7"):
            tables.build("table7", self.directory)

    def test_table7_missing_selected_source_stays_blank(self):
        baseline, selected, metadata = self.selected_table7()
        selected.update(timing_status="missing", accuracy_status="not_collected",
                        utilization_status="missing", median_time_ms="", utilization_pct="",
                        speedup_vs_estimated_radix16="")
        self.write([baseline, selected,
                    {"variant": "historical_e4aa1b0", "median_time_ms": .0079,
                     "timing_status": "collected_unvalidated", "accuracy_status": "unchecked"}], metadata)
        row = tables.build("table7", self.directory)["panels"][0]["rows"][3]
        self.assertEqual(row, ["+ reducing communication (e4aa1b0)", "—", "—", "—"])

    def test_table7_selected_partial_ratio_requires_matching_estimate_and_unchecked_utilization(self):
        for key, value in (("speedup_vs_estimated_radix16", 99), ("speedup_vs_estimated_radix16", "nan"),
                           ("speedup_vs_estimated_radix16", -1), ("utilization_status", "calculated")):
            with self.subTest(key=key, value=value):
                baseline, selected, metadata = self.selected_table7()
                selected[key] = value
                self.write([baseline, selected], metadata)
                row = tables.build("table7", self.directory)["panels"][0]["rows"][3]
                self.assertEqual(row[-1], "—")
        baseline, selected, metadata = self.selected_table7()
        baseline.update(timing_status="collected", accuracy_status="passed", median_time_ms=.008161125)
        selected["radix16_speedup"] = .008161125 / .00788889598846
        self.write([baseline, selected], metadata)
        self.assertEqual(tables.build("table7", self.directory)["panels"][0]["rows"][3][-1], "—")

    def default_table7_assembly(self):
        """Assemble retained native timings with a synthetic Table 6 baseline."""
        specification = importlib.util.spec_from_file_location(
            "table7_default_presentation_assembly", ROOT / "artifact/table7/assemble.py")
        assembly = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(assembly)
        source = self.directory / "table6"
        source.mkdir()
        records = {
            "table.csv": [{"gpu": "A100", "N": 256, "batch": 65536,
                "library": "tcFFT", "variant": "layer_256_0_A100",
                "selected_candidate": "layer_256_0_A100", "status": "collected",
                "median_time_ms": .030}],
            "data.csv": [{"gpu": "A100", "N": 256, "batch": 65536,
                "library": "tcFFT", "variant": "layer_256_0_A100",
                "candidate": "layer_256_0_A100", "precision": "fp16",
                "input": "complex", "passed": True, "time_ms": .030}],
        }
        for name, rows in records.items():
            with (source / name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        (source / "metadata.json").write_text(json.dumps({
            "evaluation": "table6", "profile": "paper",
            "assembly_schema": "table6-validated-parts-v1",
            "outputs": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                        for name in records}}))
        output = self.directory / "assembled"
        metadata = assembly.assemble(
            [ROOT / "artifact/table7/reference/a100-20261107"], output, "paper", table6=source)
        with (output / "table.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(set(metadata["collected"]["unchecked"]), {
            "register_branch", "register_uniform", "historical_e4aa1b0", "historical_652808b"})
        self.assertEqual(metadata["collected"]["validated"], [])
        return rows, metadata

    def test_table7_default_assembly_keeps_both_qualified_timing_ratios(self):
        rows, metadata = self.default_table7_assembly()
        self.write(rows, metadata)
        layout = tables.build("table7", self.directory)
        displayed = layout["panels"][0]["rows"]
        baseline = next(row for row in rows if row["variant"] == "radix16")
        for index, variant in ((3, "register_branch"), (4, "register_uniform")):
            measured = next(row for row in rows if row["variant"] == variant)
            expected = float(baseline["estimated_time_ms"]) / float(measured["median_time_ms"])
            self.assertEqual(displayed[index][-1], f"{expected:.2f}")
            self.assertTrue(displayed[index][0].endswith("†"))
        notes = " ".join(layout["notes"])
        self.assertIn("correctness was not checked", notes)
        self.assertIn("do not represent correctness-validated complete-FFT speedups", notes)
        self.assertEqual(displayed[2][1:], ["—", "—", "—"])

    def test_table7_default_uniform_ratio_requires_known_source_and_consistent_evidence(self):
        rows, metadata = self.default_table7_assembly()
        for field, value in (
                ("source_variant", ""), ("source_commit", "unknown"),
                ("implementation_kind", "historical_partial_kernel"),
                ("operation_coverage", "partial_fft_operand_setup_omitted"),
                ("timing_status", "missing"), ("accuracy_status", "passed"),
                ("utilization_status", "calculated"), ("median_time_ms", "nan"),
                ("median_time_ms", "inf"), ("median_time_ms", 0),
                ("speedup_vs_estimated_radix16", 99), ("speedup_vs_estimated_radix16", "nan"),
                ("speedup_vs_estimated_radix16", "inf"), ("speedup_vs_estimated_radix16", -1)):
            with self.subTest(field=field, value=value):
                modified = [dict(row, **{field: value}) if row["variant"] == "register_uniform"
                            else dict(row) for row in rows]
                self.write(modified, metadata)
                self.assertEqual(tables.build("table7", self.directory)["panels"][0]["rows"][4][-1], "—")

    def test_table7_default_uniform_ratio_does_not_relax_measured_baseline_validation(self):
        rows, metadata = self.default_table7_assembly()
        for status, accuracy in (("collected", "passed"), ("collected_unvalidated", "unchecked")):
            with self.subTest(status=status, accuracy=accuracy):
                modified = [dict(row) for row in rows]
                baseline = next(row for row in modified if row["variant"] == "radix16")
                baseline.update(timing_status=status, accuracy_status=accuracy,
                                median_time_ms=baseline["estimated_time_ms"], utilization_status="calculated")
                uniform = next(row for row in modified if row["variant"] == "register_uniform")
                uniform["radix16_speedup"] = uniform["speedup_vs_estimated_radix16"]
                self.write(modified, metadata)
                self.assertEqual(tables.build("table7", self.directory)["panels"][0]["rows"][4][-1], "—")

    def test_table7_remapped_source_requires_selection_metadata(self):
        baseline, selected, _ = self.selected_table7()
        self.write([baseline, selected], {})
        with self.assertRaisesRegex(ValueError, "matching paper-row source metadata"):
            tables.build("table7", self.directory)

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
