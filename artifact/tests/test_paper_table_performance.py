"""Formatting must preserve measurement provenance and partial-result gaps."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import paper_table_performance as paper


class PerformanceTableTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def write(self, name, rows):
        fields = list(dict.fromkeys(key for row in rows for key in row))
        with (self.directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def metadata(self, **values):
        (self.directory / "metadata.json").write_text(json.dumps(values))

    def table3_rows(self, sizes=paper.SIZES):
        rows = []
        for n in sizes:
            for library, precision, time in (("ThunderFFT", "fp16", 2), ("cuFFTDx", "fp16", 6),
                                              ("ThunderFFT", "tf32", 5), ("cuFFTDx", "fp32", 10)):
                rows.append(dict(gpu="A100", N=n, library=library, precision=precision,
                                 placement="register", median_time_us=time, passed="true",
                                 status="collected", paper_target_us=999))
        return rows

    def comparisons(self, panel):
        values = {(cell["row"], cell["col"]): cell for cell in panel["comparisons"]}
        self.assertEqual(len(values), len(panel["comparisons"]))
        return values

    def test_table2_paper_order_groups_and_model_coefficient(self):
        self.write("table.csv", [dict(design="Ideal FFT", peak_tflops=78,
                                     arithmetic_coefficient=5, utilization_ceiling=1, relative_time=1),
                                 dict(design="Radix-4", peak_tflops=312, mma_shape="m16n8k8",
                                      arithmetic_coefficient=16, utilization_ceiling=.25, relative_time=3.2),
                                 dict(design="Radix-8", peak_tflops=312, mma_shape="m16n16k16",
                                      arithmetic_coefficient=64/3, utilization_ceiling=.5, relative_time=64/30),
                                 dict(design="Radix-16", peak_tflops=312, mma_shape="m16n32k32",
                                      arithmetic_coefficient=32, utilization_ceiling=1, relative_time=1.6)])
        panel = paper.build("table2", self.directory)["panels"][0]
        self.assertEqual([row[2] for row in panel["rows"]],
                         ["Design", "Ideal FFT", "cuFFTDx", "Radix-4", "Radix-8", "Radix-16"])
        self.assertEqual(panel["rows"][2][4:], ["5.0", "—", "—"])
        self.assertEqual(panel["rows"][4][4:], ["21.3", "50%", "2.13"])
        self.assertIn(dict(row=1, col=0, rowspan=2, colspan=1), panel["spans"])
        self.assertIn(dict(row=3, col=1, rowspan=3, colspan=1), panel["spans"])

    def test_table2_marks_measured_values_and_preserves_preset_scope(self):
        self.write("table.csv", [dict(design="Ideal FFT", arithmetic_coefficient=5),
                                 dict(design="cuFFTDx", arithmetic_coefficient="",
                                      utilization_ceiling=.61234, relative_time=1.6334)])
        self.metadata(measured_source={"cufftdx_mode": "preset"})
        layout = paper.build("table2", self.directory)
        self.assertEqual(layout["panels"][0]["rows"][2][4:], ["5.0", "61.2%†", "1.63†"])
        self.assertTrue(any("saved presets" in note for note in layout["notes"]))

    def test_table2_compares_printed_numbers_to_unrounded_model_and_measurement(self):
        self.write("table.csv", [dict(design="Ideal FFT", peak_tflops=78, arithmetic_coefficient=5),
                                 dict(design="cuFFTDx", utilization_ceiling=.61234, relative_time=1.6334),
                                 dict(design="Radix-8", arithmetic_coefficient=64/3,
                                      utilization_ceiling=.5, relative_time=64/30)])
        values = self.comparisons(paper.build("table2", self.directory)["panels"][0])
        self.assertEqual(len(values), 17)
        self.assertEqual((values[4, 4]["paper"], values[4, 4]["current"], values[4, 4]["status"]),
                         (21.3, 64/3, "model"))
        self.assertEqual(values[4, 6]["paper"], 2.13)
        self.assertEqual(values[4, 6]["current"], 64/30)
        self.assertAlmostEqual(values[2, 5]["current"], 61.234)
        self.assertEqual(values[2, 5]["paper"], 55.8)
        self.assertEqual(values[2, 5]["status"], "measured")
        self.assertEqual(values[2, 6]["current"], 1.6334)
        self.assertEqual(values[5, 6]["status"], "missing")
        self.assertTrue(all(col in (1, 4, 5, 6) for row, col in values))

    def test_table3_header_and_matched_precision_mapping(self):
        self.write("table.csv", self.table3_rows((64,)))
        layout = paper.build("table3", self.directory)
        register, shared = layout["panels"]
        self.assertEqual(len(register["rows"]), 11)
        self.assertTrue(all(len(row) == 19 for row in register["rows"]))
        self.assertEqual(register["rows"][0][1], "FP16")
        self.assertEqual(register["rows"][0][10], "TF32 / FP32")
        self.assertEqual(register["rows"][2], [""] + list(paper.GPUS) * 6)
        self.assertEqual([register["rows"][3][col] for col in (1, 4, 7, 10, 13, 16)],
                         ["6.00", "2.00", "3.00", "10.00", "5.00", "2.00"])
        self.assertEqual(register["rows"][3][2], "—")
        self.assertEqual(shared["rows"][3][1:], ["—"] * 18)
        self.assertIn(dict(row=0, col=10, rowspan=1, colspan=9), register["spans"])

    def test_table3_reference_precision_pairing_and_printed_speedups(self):
        self.write("table.csv", self.table3_rows((64,)))
        panel = paper.build("table3", self.directory)["panels"][0]
        values = self.comparisons(panel)
        self.assertEqual(len(values), 7 * 18 + 6)
        self.assertEqual([values[3, col]["paper"] for col in (1, 4, 7, 10, 13, 16)],
                         [2.89, 2.25, 1.28, 8.14, 3.55, 2.29])
        self.assertNotEqual(values[3, 7]["paper"], 2.89 / 2.25)
        self.assertEqual(values[10, 7]["paper"], 1.31)
        self.assertEqual(values[10, 16]["paper"], 2.19)
        self.assertEqual(values[10, 7]["current"], None)
        self.assertEqual(values[10, 7]["status"], "missing")
        self.assertTrue(all(row >= 3 and col > 0 for row, col in values))

    def test_table3_comparison_operands_preserve_raw_measurement_precision(self):
        rows = self.table3_rows((64,))
        rows[0]["median_time_us"] = 2.0001
        rows[1]["median_time_us"] = 6.0049
        self.write("table.csv", rows)
        values = self.comparisons(paper.build("table3", self.directory)["panels"][0])
        self.assertEqual(values[3, 1]["current_text"], "6.00")
        self.assertEqual(values[3, 1]["current"], 6.0049)
        self.assertEqual(values[3, 4]["current"], 2.0001)
        self.assertEqual(values[3, 7]["current"], 6.0049 / 2.0001)
        self.assertEqual(values[3, 7]["status"], "derived")

    def test_table3_missing_reference_does_not_use_arbitrary_csv_paper_target(self):
        self.write("table.csv", self.table3_rows((64,)))
        targets = paper._csv(paper.TABLE3_TIMING_FILE)
        self.write("targets.csv", [row for row in targets if not (
            row["gpu"] == "A100" and row["N"] == "64" and row["precision"] == "fp16"
            and row["placement"] == "register" and row["implementation"] == "xfft")])
        with mock.patch.object(paper, "TABLE3_TIMING_FILE", self.directory / "targets.csv"):
            values = self.comparisons(paper.build("table3", self.directory)["panels"][0])
        self.assertEqual(values[3, 4]["current"], 2)
        self.assertIsNone(values[3, 4]["paper"])
        self.assertEqual(values[3, 4]["paper_text"], "—")
        self.assertEqual(values[3, 4]["status"], "missing_reference")

    def test_table3_missing_measurement_preserves_reference_without_substitution(self):
        rows = self.table3_rows((64,))
        rows[0]["status"] = "pending"
        self.write("table.csv", rows)
        values = self.comparisons(paper.build("table3", self.directory)["panels"][0])
        self.assertEqual(values[3, 4]["paper"], 2.25)
        self.assertIsNone(values[3, 4]["current"])
        self.assertEqual(values[3, 4]["current_text"], "—")
        self.assertEqual(values[3, 4]["status"], "missing")
        self.assertIsNone(values[3, 7]["current"])

    def test_table3_geomean_requires_all_seven_sizes(self):
        self.write("table.csv", self.table3_rows())
        full = paper.build("table3", self.directory)["panels"][0]["rows"][-1]
        self.assertEqual(full[7:10], ["3.00", "—", "—"])
        self.assertEqual(full[16:19], ["2.00", "—", "—"])
        self.write("table.csv", self.table3_rows(paper.SIZES[:-1]))
        self.write("speedup.csv", [dict(gpu="A100", N="geomean(6 sizes)", thunderfft_precision="fp16",
                                       cufftdx_precision="fp16", placement="register", speedup=3)])
        partial = paper.build("table3", self.directory)["panels"][0]["rows"][-1]
        self.assertEqual(partial[7], "—")
        self.assertEqual(partial[16], "—")

    def test_table3_never_uses_pending_failed_or_nonfinite_times(self):
        rows = self.table3_rows((64,))
        rows[0]["status"] = "pending"
        rows[1]["passed"] = "false"
        rows[2]["median_time_us"] = "nan"
        rows[3]["median_time_us"] = "inf"
        self.write("table.csv", rows)
        row = paper.build("table3", self.directory)["panels"][0]["rows"][3]
        self.assertEqual(row[1:], ["—"] * 18)

    def test_table3_checks_recorded_speedups_against_actual_times(self):
        self.write("table.csv", self.table3_rows((64,)))
        self.write("speedup.csv", [dict(gpu="A100", N=64, thunderfft_precision="fp16",
                                       cufftdx_precision="fp16", placement="register", speedup=3)])
        paper.build("table3", self.directory)
        self.write("speedup.csv", [dict(gpu="A100", N=64, thunderfft_precision="fp16",
                                       cufftdx_precision="fp16", placement="register", speedup=2)])
        with self.assertRaisesRegex(ValueError, "speedup disagrees"):
            paper.build("table3", self.directory)

    def test_table3_checks_full_geomean_and_notes_partial_preset(self):
        self.write("table.csv", self.table3_rows())
        self.write("speedup.csv", [dict(gpu="A100", N="geomean(7 sizes)", thunderfft_precision="fp16",
                                       cufftdx_precision="fp16", placement="register", speedup=3)])
        self.metadata(profile="smoke", status="partial", cufftdx_selection={"A100": {"mode": "preset"}})
        notes = " ".join(paper.build("table3", self.directory)["notes"])
        self.assertIn("saved presets", notes)
        self.assertIn("Smoke-profile", notes)
        self.assertIn("partial", notes)
        self.write("speedup.csv", [dict(gpu="A100", N="geomean(7 sizes)", thunderfft_precision="fp16",
                                       cufftdx_precision="fp16", placement="register", speedup=8)])
        with self.assertRaisesRegex(ValueError, "geometric mean disagrees"):
            paper.build("table3", self.directory)

    def test_table4_transposes_only_measured_columns(self):
        self.write("table.csv", [dict(gpu="B200", hardware_peak_tflops=1100, status="pending",
                                     ideal_mma_status="pending", paper_ideal_mma_tflops=275.8, paper_unit_tflops=202.34),
                                 dict(gpu="A100", hardware_peak_tflops=156, status="complete",
                                      ideal_mma_status="measured", ideal_mma_tflops=150.236, unit_tflops=100.128),
                                 dict(gpu="H100", hardware_peak_tflops=378, status="partial",
                                      ideal_mma_status="measured", ideal_mma_tflops=240.5)])
        rows = paper.build("table4", self.directory)["panels"][0]["rows"]
        self.assertEqual(rows[0], ["", "A100", "H100", "B200"])
        self.assertEqual(rows[1][1:], ["156.00", "378.00", "1,100.00"])
        self.assertEqual(rows[2][1:], ["150.24", "240.50", "—"])
        self.assertEqual(rows[3][1:], ["100.13", "—", "—"])

    def test_table4_compares_reference_measured_and_derived_without_filling_gaps(self):
        self.write("table.csv", [dict(gpu="A100", hardware_peak_tflops=156, status="complete",
                                      ideal_mma_status="measured", ideal_mma_tflops=150.236,
                                      unit_tflops=100.128),
                                 dict(gpu="B200", hardware_peak_tflops=1100, status="pending",
                                      ideal_mma_status="pending", ideal_mma_tflops=999,
                                      unit_tflops=888, paper_ideal_mma_tflops=777)])
        values = self.comparisons(paper.build("table4", self.directory)["panels"][0])
        self.assertEqual(len(values), 9)
        self.assertEqual(values[1, 1]["status"], "reference")
        self.assertEqual(values[2, 1]["status"], "measured")
        self.assertEqual(values[3, 1]["status"], "derived")
        self.assertEqual(values[2, 1]["paper"], 147.22)
        self.assertEqual(values[2, 1]["current"], 150.236)
        self.assertEqual(values[2, 3]["paper"], 275.8)
        self.assertIsNone(values[2, 3]["current"])
        self.assertIsNone(values[3, 3]["current"])

    def test_duplicate_rows_are_not_silently_replaced(self):
        rows = self.table3_rows((64,))
        self.write("table.csv", rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            paper.build("table3", self.directory)


if __name__ == "__main__":
    unittest.main()
