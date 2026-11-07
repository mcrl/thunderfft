"""CPU checks for declared projections, source contracts, and preserved measurements."""
import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("project_baseline", HERE / "project_baseline.py")
project = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(project)


def csv_file(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class ProjectionTests(unittest.TestCase):
    def test_documented_values_and_formula_are_not_fixed_seventy_percent(self):
        sources, records = project.documented_sources()
        values = {row["library"]: project.project(row) for row in sources}
        self.assertEqual(len(records), 1)
        for library, source_us, target_us, utilization in (
                ("tcFFT", 21.8, 8.175, 63.146425782169835),
                ("FlashFFTConv", 18.1, 6.7875, 76.05481116247345)):
            row = values[library]
            self.assertAlmostEqual(row["source_time_us"], source_us)
            self.assertAlmostEqual(row["projected_time_us"], target_us)
            self.assertAlmostEqual(row["source_utilization_pct"], utilization)
            self.assertEqual(row["source_evidence"], "documented_rounded_timing")
            self.assertEqual(row["timing_status"], "estimated")
            self.assertFalse(row["full_paper_comparison"])
            self.assertIn("not an implemented", row["target_work_model"])

    def test_rejects_invalid_timings_and_unsupported_kernels(self):
        for timing in (0, -1, math.inf, math.nan):
            with self.subTest(timing=timing), self.assertRaises(ValueError):
                project.project({"library": "tcFFT", "time_us": timing, "evidence": "test"})
        with self.assertRaises(ValueError):
            project.project({"library": "cuFFTDx", "time_us": 23, "evidence": "test"})

    def test_table6_requires_matching_fp16_raw_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            table = {"gpu": "A100", "N": "256", "batch": "65536", "library": "FlashFFTConv",
                     "variant": "conv_kernel", "selected_candidate": "conv_kernel", "median_time_ms": "0.0181"}
            raw = {"gpu": "A100", "N": "256", "batch": "65536", "library": "FlashFFTConv",
                   "variant": "conv_kernel", "candidate": "conv_kernel", "precision": "fp16",
                   "input": "complex", "time_ms": "0.0181001"}
            path = root / "table.csv"
            csv_file(path, [table])
            with self.assertRaisesRegex(ValueError, "adjacent data.csv"):
                project.table6_sources(path)
            csv_file(root / "data.csv", [raw])
            sources, records = project.table6_sources(path)
            self.assertAlmostEqual(sources[0]["time_us"], 18.1)
            self.assertEqual(len(records), 2)
            for key, invalid in (("precision", "tf32"), ("N", "64"), ("batch", "4096"),
                                 ("input", "real"), ("candidate", "other")):
                csv_file(root / "data.csv", [dict(raw, **{key: invalid})])
                with self.subTest(key=key), self.assertRaises(ValueError):
                    project.table6_sources(path)
            csv_file(root / "data.csv", [dict(raw, time_ms="0.02")])
            with self.assertRaisesRegex(ValueError, "disagrees"):
                project.table6_sources(path)

    def test_comparison_preserves_source_rows_and_does_not_overwrite_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = {"variant": "historical_652808b", "N": "64", "batch": "65536", "precision": "tf32",
                        "median_time_ms": "0.00348160004616", "timing_status": "collected_unvalidated",
                        "accuracy_status": "unchecked", "source_commit": "652808b", "paper_identity": "unconfirmed",
                        "known_source_issues": "operand setup omitted", "utilization_pct": "74.13574562"}
            csv_file(root / "kernel_comparison.csv", [original])
            (root / "metadata.json").write_text(json.dumps({"evaluation": "table7", "gpu": {"name": "NVIDIA A100 80GB PCIe"}}))
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            sources, records = project.documented_sources()
            project.write_outputs(root / "projection", sources, records, root / "kernel_comparison.csv")
            _, rows = project.read_csv(root / "projection/comparison.csv")
            measured = next(row for row in rows if row["variant"] == original["variant"])
            for key, value in original.items():
                self.assertEqual(measured[key], value)
            self.assertAlmostEqual(float(measured["speedup_vs_projected_flashfftconv"]), 6.7875 / 3.48160004616)
            for row in rows[:2]:
                self.assertEqual(row["timing_status"], "estimated")
                self.assertEqual(row["median_time_ms"], "")
            for name, data in before.items():
                self.assertEqual((root / name).read_bytes(), data)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                project.write_outputs(root / "projection", sources, records, root / "kernel_comparison.csv")
            self.assertFalse((root / "table.csv").exists())

    def test_comparison_rejects_wrong_gpu_or_workload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"variant": "register_branch", "N": "64", "batch": "65536", "precision": "tf32",
                   "median_time_ms": "0.012", "timing_status": "collected_unvalidated", "accuracy_status": "unchecked"}
            path = root / "kernel_comparison.csv"
            estimates = [project.project(item) for item in project.documented_sources()[0]]
            for gpu, batch in (("NVIDIA H100", "65536"), ("A100", "1024")):
                csv_file(path, [dict(row, batch=batch)])
                (root / "metadata.json").write_text(json.dumps({"evaluation": "table7", "gpu": {"name": gpu}}))
                with self.subTest(gpu=gpu, batch=batch), self.assertRaises(ValueError):
                    project.comparison_rows(path, estimates)


if __name__ == "__main__":
    unittest.main()
