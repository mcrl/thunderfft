"""CPU checks of provenance, coverage gates, and unit conversion (synthetic input)."""

import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("figure2_plot", HERE / "plot.py")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


class FigureSourceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic-table3-run"
        self.source.mkdir()
        self.status = {"evaluation": "table3", "profile": "smoke", "status": "completed", "exit_code": 0}
        self.rows = [{"gpu": "A100", "N": size, "batch": 1024, "placement": "register",
                      "library": "ThunderFFT", "precision": precision, "trial": trial,
                      "time_us": 1000.0 + 100 * trial, "e2e_out_of_place_us": 2000,
                      "rel_l2": 0.001, "passed": "true", "inside_repeats": 256}
                     for size in PLOT.SIZES for precision in PLOT.PRECISIONS for trial in (0, 1)]
        self.write_source()

    def write_source(self):
        (self.source / "status.json").write_text(json.dumps(self.status))
        with (self.source / "data.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def test_plot_preserves_trial_data_and_converts_units(self):
        output = self.root / "plot"
        result = subprocess.run([sys.executable, str(HERE / "plot.py"), "--source-dir",
                                 str(self.source), "--output-dir", str(output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["source"]["source_sha256"], PLOT.sha256(self.source / "data.csv"))
        self.assertEqual(metadata["source"]["source_run"], self.source.name)
        with (output / "data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 28)
        self.assertEqual(float(rows[0]["time_ms"]), 1.0)
        with (output / "summary.csv").open(newline="") as stream:
            summaries = list(csv.DictReader(stream))
        self.assertEqual(len(summaries), 14)
        self.assertEqual(float(summaries[0]["median_ms"]), 1.05)
        self.assertTrue((output / "figure.pdf").read_bytes().startswith(b"%PDF"))

    def test_invalid_evidence_is_rejected(self):
        alterations = {
            "missing configuration": lambda rows: [row for row in rows if not (row["N"] == 1024 and row["precision"] == "TF32")],
            "duplicate trial": lambda rows: rows + [dict(rows[0])],
            "failed correctness": lambda rows: [{**row, "passed": "false"} if i == 0 else row for i, row in enumerate(rows)],
            "nonfinite timing": lambda rows: [{**row, "time_us": "nan"} if i == 0 else row for i, row in enumerate(rows)],
            "large numerical error": lambda rows: [{**row, "rel_l2": 0.28} if i == 0 else row for i, row in enumerate(rows)],
            "inconsistent trials": lambda rows: rows[1:],
            "mixed batches": lambda rows: [{**row, "batch": 16} if i == 0 else row for i, row in enumerate(rows)],
        }
        original = self.rows
        for label, alter in alterations.items():
            with self.subTest(label=label):
                self.rows = alter(original)
                self.write_source()
                with self.assertRaises(ValueError):
                    PLOT.read_source(self.source, "smoke")

    def test_completed_source_and_matching_profile_are_required(self):
        self.status["status"] = "failed"
        self.write_source()
        with self.assertRaises(ValueError):
            PLOT.read_source(self.source, "smoke")
        self.status["status"] = "completed"
        self.write_source()
        with self.assertRaises(ValueError):
            PLOT.read_source(self.source, "paper")

    def test_non_a100_source_is_pending(self):
        self.rows = [{**row, "gpu": "H100"} for row in self.rows]
        self.write_source()
        with self.assertRaises(PLOT.Pending):
            PLOT.read_source(self.source, "smoke")

    def test_lowercase_precision_labels_from_table3_are_accepted(self):
        self.rows = [{**row, "precision": row["precision"].lower()} for row in self.rows]
        self.write_source()
        _, rows, groups, _ = PLOT.read_source(self.source, "smoke")
        self.assertEqual(len(groups), 14)
        self.assertEqual(rows[0]["precision"], "tf32")

    def test_paper_repeats_are_checked(self):
        self.status["profile"] = "paper"
        self.rows = [{**row, "batch": 65536, "inside_repeats": 256} for row in self.rows]
        self.write_source()
        with self.assertRaises(ValueError):
            PLOT.read_source(self.source, "paper")


if __name__ == "__main__":
    unittest.main()
