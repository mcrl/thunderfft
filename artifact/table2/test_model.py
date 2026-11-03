"""CPU checks of paper-model results, invalid inputs, and calculated provenance."""

import copy
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("table2_model", HERE / "model.py")
MODEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODEL)


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "assumptions.json").read_text())

    def test_paper_values_and_mma_bandwidth_units(self):
        rows = MODEL.calculate(self.config)
        self.assertEqual(len(rows), 4)
        ideal, radix4, radix8, radix16 = rows
        self.assertEqual(ideal["relative_time"], 1.0)
        self.assertEqual(ideal["arithmetic_coefficient"], 5.0)
        expected = ((radix4, 16, 512, 1, 512, 0.25, 3.2),
                    (radix8, 64 / 3, 1024, 4, 256, 0.5, 32 / 15),
                    (radix16, 32, 2048, 16, 128, 1.0, 1.6))
        for row, arithmetic, traffic, cycles, bandwidth, utilization, relative in expected:
            with self.subTest(design=row["design"]):
                self.assertAlmostEqual(row["arithmetic_coefficient"], arithmetic)
                self.assertAlmostEqual(row["shared_bytes_per_mma"], traffic)
                self.assertAlmostEqual(row["mma_issue_cycles"], cycles)
                self.assertAlmostEqual(row["required_shared_bytes_per_cycle"], bandwidth)
                self.assertAlmostEqual(row["utilization_ceiling"], utilization)
                self.assertAlmostEqual(row["relative_time"], relative)
                self.assertEqual(row["origin"], "calculated")

    def test_half_bandwidth_doubles_bandwidth_limited_times(self):
        config = copy.deepcopy(self.config)
        config["shared_bytes_per_sm_cycle"] = 64
        rows = MODEL.calculate(config)
        self.assertEqual(rows[0]["relative_time"], 1)
        self.assertEqual([row["utilization_ceiling"] for row in rows[1:]], [0.125, 0.25, 0.5])
        for row, expected in zip(rows[1:], (6.4, 64 / 15, 3.2)):
            self.assertAlmostEqual(row["relative_time"], expected)

    def test_unlimited_bandwidth_caps_utilization_at_one(self):
        config = copy.deepcopy(self.config)
        config["shared_bytes_per_sm_cycle"] = 1024
        rows = MODEL.calculate(config)
        self.assertEqual([row["utilization_ceiling"] for row in rows], [1, 1, 1, 1])
        for row, expected in zip(rows[1:], (0.8, 16 / 15, 1.6)):
            self.assertAlmostEqual(row["relative_time"], expected)

    def test_invalid_numerical_inputs_and_undefined_radices_are_rejected(self):
        keys = ("cuda_peak_tflops", "tensor_peak_tflops", "shared_bytes_per_sm_cycle",
                "tensor_ops_per_sm_cycle", "mma_m", "component_bytes")
        for key in keys:
            for value in (0, -1, math.inf, -math.inf, math.nan):
                with self.subTest(key=key, value=value):
                    config = copy.deepcopy(self.config)
                    config[key] = value
                    with self.assertRaises(ValueError):
                        MODEL.calculate(config)
        for radix in (0, 2, 3, 32):
            with self.subTest(radix=radix):
                config = copy.deepcopy(self.config)
                config["radices"] = [radix]
                with self.assertRaises(ValueError):
                    MODEL.calculate(config)

    def test_generated_artifact_excludes_measured_cells_and_keeps_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            result = subprocess.run([sys.executable, str(HERE / "model.py"), "--output", str(output)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            with (output / "data.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["origin"] for row in rows}, {"calculated"})
            self.assertFalse(any("cuFFTDx" in row["design"] for row in rows))
            self.assertEqual((output / "data.csv").read_bytes(), (output / "table.csv").read_bytes())
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertFalse(metadata["full_paper_comparison"])
            self.assertIn("measured cuFFTDx row", metadata["excluded"])
            self.assertEqual(metadata["assumptions"], self.config)
            self.assertEqual(metadata["assumptions_sha256"],
                             hashlib.sha256((HERE / "assumptions.json").read_bytes()).hexdigest())
            self.assertEqual(metadata["generator_sha256"],
                             hashlib.sha256((HERE / "model.py").read_bytes()).hexdigest())
            self.assertEqual(len(metadata["assumptions"]["paper_pdf_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
