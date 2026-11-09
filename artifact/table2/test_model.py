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
FIXTURE_SPEC = importlib.util.spec_from_file_location("table2_model_table3_fixture", HERE.parent / "table3/test_assemble.py")
FIXTURE = importlib.util.module_from_spec(FIXTURE_SPEC)
FIXTURE_SPEC.loader.exec_module(FIXTURE)


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


def write_table3(directory, rows, profile="paper", selected=None):
    """Minimal completed Table 3 result directory."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "status.json").write_text(json.dumps(
        {"evaluation": "table3", "status": "completed", "exit_code": 0, "profile": profile}))
    fields = ["gpu", "N", "batch", "placement", "library", "precision", "trial", "time_us",
              "rel_l2", "passed", "inside_repeats", "candidate"]
    with (directory / "data.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if selected is not None:
        with (directory / "cufftdx_candidates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["gpu", "N", "precision", "placement", "candidate", "selected"])
            writer.writeheader()
            writer.writerow({"gpu": "A100", "N": 64, "precision": "fp16", "placement": "register",
                             "candidate": selected, "selected": "true"})


def dx_rows(candidate, times, **overrides):
    return [{"gpu": "A100", "N": 64, "batch": 65536, "placement": "register", "library": "cuFFTDx",
             "precision": "fp16", "trial": trial, "time_us": value, "rel_l2": 1e-3, "passed": "true",
             "inside_repeats": 1000, "candidate": candidate, **overrides}
            for trial, value in enumerate(times)]


class MeasuredRowTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "assumptions.json").read_text())
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_model(self, source, profile="paper"):
        output = self.root / "out"
        result = subprocess.run([sys.executable, str(HERE / "model.py"), "--output", str(output),
                                 "--profile", profile, "--source-dir", str(source)],
                                text=True, capture_output=True)
        return result, output

    def test_paper_value_gives_paper_row(self):
        row = MODEL.measured_row(self.config, 2.89, 65536, "x")
        self.assertAlmostEqual(row["ideal_time_us"], 5 * 64 * 6 * 65536 / 78e12 * 1e6)
        self.assertEqual(round(row["relative_time"], 2), 1.79)
        self.assertEqual(round(100 * row["utilization_ceiling"], 1), 55.8)
        self.assertEqual(row["origin"], "measured")

    def test_fastest_candidate_is_used_and_inserted_after_ideal(self):
        source = self.root / "t3"
        rows = (dx_rows("block_default", [3.5, 3.4, 3.6]) + dx_rows("thread_t32", [2.9, 2.88, 2.95])
                + dx_rows("block_f1_e8", [2.5, 2.5, 2.5], N=128))
        write_table3(source, rows, selected="thread_t32")
        result, output = self.run_model(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output / "table.csv").open(newline="") as stream:
            table = list(csv.DictReader(stream))
        self.assertEqual([row["design"] for row in table],
                         ["Ideal FFT", "cuFFTDx", "Radix-4", "Radix-8", "Radix-16"])
        self.assertEqual(table[1]["candidate"], "thread_t32")
        self.assertAlmostEqual(float(table[1]["measured_time_us"]), 2.9)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertTrue(metadata["full_paper_comparison"])
        self.assertEqual(metadata["excluded"], [])
        self.assertEqual(metadata["measured_source"]["candidates_compared"], 2)
        self.assertEqual(metadata["measured_source"]["cufftdx_mode"], "all")
        self.assertIn("smallest median", metadata["measured_source"]["cufftdx_selection_rule"])

    def test_preset_source_preserves_mode_hash_and_current_measurement(self):
        source = FIXTURE.make_preset_part(self.root / "preset")
        FIXTURE.MEASURE.atomic_json(source / "status.json", {
            "evaluation": "table3", "profile": "smoke", "status": "completed", "exit_code": 0})
        result, output = self.run_model(source, "smoke")
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        measured = metadata["measured_source"]
        self.assertEqual(measured["cufftdx_mode"], "preset")
        self.assertEqual(measured["cufftdx_preset_sha256"],
                         MODEL.MEASURE.preset_descriptor("A100")["manifest_sha256"])
        self.assertEqual(measured["candidates_compared"], 1)
        self.assertEqual(measured["trial_time_us"], [2.0, 4.0])
        self.assertIn("fixed preset candidate remeasured", (output / "table.md").read_text())
        self.assertEqual(metadata["assumptions"], self.config)

    def test_preset_rejects_a_different_measured_candidate(self):
        source = FIXTURE.make_preset_part(self.root / "preset")
        FIXTURE.MEASURE.atomic_json(source / "status.json", {
            "evaluation": "table3", "profile": "smoke", "status": "completed", "exit_code": 0})
        path = source / "data.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row["candidate"] = "unconfigured_candidate"
        FIXTURE.MEASURE.write_csv(path, rows, fields)
        self.assertEqual(self.run_model(source, "smoke")[0].returncode, 65)

    def test_source_without_cufftdx_keeps_calculated_rows(self):
        source = self.root / "t3"
        write_table3(source, [dict(row, library="ThunderFFT", candidate="") for row in dx_rows("", [2.2])])
        result, output = self.run_model(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["excluded"], ["measured cuFFTDx row"])
        self.assertIn("no cuFFTDx", metadata["excluded_reason"])

    def test_invalid_sources_are_rejected(self):
        cases = {"failed_gate": dx_rows("a", [2.9], passed="false"),
                 "relL2": dx_rows("a", [2.9], rel_l2=0.01),
                 "smoke_batch": dx_rows("a", [2.9], batch=1024)}
        for name, rows in cases.items():
            with self.subTest(name=name):
                source = self.root / name
                write_table3(source, rows)
                result, _ = self.run_model(source)
                self.assertEqual(result.returncode, 65, result.stdout)
        source = self.root / "profile"
        write_table3(source, dx_rows("a", [2.9]), profile="smoke")
        self.assertEqual(self.run_model(source)[0].returncode, 65)
        source = self.root / "selection"
        write_table3(source, dx_rows("a", [2.9]) + dx_rows("b", [3.0]), selected="b")
        self.assertEqual(self.run_model(source)[0].returncode, 65)


if __name__ == "__main__":
    unittest.main()
