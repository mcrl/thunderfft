"""CPU-only integrity checks for the archived, unchecked timing evidence."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import unittest


HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class HistoricalEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((HERE / "manifest.json").read_text())
        cls.rows = {row["id"]: row for row in read_csv(HERE / "measurements.csv")}
        cls.measurements = {row["id"]: row for row in cls.manifest["measurements"]}
        cls.coverage = json.loads((HERE / "coverage.json").read_text())

    def test_portable_files_and_hashes(self):
        self.assertEqual(len(self.rows), 16)
        self.assertEqual(set(self.rows), set(self.measurements))
        records = [self.manifest["measurements_csv"], self.manifest["coverage"]]
        for item in self.measurements.values():
            records.extend([item["raw_csv"], item["native_log"]])
            for value in [item["binary_sha256"], item["driver_sha256"],
                          *item["source_sha256"].values()]:
                self.assertRegex(value, r"^[a-f0-9]{64}$")
        for record in records:
            with self.subTest(path=record["path"]):
                path = Path(record["path"])
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)
                self.assertEqual(digest(HERE / path), record["sha256"])
                text = (HERE / path).read_text()
                self.assertNotIn("/home/", text)
                self.assertNotIn("/root/", text)

    def test_raw_timing_and_resource_rows(self):
        method = self.manifest["methodology"]
        for ident, item in self.measurements.items():
            with self.subTest(candidate=ident):
                summary = self.rows[ident]
                raw_rows = read_csv(HERE / item["raw_csv"]["path"])
                self.assertEqual(len(raw_rows), 1)
                raw = raw_rows[0]
                self.assertEqual(raw["variant"], item["native_cli_variant"])
                self.assertEqual(int(raw["N"]), method["N"])
                self.assertEqual(int(raw["batch"]), method["batch"])
                self.assertEqual(raw["precision"], method["precision"])
                self.assertEqual(int(raw["inside_repeats"]), method["inside_repeats"])
                self.assertEqual(raw["trial"], "0")
                expected = (float(raw["t_2R_us"]) - float(raw["t_R_us"])) / int(raw["inside_repeats"])
                self.assertGreater(expected, 0)
                self.assertTrue(math.isfinite(expected))
                self.assertAlmostEqual(float(raw["compute_us"]), expected, delta=1e-7)
                for field in ["compute_us", "t_R_us", "t_2R_us"]:
                    self.assertEqual(summary[field], raw[field])
                log = (HERE / item["native_log"]["path"]).read_text()
                for field in ["registers", "local_bytes", "static_shared_bytes"]:
                    found = re.search(r"\b" + field + r"=(\d+)", log)
                    self.assertIsNotNone(found)
                    self.assertEqual(found.group(1), summary[field])
                for expected_token in ["N=64", "batch=65536", "R=1000", "trials=1",
                                       "warmups_per_repeat=1", "block=32", "grid=4096",
                                       "cuda_compiler=12.9.41", "driver=12090", "runtime=12090",
                                       "gpu_uuid=" + method["gpu_uuid"]]:
                    self.assertIn(expected_token, log)
                self.assertIn("gpu=" + method["gpu"], log)

    def test_evidence_cannot_claim_numerical_validation(self):
        self.assertFalse(self.manifest["ae_import_eligible"])
        self.assertFalse(self.manifest["collection_import"]["direct_import_supported"])
        self.assertIn("archive_format_only", self.manifest["collection_import"]["ae_import_eligible_false_scope"])
        self.assertFalse(self.manifest["full_paper_reproduction"])
        self.assertEqual(self.manifest["paper_identity"], "unconfirmed")
        method = self.manifest["methodology"]
        self.assertFalse(method["accuracy_checked"])
        self.assertFalse(method["output_guards_checked"])
        self.assertEqual(method["trials"], 1)
        self.assertEqual(method["warmups_per_repeat"], 1)
        self.assertFalse(method["fast_math"])
        for ident, item in self.measurements.items():
            with self.subTest(candidate=ident):
                self.assertFalse(item["ae_import_eligible"])
                self.assertEqual(item["ae_import_ineligibility_reason"],
                                 "archived_raw_format_is_not_a_current_collection_bundle")
                self.assertFalse(item["accuracy_checked"])
                self.assertFalse(item["output_guards_checked"])
                raw = read_csv(HERE / item["raw_csv"]["path"])[0]
                self.assertEqual(raw["passed"], "0")
                self.assertTrue(math.isnan(float(raw["relative_l2"])))
                self.assertTrue(math.isnan(float(raw["max_transform_relative_l2"])))
                log = (HERE / item["native_log"]["path"]).read_text()
                self.assertNotIn("validation_input=", log)
                self.assertRegex(log, r"correctness_scope=(?:unchecked|not_checked)\b")
                self.assertEqual(self.rows[ident]["accuracy_status"], "not_checked_by_user_request")

    def test_coverage_distinguishes_measured_and_equivalent_sources(self):
        commits = self.coverage["commits"]
        self.assertEqual(len(commits), 50)
        self.assertEqual(len({row["commit"] for row in commits}), 50)
        self.assertEqual(sum(len(row["variants"]) for row in commits), 87)
        independently_timed = []
        for commit in commits:
            for variant in commit["variants"]:
                measured = self.measurements[variant["measurement_id"]]
                self.assertEqual(measured["family"], "register")
                self.assertEqual(variant["normalized_source_closure_sha256"],
                                 measured["normalized_source_closure_sha256"])
                self.assertEqual(variant["kernel"], measured["kernel"])
                self.assertNotIn("compute_us", variant)
                direct = commit["commit"] == measured["commit"]
                expected = "independently_timed" if direct else "source_equivalent_not_independently_timed"
                self.assertEqual(variant["coverage"], expected)
                if direct:
                    independently_timed.append(variant["measurement_id"])
        self.assertEqual(len(independently_timed), 12)
        self.assertEqual(len(set(independently_timed)), 12)

    def test_source_selection_keeps_missing_rows_and_known_defects(self):
        selection = {row["row"]: row for row in self.manifest["paper_row_selection"]}
        for name in ["radix16", "radix4_compute"]:
            self.assertEqual(selection[name]["status"], "original_source_missing")
            self.assertIsNone(selection[name]["measurement_id"])
        for name, ident in [("register_branch", "51e346c-default"),
                            ("register_uniform", "5e37821-default")]:
            self.assertEqual(selection[name]["measurement_id"], ident)
            self.assertFalse(selection[name]["exact_paper_identity_confirmed"])
            self.assertEqual(self.rows[ident]["source_selection_role"], "candidate_" + name)
        self.assertTrue(self.measurements["shared_radix8_22f067e"]["known_source_issues"])
        self.assertTrue(self.measurements["e4aa1b0-default"]["known_source_issues"])
        self.assertTrue(self.measurements["bbeffb2-default"]["host_header_adapter"])
        method = self.manifest["methodology"]
        self.assertIn("float PI", method["register_host_adapter"])
        self.assertIn("double-PI", method["selected_source_protocol_difference"])
        self.assertIn("three warmups", method["selected_source_protocol_difference"])
        inputs = method["input_by_family"]
        register = inputs["register"]
        self.assertEqual(register["kind"], "constant_complex")
        self.assertEqual((register["real"], register["imaginary"]), (0.125, -0.25))
        self.assertIsNone(register["effective_seed"])
        self.assertFalse(register["logged_seed_used"])
        shared = inputs["shared_radix8"]
        self.assertEqual(shared["kind"], "bounded_complex_gaussian")
        self.assertEqual(shared["effective_seed"], 1234567)
        self.assertTrue(shared["logged_seed_used"])
        self.assertEqual(shared["component_clamp"], [-0.75, 0.75])
        artifact_input = method["selected_artifact_timing_only_input"]
        self.assertEqual(artifact_input["kind"], register["kind"])
        self.assertEqual((artifact_input["real"], artifact_input["imaginary"]), (0.125, -0.25))
        self.assertIsNone(artifact_input["effective_seed"])
        self.assertFalse(artifact_input["input_generation_timed"])
        self.assertEqual(artifact_input["warmups_per_repeat"], 1)
        prior_input = method["prior_selected_artifact_timing_only_input"]
        self.assertEqual(prior_input["kind"], shared["kind"])
        self.assertEqual(prior_input["effective_seed"], shared["effective_seed"])
        self.assertFalse(prior_input["included_in_archived_16_measurements"])
        partial_sources = self.manifest["available_partial_unit_sources"]
        self.assertEqual({item["current_variant"] for item in partial_sources},
                         {"historical_e4aa1b0", "historical_652808b"})


if __name__ == "__main__":
    unittest.main()
