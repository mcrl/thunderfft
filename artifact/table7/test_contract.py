#!/usr/bin/env python3
"""CPU-only checks for Table 7 selection and native-result validation."""

import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("table7_measure", HERE / "measure.py")
measure = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = measure
SPEC.loader.exec_module(measure)

FIELDS = (
    "variant", "N", "batch", "precision", "trial", "inside_repeats",
    "t_R_us", "t_2R_us", "compute_us", "relative_l2",
    "max_transform_relative_l2", "nonfinite_outputs", "output_guard_changed", "passed",
)


class VariantSelectionContract(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "schema_version": 1,
            "variants": [
                {"id": "radix16", "status": "missing"},
                {"id": "radix4_compute", "status": "missing"},
                {"id": "register_branch", "status": "available"},
                {"id": "register_uniform", "status": "available"},
            ],
        }

    def test_default_selection_cannot_silently_drop_missing_variants(self):
        for text in (None, "all"):
            with self.subTest(text=text), self.assertRaises(measure.MissingVariants):
                measure.select_variants(text, self.manifest)

    def test_available_subset_requires_explicit_selection(self):
        selected = measure.select_variants(
            "register_branch,register_uniform", self.manifest)
        self.assertEqual([entry["id"] for entry in selected],
                         ["register_branch", "register_uniform"])

    def test_requested_order_is_preserved(self):
        selected = measure.select_variants(
            "register_uniform,register_branch", self.manifest)
        self.assertEqual([entry["id"] for entry in selected],
                         ["register_uniform", "register_branch"])

    def test_missing_unknown_duplicate_or_empty_selection_is_rejected(self):
        for text in ("radix16", "register_branch,radix4_compute", "unknown",
                     "register_branch,register_branch", "register_branch,", ","):
            with self.subTest(text=text), self.assertRaises(ValueError):
                measure.select_variants(text, self.manifest)

    def test_shipped_manifest_preserves_full_paper_coverage(self):
        manifest = json.loads((HERE / "variants.json").read_text())
        entries = {entry["id"]: entry for entry in manifest["variants"]}
        self.assertEqual(set(entries), {
            "radix16", "radix4_compute", "register_branch", "register_uniform",
        })
        self.assertEqual(entries["radix16"]["status"], "missing")
        self.assertEqual(entries["radix4_compute"]["status"], "missing")
        self.assertEqual(entries["register_branch"]["status"], "available")
        self.assertEqual(entries["register_uniform"]["status"], "available")

    def test_default_selection_check_exits_pending_before_measurement(self):
        environment = dict(os.environ)
        environment.pop("THUNDERFFT_TABLE7_VARIANTS", None)
        environment.pop("THUNDERFFT_COLLECTION", None)
        environment["CUDA_VISIBLE_DEVICES"] = ""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(HERE / "measure.py"), "--check-selection",
                 "--output", directory], env=environment, text=True,
                capture_output=True, timeout=15, check=False,
            )
            self.assertEqual(result.returncode, 78, result.stdout + result.stderr)
            coverage = json.loads((Path(directory) / "coverage.json").read_text())
            self.assertIn("radix16", json.dumps(coverage))
            self.assertIn("radix4_compute", json.dumps(coverage))
            self.assertFalse((Path(directory) / "data.csv").exists())


class RecoveredSourceContract(unittest.TestCase):
    def test_recovered_candidates_retain_original_commit_provenance(self):
        path = HERE / "ablation" / "kernels" / "provenance.json"
        provenance = json.loads(path.read_text())
        self.assertEqual(provenance["source_repository"],
                         "https://github.com/rhrnald/fft")
        self.assertEqual(provenance["source_commit"],
                         "403c6ae8b152684d5b60bf04ebf067f02e97e026")
        self.assertIn(provenance["source_commit"], provenance["source_commit_url"])
        self.assertTrue(provenance["source_files"])
        for source in provenance["source_files"]:
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
        self.assertTrue(provenance["extracted_functions"])
        for function in provenance["extracted_functions"]:
            self.assertRegex(function["source_excerpt_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(len(function["source_lines"]), 2)
            self.assertLessEqual(*function["source_lines"])

    def test_recovered_kernel_names_and_adaptations_are_shipped(self):
        kernels = HERE / "ablation" / "kernels"
        code = (kernels / "recovered.cuh").read_text()
        provenance = json.loads((kernels / "provenance.json").read_text())
        names = {entry["name"].rsplit("::", 1)[-1]
                 for entry in provenance["kernels"]}
        self.assertEqual(names, {
            "fft_kernel_radix64_batch16", "fft_kernel_radix64_batch16_branch",
        })
        for name in names:
            self.assertIn(name + "(", code)
        self.assertIn("namespace thff_table7_legacy", code)
        self.assertTrue(provenance["adaptations"])
        self.assertTrue(provenance["known_limitations"])


class NativeOutputContract(unittest.TestCase):
    def raw(self, **changes):
        row = dict(zip(FIELDS, (
            "register_uniform", 64, 64, "tf32", 0, 128,
            100, 228, 1, 0.001, 0.002, 0, 0, 1,
        )))
        row.update(changes)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
        return stream.getvalue()

    def parse(self, text, **changes):
        options = dict(variant="register_uniform", batch=64, repeats=128, trials=1)
        options.update(changes)
        return measure.parse_raw(text, **options)

    def test_valid_native_csv(self):
        self.assertEqual(len(self.parse(self.raw())), 1)

    def test_numerical_failures_are_rejected(self):
        failures = (
            {"relative_l2": "nan"}, {"relative_l2": "inf"},
            {"relative_l2": -0.001}, {"relative_l2": 0.005001},
            {"max_transform_relative_l2": "nan"},
            {"max_transform_relative_l2": "inf"},
            {"max_transform_relative_l2": -0.001},
            {"max_transform_relative_l2": 0.005001},
            {"nonfinite_outputs": 1}, {"nonfinite_outputs": -1},
            {"output_guard_changed": 1}, {"output_guard_changed": -1},
            {"passed": 0},
        )
        for changes in failures:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(self.raw(**changes))

    def test_error_threshold_is_inclusive(self):
        self.assertEqual(len(self.parse(self.raw(
            relative_l2=0.005, max_transform_relative_l2=0.005))), 1)

    def test_timings_must_be_finite_positive_and_consistent(self):
        failures = (
            {"t_R_us": 0}, {"t_R_us": -1}, {"t_R_us": "nan"},
            {"t_2R_us": 0}, {"t_2R_us": "inf"}, {"t_2R_us": 100},
            {"t_2R_us": 99}, {"compute_us": 0},
            {"compute_us": -1}, {"compute_us": "nan"},
            {"compute_us": "inf"}, {"compute_us": 2},
        )
        for changes in failures:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(self.raw(**changes))

    def test_native_configuration_must_match_requested_run(self):
        failures = (
            {"variant": "register_branch"}, {"N": 128},
            {"batch": 65536}, {"precision": "fp16"},
            {"inside_repeats": 127}, {"trial": -1}, {"trial": 1},
        )
        for changes in failures:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(self.raw(**changes))

    def test_incomplete_duplicate_or_malformed_csv_is_rejected(self):
        valid = self.raw()
        inputs = (
            ("", {}),
            (valid.splitlines()[0] + "\n", {}),
            (valid, {"trials": 2}),
            (valid + valid.splitlines()[1] + "\n", {}),
            (valid + "register_uniform,64\n", {}),
            (valid.replace("compute_us", "time"), {}),
            (valid.replace("128,100", "128.0,100"), {}),
        )
        for text, changes in inputs:
            with self.subTest(text=text, changes=changes), self.assertRaises(ValueError):
                self.parse(text, **changes)

    def test_all_requested_trials_are_accepted_once(self):
        text = self.raw() + self.raw(trial=1).splitlines()[1] + "\n"
        self.assertEqual(len(self.parse(text, trials=2)), 2)


if __name__ == "__main__":
    unittest.main()
