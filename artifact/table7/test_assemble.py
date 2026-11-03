#!/usr/bin/env python3
"""CPU fixtures for split Table 7 collection and offline assembly."""
import contextlib
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
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("table7_assemble_checks", HERE / "assemble.py")
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)
measure = assembly.measure

UUID = "GPU-01234567-89ab-cdef-0123-456789abcdef"
GPU = ("gpu=NVIDIA A100 80GB PCIe device=0 sm=8.0 sm_count=108 driver=12090 runtime=12090 cuda_compiler=12.9.86\n"
       f"gpu_uuid={UUID}\n")


def fixture_rows(variant):
    return [dict(zip(measure.RAW_FIELDS, (variant, 64, 64, "tf32", 0, 128, 100, 228, 1, .001, .002, 0, 0, 1)))]


def fixture_profile(part, variant):
    # Deliberately unusable legacy diagnostic sidecars. Table utilization must
    # be recomputed from timing, without reading or requiring these files.
    (part / f"{variant}.ncu.csv").write_text("diagnostic_counter,value\ncycles_active,75.5\n")
    (part / f"{variant}.profile.json").write_text("not a Table 7 utilization input\n")


def fixture_part(root, name, variant="register_uniform", phase="timing"):
    part = root / name
    part.mkdir()
    metadata = {"evaluation": "table7", "profile": "smoke", "phase": phase,
                "status": "completed_declared_subset", "settings": {"batch": 64, "R": 128, "trials": 1, "precision": "TF32"},
                "coverage": {"measured_variants": [variant]},
                "source_sha256": {"ablation/kernels/recovered.cuh": "a" * 64}, "binary_sha256": "b" * 64}
    (part / "metadata.json").write_text(json.dumps(metadata))
    (part / "status.json").write_text(json.dumps({"status": "completed"}))
    rows = []
    if phase in ("timing", "both"):
        rows = fixture_rows(variant)
        measure.write_csv(part / f"{variant}.csv", rows, measure.RAW_FIELDS)
        (part / f"{variant}.log").write_text(GPU)
    measure.write_csv(part / "data.csv", rows, measure.RAW_FIELDS)
    if phase in ("profile", "both"):
        fixture_profile(part, variant)
    return part


class AssemblyChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.output = self.root / "assembled"

    def tearDown(self):
        self.directory.cleanup()

    def assemble(self, parts):
        return assembly.assemble(parts, self.output, "smoke")

    def table(self):
        with (self.output / "table.csv").open() as stream:
            return {row["variant"]: row for row in csv.DictReader(stream)}

    def test_empty_collection_produces_missing_rows_with_blank_measurements(self):
        result = self.assemble([])
        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(result["missing"]), 8)
        for row in self.table().values():
            for key in ("median_time_ms", "utilization_pct", "radix16_speedup"):
                self.assertEqual(row[key], "")
        self.assertFalse(result["full_paper_comparison"])

    def test_separate_timing_parts_automatically_calculate_utilization(self):
        parts = [fixture_part(self.root, "branch", "register_branch"),
                 fixture_part(self.root, "uniform")]
        result = self.assemble(parts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing"]), 4)
        table = self.table()
        self.assertEqual(float(table["register_uniform"]["median_time_ms"]), .001)
        for variant in ("register_uniform", "register_branch"):
            self.assertAlmostEqual(float(table[variant]["utilization_pct"]), .393216 / 156 * 100)
            self.assertEqual(float(table[variant]["reference_peak_tflops"]), 156)
            self.assertEqual(table[variant]["utilization_status"], "calculated")
            self.assertEqual(int(table[variant]["tensor_flops_per_batch"]), 6144 * 64)
        self.assertEqual(list(self.root.rglob("*.ncu.csv")), [])
        self.assertEqual(table["radix16"]["median_time_ms"], "")
        self.assertEqual(result["gpu"]["uuid"], UUID)
        self.assertTrue(result["parts"][0]["files"])

    def test_legacy_profile_only_cannot_supply_paper_utilization(self):
        result = self.assemble([fixture_part(self.root, "profile", phase="profile")])
        self.assertEqual(result["status"], "pending")
        self.assertEqual(self.table()["register_uniform"]["median_time_ms"], "")
        self.assertEqual(self.table()["register_uniform"]["utilization_pct"], "")
        self.assertEqual(len(result["ignored_parts"]), 1)

    def test_legacy_both_reuses_timing_and_ignores_profiler_values(self):
        part = fixture_part(self.root, "both", phase="both")
        result = self.assemble([part])
        value = float(self.table()["register_uniform"]["utilization_pct"])
        self.assertAlmostEqual(value, .393216 / 156 * 100)
        self.assertNotEqual(value, 75.5)
        self.assertTrue(result["parts"][0]["ignored_profile_counters"])
        (part / "register_uniform.ncu.csv").unlink()
        (part / "register_uniform.profile.json").unlink()
        self.assemble([part])
        self.assertEqual(float(self.table()["register_uniform"]["utilization_pct"]), value)

    def test_identical_duplicate_parts_are_harmless(self):
        first = fixture_part(self.root, "first", phase="both")
        second = fixture_part(self.root, "second", phase="both")
        result = self.assemble([first, second, first])
        self.assertEqual(result["collected"]["timing"], ["register_uniform"])
        with (self.output / "data.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 1)

    def test_conflicting_overlap_is_rejected(self):
        first = fixture_part(self.root, "first")
        second = fixture_part(self.root, "second")
        rows = fixture_rows("register_uniform")
        rows[0].update(t_2R_us=356, compute_us=2)
        for name in ("data.csv", "register_uniform.csv"):
            measure.write_csv(second / name, rows, measure.RAW_FIELDS)
        with self.assertRaisesRegex(ValueError, "Conflicting timing"):
            self.assemble([first, second])

    def test_incompatible_metadata_is_rejected(self):
        for field in ("profile", "settings", "source_sha256", "binary_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                first, second = fixture_part(root, "first"), fixture_part(root, "second")
                path = second / "metadata.json"
                meta = json.loads(path.read_text())
                if field == "settings":
                    meta[field]["batch"] = 128
                elif field == "source_sha256":
                    meta[field]["ablation/kernels/recovered.cuh"] = "c" * 64
                elif field == "binary_sha256":
                    meta[field] = "c" * 64
                else:
                    meta[field] = "paper"
                path.write_text(json.dumps(meta))
                with self.assertRaises(ValueError):
                    self.assemble([first, second])

    def test_different_actual_gpu_is_rejected(self):
        first = fixture_part(self.root, "first", "register_branch")
        second = fixture_part(self.root, "second")
        path = second / "register_uniform.log"
        path.write_text(path.read_text().replace(UUID, UUID[:-1] + "0"))
        with self.assertRaisesRegex(ValueError, "different GPUs"):
            self.assemble([first, second])

    def test_failed_parts_and_tampered_exports_are_rejected(self):
        first = fixture_part(self.root, "first")
        (first / "status.json").write_text('{"status":"failed"}')
        with self.assertRaises(ValueError):
            self.assemble([first])
        (first / "status.json").write_text('{"status":"completed"}')
        path = first / "data.csv"
        path.write_text(path.read_text().replace("0.002", "0.003"))
        with self.assertRaisesRegex(ValueError, "Raw and exported"):
            self.assemble([first])

    def test_utilization_requires_valid_numerical_timing_rows(self):
        first = fixture_part(self.root, "first")
        for filename in ("data.csv", "register_uniform.csv"):
            path = first / filename
            path.write_text(path.read_text().replace("0.002", "0.28"))
        with self.assertRaisesRegex(ValueError, "accuracy"):
            self.assemble([first])

    def test_non_a100_has_throughput_but_no_unconfirmed_reference(self):
        first = fixture_part(self.root, "first")
        path = first / "register_uniform.log"
        path.write_text(path.read_text().replace("NVIDIA A100 80GB PCIe", "NVIDIA H100 PCIe"))
        result = self.assemble([first])
        row = self.table()["register_uniform"]
        self.assertAlmostEqual(float(row["achieved_tflops"]), .393216)
        self.assertEqual(row["utilization_pct"], "")
        self.assertEqual(row["utilization_status"], "pending_reference")
        self.assertTrue(any("reference" in item["reason"] for item in result["missing"]))


class CollectionChecks(unittest.TestCase):
    def test_collection_defaults_available_variants_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", THUNDERFFT_COLLECTION="fixture")
            env.pop("THUNDERFFT_TABLE7_VARIANTS", None)
            env.pop("THUNDERFFT_TABLE7_PHASE", None)
            env.pop("THUNDERFFT_TABLE7_NCU", None)
            result = subprocess.run([sys.executable, str(HERE / "measure.py"), "--check-selection", "--output", directory], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            coverage = json.loads((Path(directory) / "coverage.json").read_text())
            self.assertEqual(coverage["requested"], "register_branch,register_uniform")
            self.assertEqual(len(coverage["source_missing"]), 2)

    def test_profiling_is_not_a_table_utilization_phase(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(measure.selected_phase(), "timing")
            os.environ["THUNDERFFT_TABLE7_NCU"] = "1"
            with self.assertRaises(ValueError):
                measure.selected_phase()
            os.environ.pop("THUNDERFFT_TABLE7_NCU")
            for phase in ("profile", "both", "invalid"):
                os.environ["THUNDERFFT_TABLE7_PHASE"] = phase
                with self.subTest(phase=phase), self.assertRaises(ValueError):
                    measure.selected_phase()

    def test_timing_orchestration_calculates_utilization_without_profiler(self):
        # All subprocesses are mocked; this checks orchestration only.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, build, output = root / "fake-binary", root / "build", root / "output"
            binary.write_text("CPU test fixture")
            build.mkdir()
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                self.assertEqual(command, [str(binary), "register_uniform", "64", "128", "1"])
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=measure.RAW_FIELDS)
                writer.writeheader()
                writer.writerows(fixture_rows("register_uniform"))
                kwargs["stdout"].write(stream.getvalue())
                kwargs["stderr"].write(GPU)
                return subprocess.CompletedProcess(command, 0)
            env = {"THUNDERFFT_COLLECTION": "fixture", "THUNDERFFT_TABLE7_VARIANTS": "register_uniform",
                   "THUNDERFFT_TABLE7_PHASE": "timing", "THUNDERFFT_TABLE7_BATCH": "64",
                   "THUNDERFFT_TABLE7_REPEATS": "128", "THUNDERFFT_TABLE7_TRIALS": "1"}
            argv = ["measure.py", "--output", str(output), "--binary", str(binary), "--build-dir", str(build)]
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(measure, "capture", return_value={}), \
                    patch.object(measure.subprocess, "run", side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 0)
            with (output / "data.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(list(output.glob("*.ncu.csv")), [])
            with (output / "table.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertAlmostEqual(float(row["utilization_pct"]), .393216 / 156 * 100)
            self.assertEqual(json.loads((output / "metadata.json").read_text())["phase"], "timing")


if __name__ == "__main__":
    unittest.main()
