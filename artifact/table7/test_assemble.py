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
GPU = ("gpu=Test GPU device=0 sm=8.0 sm_count=108 driver=12090 runtime=12090 cuda_compiler=12.9.86\n"
       f"gpu_uuid={UUID}\n")


def fixture_rows(variant):
    return [dict(zip(measure.RAW_FIELDS, (variant, 64, 64, "tf32", 0, 128, 100, 228, 1, .001, .002, 0, 0, 1)))]


def fixture_profile_log(variant):
    result = GPU + f"variant={variant} N=64 precision=tf32 batch=64 R=32 trials=1 mode=profile\n"
    for kind in ("bounded_complex_gaussian", "complex_tone"):
        result += (f"validation_input={kind} validation_batch=64 R=1 reference=cpu_fp64_dft "
                   "relative_l2=0.001 max_transform_relative_l2=0.002 nonfinite_outputs=0 "
                   "output_guard_changed=0 threshold=0.005 passed=1\n")
    return result + "profile_launch_complete=1 profile_R=32 profile_warmups=0 profile_timing_rows=0\n"


def fixture_profile(part, variant, binary_hash="b" * 64):
    kernel = "fft_kernel_radix64_batch16" + ("_branch" if variant == "register_branch" else "")
    raw = ('"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
           f'"0","{kernel}","{assembly.profiler.DEFAULT_METRIC}","%","75.5"\n')
    (part / f"{variant}.ncu.csv").write_text(raw)
    (part / f"{variant}.profile.log").write_text(fixture_profile_log(variant))
    record = {"variant": variant, "batch": 64, "R": 32, "exit_code": 0,
              "clock_control": "none", "cache_control": "none", "profiler_version": "fixture",
              "argv": ["ncu", "--launch-skip", "2", "--launch-count", "1", "--clock-control", "none", "--cache-control", "none", "binary", variant, "64", "32", "1", "--profile"],
              "binary_sha256": binary_hash, "metric": assembly.profiler.DEFAULT_METRIC,
              "measurement": assembly.profiler.parse_metric(raw, assembly.profiler.DEFAULT_METRIC)}
    (part / f"{variant}.profile.json").write_text(json.dumps(record))
    return record["measurement"]


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

    def test_separate_variants_and_profile_parts_merge_without_filling_missing(self):
        parts = [fixture_part(self.root, "branch", "register_branch"),
                 fixture_part(self.root, "uniform"),
                 fixture_part(self.root, "uniform-utilization", phase="profile")]
        result = self.assemble(parts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing"]), 5)
        table = self.table()
        self.assertEqual(float(table["register_uniform"]["median_time_ms"]), .001)
        self.assertEqual(float(table["register_uniform"]["utilization_pct"]), 75.5)
        self.assertEqual(table["register_branch"]["utilization_pct"], "")
        self.assertEqual(table["radix16"]["median_time_ms"], "")
        self.assertEqual(result["gpu"]["uuid"], UUID)
        self.assertTrue(result["parts"][0]["files"])

    def test_profile_only_is_useful_without_timing(self):
        result = self.assemble([fixture_part(self.root, "profile", phase="profile")])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(self.table()["register_uniform"]["median_time_ms"], "")
        self.assertEqual(float(self.table()["register_uniform"]["utilization_pct"]), 75.5)

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
                first, second = fixture_part(root, "first"), fixture_part(root, "second", phase="profile")
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

    def test_profile_requires_native_numerical_gates(self):
        first = fixture_part(self.root, "first", phase="profile")
        path = first / "register_uniform.profile.log"
        path.write_text(path.read_text().replace("passed=1", "passed=0"))
        with self.assertRaisesRegex(ValueError, "correctness"):
            self.assemble([first])

    def test_profile_counter_provenance_cannot_be_swapped(self):
        first = fixture_part(self.root, "first", phase="profile")
        path = first / "register_uniform.profile.json"
        record = json.loads(path.read_text())
        record["binary_sha256"] = "c" * 64
        path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.assemble([first])


class CollectionChecks(unittest.TestCase):
    def test_collection_defaults_available_variants_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", THUNDERFFT_COLLECTION="fixture")
            env.pop("THUNDERFFT_TABLE7_VARIANTS", None)
            result = subprocess.run([sys.executable, str(HERE / "measure.py"), "--check-selection", "--output", directory], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            coverage = json.loads((Path(directory) / "coverage.json").read_text())
            self.assertEqual(coverage["requested"], "register_branch,register_uniform")
            self.assertEqual(len(coverage["source_missing"]), 2)

    def test_phase_selection_keeps_legacy_ncu_flag(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(measure.selected_phase(), "timing")
            os.environ["THUNDERFFT_TABLE7_NCU"] = "1"
            self.assertEqual(measure.selected_phase(), "both")
            os.environ["THUNDERFFT_TABLE7_PHASE"] = "profile"
            self.assertEqual(measure.selected_phase(), "profile")
            os.environ["THUNDERFFT_TABLE7_PHASE"] = "invalid"
            with self.assertRaises(ValueError):
                measure.selected_phase()

    def test_profile_only_orchestration_does_not_run_timing(self):
        # All subprocesses are mocked; this checks orchestration only.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, build, output = root / "fake-binary", root / "build", root / "output"
            binary.write_text("CPU test fixture")
            build.mkdir()
            def collect(_binary, variant, batch, destination, _ncu, _metric):
                self.assertEqual(batch, 64)
                return fixture_profile(destination, variant)
            env = {"THUNDERFFT_COLLECTION": "fixture", "THUNDERFFT_TABLE7_VARIANTS": "register_uniform",
                   "THUNDERFFT_TABLE7_PHASE": "profile", "THUNDERFFT_TABLE7_BATCH": "64",
                   "THUNDERFFT_TABLE7_REPEATS": "128", "THUNDERFFT_TABLE7_TRIALS": "1"}
            argv = ["measure.py", "--output", str(output), "--binary", str(binary), "--build-dir", str(build)]
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(measure, "capture", return_value={}), \
                    patch.dict(sys.modules, {"collect_profile": assembly.profiler}), \
                    patch.object(assembly.profiler, "collect", side_effect=collect), \
                    patch.object(measure.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 0)
            with (output / "data.csv").open() as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            self.assertFalse((output / "register_uniform.csv").exists())
            self.assertEqual(json.loads((output / "metadata.json").read_text())["phase"], "profile")


if __name__ == "__main__":
    unittest.main()
