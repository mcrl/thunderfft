#!/usr/bin/env python3
"""CPU fixtures for split Table 7 collection and offline assembly."""
import contextlib
import csv
import hashlib
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


@contextlib.contextmanager
def future_recovered_variants():
    """Exercise full-table behavior with explicitly fictional recovered sources."""
    path = HERE / "variants.json"
    manifest = json.loads(path.read_text())
    manifest["candidate_implementations"] = []
    uniform_source = next(entry["source"] for entry in manifest["variants"] if entry["id"] == "register_uniform")
    for entry in manifest["variants"]:
        if entry["status"] != "available":
            entry.update(status="available", implementation_kind="recovered",
                         source=uniform_source)
    # Fictional future recovery also replaces the current artifact selection.
    manifest["paper_row_sources"] = {entry["id"]: {"source_variant": entry["id"],
                                   "selection_status": "fixture_recovered"}
                                   for entry in manifest["variants"]}
    original_read = Path.read_text
    def read_text(candidate, *args, **kwargs):
        if candidate == path:
            return json.dumps(manifest)
        return original_read(candidate, *args, **kwargs)
    # These fixture counts are not declarations about the unrecovered sources.
    with patch.object(Path, "read_text", read_text), \
            patch.dict(measure.utilization.FLOPS_PER_FFT, {"radix16": 12288, "radix4_compute": 6144}):
        yield


def fixture_rows(variant, time_us=1):
    return [dict(zip(measure.RAW_FIELDS, (variant, 64, 64, "tf32", 0, 128,
                                       100, 100 + 128 * time_us, time_us, .001, .002, 0, 0, 1)))]


def fixture_profile(part, variant):
    # Deliberately unusable legacy diagnostic sidecars. Table utilization must
    # be recomputed from timing, without reading or requiring these files.
    (part / f"{variant}.ncu.csv").write_text("diagnostic_counter,value\ncycles_active,75.5\n")
    (part / f"{variant}.profile.json").write_text("not a Table 7 utilization input\n")


def fixture_part(root, name, variant="register_uniform", phase="timing", time_us=1,
                 timing_only=False, trials=1, input_pattern=None):
    part = root / name
    part.mkdir()
    manifest = json.loads((HERE / "variants.json").read_text())
    hashes = {source: hashlib.sha256((HERE / source).read_bytes()).hexdigest()
              for entry in measure.manifest_entries(manifest) if entry["status"] == "available"
              for source in (entry["source"], *entry.get("source_dependencies", []))}
    hashes["measure.py"] = "a" * 64
    metadata = {"evaluation": "table7", "profile": "smoke", "phase": phase,
                "status": "completed_timing_only" if timing_only else "completed_declared_subset",
                "timing_only": timing_only, "accuracy_status": "unchecked" if timing_only else "passed",
                "settings": {"batch": 64, "R": 128, "trials": trials, "precision": "TF32",
                             "warmups_per_repeat_count": 1 if timing_only else 3},
                "coverage": {"measured_variants": [variant]},
                "source_sha256": hashes, "binary_sha256": "b" * 64}
    if input_pattern is not None:
        metadata["settings"]["input_pattern"] = input_pattern
    (part / "metadata.json").write_text(json.dumps(metadata))
    (part / "status.json").write_text(json.dumps({"status": "completed"}))
    rows = []
    if phase in ("timing", "both"):
        rows = [dict(fixture_rows(variant, time_us)[0], trial=trial) for trial in range(trials)]
        if timing_only:
            for row in rows:
                row.update(relative_l2="nan", max_transform_relative_l2="nan",
                           nonfinite_outputs=-1, output_guard_changed=-1, passed=0)
        measure.write_csv(part / f"{variant}.csv", rows, measure.RAW_FIELDS)
        log = GPU + ("accuracy_status=unchecked timing_only=1 R1_validation=0 output_guard_checks=0\n"
                     if timing_only else "")
        if input_pattern is not None:
            log += f"timing_input={input_pattern}\n"
        (part / f"{variant}.log").write_text(log)
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

    def test_reconstructed_shared_stage_maps_to_paper_role_with_unchanged_native_evidence(self):
        for timing_only in (False, True):
            with self.subTest(timing_only=timing_only):
                part = fixture_part(self.root, f"shared-{timing_only}", "shared_stage_radix4",
                                    timing_only=timing_only, time_us=12)
                result = self.assemble([part])
                row = self.table()["radix4_compute"]
                self.assertEqual(row["source_variant"], "shared_stage_radix4")
                self.assertEqual(row["implementation_kind"], "reconstructed")
                self.assertEqual(row["source_commit"], "")
                self.assertEqual(row["paper_identity"], "unconfirmed")
                self.assertEqual(row["operation_coverage"], "complete_fft_candidate")
                self.assertEqual(row["accuracy_status"], "unchecked" if timing_only else "passed")
                self.assertEqual(float(row["median_time_ms"]), .012)
                self.assertEqual(int(row["tensor_flops_per_fft"]), 6144)
                self.assertEqual(row["source_sha256"], hashlib.sha256((HERE / row["source"]).read_bytes()).hexdigest())
                self.assertEqual(row["binary_sha256"], "b" * 64)
                self.assertFalse(result["full_paper_comparison"])
                with (self.output / "kernel_comparison.csv").open() as stream:
                    self.assertEqual([item["variant"] for item in csv.DictReader(stream)], ["shared_stage_radix4"])
                self.assertNotIn("radix4_compute.timing", {item["item"] for item in result["missing"]})

    def test_reconstructed_shared_stage_requires_matching_source_and_has_no_fallback(self):
        part = fixture_part(self.root, "stale-shared", "shared_stage_radix4")
        path = part / "metadata.json"
        meta = json.loads(path.read_text())
        meta["source_sha256"]["ablation/kernels/shared_stage_radix4.cuh"] = "f" * 64
        path.write_text(json.dumps(meta))
        result = self.assemble([part])
        row = self.table()["radix4_compute"]
        self.assertEqual(row["source_variant"], "shared_stage_radix4")
        self.assertEqual(row["timing_status"], "missing")
        self.assertEqual(row["median_time_ms"], "")
        self.assertEqual(result["ignored_variants"][0]["variant"], "shared_stage_radix4")

    def test_reconstructed_shared_stage_rejects_missing_or_stale_helper_hash(self):
        dependency = "ablation/kernels/historical_uniform_5e37821.cuh"
        for stale in (False, True):
            with self.subTest(stale=stale):
                part = fixture_part(self.root, f"dependency-{stale}", "shared_stage_radix4")
                path = part / "metadata.json"
                meta = json.loads(path.read_text())
                if stale:
                    meta["source_sha256"][dependency] = "f" * 64
                else:
                    del meta["source_sha256"][dependency]
                path.write_text(json.dumps(meta))
                result = self.assemble([part])
                row = self.table()["radix4_compute"]
                self.assertEqual(row["timing_status"], "missing")
                self.assertEqual(row["median_time_ms"], "")
                ignored, = result["ignored_variants"]
                self.assertEqual(ignored["variant"], "shared_stage_radix4")
                self.assertIn("source dependency hash", ignored["reason"])
                self.assertIn(dependency, ignored["reason"])
                self.assertIn("does not match" if stale else "missing", ignored["reason"])

    def test_separate_timing_parts_automatically_calculate_utilization(self):
        parts = [fixture_part(self.root, "branch", "historical_e4aa1b0", timing_only=True),
                 fixture_part(self.root, "uniform")]
        result = self.assemble(parts)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing"]), 4)
        table = self.table()
        self.assertEqual(float(table["register_uniform"]["median_time_ms"]), .001)
        for variant in ("register_uniform", "register_branch"):
            self.assertAlmostEqual(float(table[variant]["utilization_pct"]), .393216 / 156 * 100)
            self.assertEqual(float(table[variant]["reference_peak_tflops"]), 156)
            self.assertEqual(table[variant]["utilization_status"],
                             "calculated_unvalidated" if variant == "register_branch" else "calculated")
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

    def test_unchecked_and_validated_parts_preserve_different_protocols_and_sources(self):
        diagnostic = fixture_part(self.root, "unchecked", "historical_e4aa1b0", timing_only=True,
                                  trials=2, input_pattern="constant_0.125_minus0.25")
        path = diagnostic / "metadata.json"
        meta = json.loads(path.read_text())
        meta["binary_sha256"] = "c" * 64
        meta["source_sha256"]["measure.py"] = "d" * 64
        path.write_text(json.dumps(meta))
        valid = fixture_part(self.root, "validated", input_pattern="bounded_complex_gaussian_seed1234567")
        result = self.assemble([diagnostic, valid])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["collected"]["timing"], ["historical_e4aa1b0", "register_uniform"])
        self.assertEqual(result["collected"]["unchecked"], ["historical_e4aa1b0"])
        self.assertEqual(result["accuracy_status"], "mixed")
        self.assertTrue(result["protocol_heterogeneous"])
        self.assertEqual(result["ignored_parts"], [])
        branch = self.table()["register_branch"]
        self.assertEqual(branch["timing_status"], "collected_unvalidated")
        self.assertEqual(branch["accuracy_status"], "unchecked")
        self.assertEqual(branch["utilization_status"], "calculated_unvalidated")
        self.assertEqual(branch["max_transform_relative_l2"], "")
        self.assertEqual(branch["trials"], "2")
        self.assertEqual(branch["warmups_per_repeat_count"], "1")
        self.assertEqual(branch["binary_sha256"], "c" * 64)
        self.assertEqual(branch["input_pattern"], "constant_0.125_minus0.25")
        self.assertEqual(float(self.table()["register_uniform"]["median_time_ms"]), .001)
        self.assertFalse(result["full_paper_comparison"])

    def test_four_separate_timings_complete_table_and_calculate_baseline_speedup(self):
        times = {"radix16": 8, "radix4_compute": 12, "register_branch": 7, "register_uniform": 4}
        parts = [fixture_part(self.root, variant, variant, time_us=time_us)
                 for variant, time_us in times.items()]
        with future_recovered_variants():
            result = self.assemble(parts)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["missing"], [])
        self.assertFalse(result["full_paper_comparison"])
        self.assertEqual(result["paper_identity"], "unconfirmed")
        for variant, row in self.table().items():
            self.assertAlmostEqual(float(row["radix16_speedup"]), 8 / times[variant])
            expected_flops = 12288 if variant == "radix16" else 6144
            self.assertEqual(int(row["tensor_flops_per_fft"]), expected_flops)
            self.assertAlmostEqual(float(row["utilization_pct"]), expected_flops * 64 / times[variant] / 1e6 / 156 * 100)

    def test_selected_historical_kernel_supplies_only_its_selected_presentation_row(self):
        parts = [fixture_part(self.root, variant, variant, timing_only=True)
                 for variant in ("historical_e4aa1b0", "historical_652808b")]
        result = self.assemble(parts)
        table = self.table()
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(table), 5)
        self.assertEqual(len(result["missing"]), 6)
        for variant in ("radix16", "radix4_compute", "register_uniform"):
            self.assertEqual(table[variant]["median_time_ms"], "")
        for variant in ("register_branch", "historical_652808b"):
            row = table[variant]
            self.assertEqual(row["accuracy_status"], "unchecked")
            self.assertEqual(row["implementation_kind"], "historical_partial_kernel")
            self.assertTrue(row["operation_coverage"].startswith("partial_fft_"))
            self.assertIn("not a mathematically valid FFT", row["known_source_issues"])
            self.assertEqual(row["timing_scope"], "repeated_kernel_compute")
            self.assertEqual(row["paper_identity"], "unconfirmed")
            self.assertEqual(row["radix16_speedup"], "")
            self.assertRegex(row["source_commit"], "^[0-9a-f]{40}$")
            self.assertRegex(row["source_sha256"], "^[0-9a-f]{64}$")
        with (self.output / "kernel_comparison.csv").open() as stream:
            comparison = list(csv.DictReader(stream))
        self.assertEqual([row["variant"] for row in comparison], ["historical_e4aa1b0", "historical_652808b"])
        self.assertFalse(result["full_paper_comparison"])

    def test_native_branch_never_substitutes_for_selected_e4aa_source(self):
        original = fixture_part(self.root, "original", "register_branch", time_us=11.94)
        result = self.assemble([original])
        row = self.table()["register_branch"]
        self.assertEqual(row["source_variant"], "historical_e4aa1b0")
        self.assertEqual(row["median_time_ms"], "")
        self.assertTrue(any(item["item"] == "register_branch.timing" for item in result["missing"]))
        with (self.output / "kernel_comparison.csv").open() as stream:
            measured = list(csv.DictReader(stream))
        self.assertEqual([row["variant"] for row in measured], ["register_branch"])
        self.assertAlmostEqual(float(measured[0]["median_time_ms"]), .01194)

    def test_selected_source_does_not_duplicate_or_relabel_native_exports(self):
        parts = [fixture_part(self.root, "original", "register_branch", time_us=11.94),
                 fixture_part(self.root, "selected", "historical_e4aa1b0", time_us=7.9, timing_only=True)]
        result = self.assemble(parts)
        table = self.table()
        self.assertEqual(len(table), 4)
        self.assertNotIn("historical_e4aa1b0", table)
        selected = table["register_branch"]
        self.assertEqual(selected["source_variant"], "historical_e4aa1b0")
        self.assertAlmostEqual(float(selected["median_time_ms"]), .0079)
        self.assertTrue(selected["source_commit"].startswith("e4aa1b0"))
        self.assertEqual(selected["accuracy_status"], "unchecked")
        self.assertEqual(selected["paper_identity"], "unconfirmed")
        self.assertIn("operand_setup_omitted", selected["operation_coverage"])
        self.assertEqual(result["paper_row_sources"]["register_branch"]["selection_status"], "artifact_selected")
        for name in ("data.csv", "kernel_comparison.csv"):
            with (self.output / name).open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["variant"] for row in rows], ["register_branch", "historical_e4aa1b0"])

    def test_archived_e4aa_evidence_is_reusable_without_rewriting_it(self):
        archive = HERE / "reference/a100-20261107"
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in archive.rglob("*") if path.is_file()}
        result = assembly.assemble([archive], self.output, "paper")
        selected = self.table()["register_branch"]
        self.assertAlmostEqual(float(selected["median_time_ms"]), .00788889598846)
        self.assertEqual(selected["source_variant"], "historical_e4aa1b0")
        self.assertEqual(selected["accuracy_status"], "unchecked")
        self.assertEqual(selected["timing_status"], "collected_unvalidated")
        self.assertNotIn("register_branch.timing", {item["item"] for item in result["missing"]})
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before})

    def test_partial_kernel_cannot_claim_validated_output(self):
        part = fixture_part(self.root, "partial", "historical_e4aa1b0")
        with self.assertRaisesRegex(ValueError, "requires timing-only"):
            self.assemble([part])

    def test_unchecked_duplicate_nan_rows_are_equal_but_changed_provenance_is_rejected(self):
        first = fixture_part(self.root, "first", timing_only=True)
        second = fixture_part(self.root, "second", timing_only=True)
        result = self.assemble([first, second])
        self.assertEqual(result["collected"]["unchecked"], ["register_uniform"])
        path = second / "metadata.json"
        meta = json.loads(path.read_text())
        meta["binary_sha256"] = "c" * 64
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Conflicting source or measurement provenance"):
            self.assemble([first, second])

    def test_unchecked_parts_need_consistent_metadata_log_and_raw_sentinels(self):
        for fault in ("metadata", "log", "raw", "input", "coverage"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                part = fixture_part(root, "part", timing_only=True)
                if fault in ("metadata", "input", "coverage"):
                    path = part / "metadata.json"
                    meta = json.loads(path.read_text())
                    if fault == "metadata":
                        meta["accuracy_status"] = "passed"
                    elif fault == "input":
                        meta["settings"]["input_pattern"] = "constant_0.125_minus0.25"
                    else:
                        meta["variant_provenance"] = {"register_uniform": {"operation_coverage": "fabricated"}}
                    path.write_text(json.dumps(meta))
                elif fault == "log":
                    (part / "register_uniform.log").write_text(GPU)
                else:
                    for filename in ("data.csv", "register_uniform.csv"):
                        path = part / filename
                        path.write_text(path.read_text().replace(",nan,nan,", ",0,nan,"))
                with self.assertRaises(ValueError):
                    assembly.assemble([part], root / "output", "smoke")

    def test_baseline_subset_keeps_uncollected_variants_blank(self):
        parts = [fixture_part(self.root, "baseline", "radix16", time_us=8),
                 fixture_part(self.root, "uniform", time_us=4)]
        with future_recovered_variants():
            result = self.assemble(parts)
        self.assertEqual(result["status"], "partial")
        table = self.table()
        self.assertEqual(float(table["radix16"]["radix16_speedup"]), 1)
        self.assertEqual(float(table["register_uniform"]["radix16_speedup"]), 2)
        self.assertEqual(table["radix4_compute"]["radix16_speedup"], "")
        self.assertEqual(table["register_branch"]["median_time_ms"], "")

    def test_previous_reconstruction_cannot_complete_missing_paper_rows(self):
        parts = [fixture_part(self.root, "baseline", "radix16", time_us=8),
                 fixture_part(self.root, "shared", "radix4_compute", time_us=12)]
        result = self.assemble(parts)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["collected"]["timing"], [])
        self.assertEqual({entry["variant"] for entry in result["ignored_variants"]},
                         {"radix16", "radix4_compute"})
        for entry in result["ignored_variants"]:
            self.assertIn("not been recovered", entry["reason"])
        for row in self.table().values():
            self.assertEqual(row["median_time_ms"], "")
            self.assertEqual(row["radix16_speedup"], "")

    def test_old_mixed_bundle_keeps_only_unchanged_recovered_source(self):
        part = fixture_part(self.root, "old-all")
        selected = ["radix16", "radix4_compute", "register_branch", "register_uniform"]
        combined = []
        for variant in selected:
            rows = fixture_rows(variant)
            combined.extend(rows)
            measure.write_csv(part / f"{variant}.csv", rows, measure.RAW_FIELDS)
            (part / f"{variant}.log").write_text(GPU)
        measure.write_csv(part / "data.csv", combined, measure.RAW_FIELDS)
        path = part / "metadata.json"
        meta = json.loads(path.read_text())
        meta["coverage"]["measured_variants"] = selected
        # The old all-four binary had only the 403c6ae branch source, so it must
        # not be presented as a measurement of the new 51e346c branch candidate.
        del meta["source_sha256"]["ablation/kernels/historical_branch_51e346c.cuh"]
        path.write_text(json.dumps(meta))
        result = self.assemble([part])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["collected"]["timing"], ["register_uniform"])
        self.assertEqual(result["parts"][0]["variants"], ["register_uniform"])
        ignored = {entry["variant"]: entry["reason"] for entry in result["ignored_variants"]}
        self.assertEqual(set(ignored), {"radix16", "radix4_compute", "register_branch"})
        self.assertIn("source hash", ignored["register_branch"])
        self.assertEqual(float(self.table()["register_uniform"]["median_time_ms"]), .001)
        self.assertEqual(self.table()["register_branch"]["median_time_ms"], "")
        self.assertTrue(all(row["radix16_speedup"] == "" for row in self.table().values()))

    def test_experimental_variant_ids_are_not_paper_rows(self):
        part = fixture_part(self.root, "experimental", "reconstructed_radix16")
        with self.assertRaisesRegex(ValueError, "measured-variant coverage"):
            self.assemble([part])

    def test_changed_current_source_is_not_accepted_as_the_same_candidate(self):
        part = fixture_part(self.root, "changed")
        path = part / "metadata.json"
        meta = json.loads(path.read_text())
        manifest = json.loads((HERE / "variants.json").read_text())
        source = next(entry["source"] for entry in manifest["variants"] if entry["id"] == "register_uniform")
        meta["source_sha256"][source] = "f" * 64
        path.write_text(json.dumps(meta))
        result = self.assemble([part])
        self.assertEqual(result["status"], "pending")
        self.assertIn("source hash", result["ignored_variants"][0]["reason"])

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
                    meta[field]["measure.py"] = "c" * 64
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
    def test_default_selection_includes_selected_historical_kernel_and_validated_mode_remains_available(self):
        manifest = json.loads((HERE / "variants.json").read_text())
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(measure.requested_variants(manifest),
                             "register_branch,register_uniform,shared_stage_radix4,historical_e4aa1b0,historical_652808b")
            os.environ["THUNDERFFT_TABLE7_TIMING_ONLY"] = "0"
            self.assertEqual(measure.requested_variants(manifest), "register_branch,register_uniform,shared_stage_radix4")
            os.environ["THUNDERFFT_TABLE7_VARIANTS"] = "all"
            self.assertEqual(measure.requested_variants(manifest), "all")

    def test_timing_only_default_and_explicit_validation_choice_match_native(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(measure.selected_timing_only())
            os.environ["THUNDERFFT_TABLE7_TIMING_ONLY"] = "0"
            self.assertFalse(measure.selected_timing_only())
            os.environ["THUNDERFFT_TABLE7_TIMING_ONLY"] = "true"
            with self.assertRaises(ValueError):
                measure.selected_timing_only()
            for key, maximum in (("THUNDERFFT_TABLE7_BATCH", 1048576), ("THUNDERFFT_TABLE7_TRIALS", 100)):
                os.environ[key] = str(maximum)
                self.assertEqual(measure.setting(key, 1, maximum=maximum), maximum)
                os.environ[key] = str(maximum + 1)
                with self.assertRaises(ValueError):
                    measure.setting(key, 1, maximum=maximum)

    def test_timing_only_routes_native_flag_and_assembles_unvalidated_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, build, output = root / "fake-binary", root / "build", root / "output"
            binary.write_text("CPU fixture")
            build.mkdir()
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                self.assertEqual(command, [str(binary), "register_uniform", "64", "128", "1", "--timing-only"])
                rows = fixture_rows("register_uniform")
                rows[0].update(relative_l2="nan", max_transform_relative_l2="nan",
                               nonfinite_outputs=-1, output_guard_changed=-1, passed=0)
                writer = csv.DictWriter(kwargs["stdout"], fieldnames=measure.RAW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
                kwargs["stderr"].write(GPU + "accuracy_status=unchecked timing_only=1 R1_validation=0 output_guard_checks=0\n"
                                       "timing_input=constant_0.125_minus0.25\n")
                return subprocess.CompletedProcess(command, 0)
            env = {"THUNDERFFT_TABLE7_VARIANTS": "register_uniform", "THUNDERFFT_TABLE7_TIMING_ONLY": "1",
                   "THUNDERFFT_TABLE7_BATCH": "64", "THUNDERFFT_TABLE7_REPEATS": "128", "THUNDERFFT_TABLE7_TRIALS": "1"}
            argv = ["measure.py", "--output", str(output), "--binary", str(binary), "--build-dir", str(build)]
            with patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(measure, "capture", return_value={}), \
                    patch.object(measure.subprocess, "run", side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 0)
            self.assertEqual(len(calls), 1)
            meta = json.loads((output / "metadata.json").read_text())
            self.assertEqual(meta["status"], "completed_timing_only")
            self.assertEqual(meta["accuracy_status"], "unchecked")
            self.assertEqual(meta["settings"]["warmups_per_repeat_count"], 1)
            self.assertEqual(meta["settings"]["input_pattern"], "constant_0.125_minus0.25")
            self.assertEqual(meta["coverage"]["gpu_validation"], "not_run_timing_only")
            with (output / "table.csv").open() as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["timing_status"], "collected_unvalidated")
            self.assertEqual(row["utilization_status"], "calculated_unvalidated")
            self.assertEqual(row["max_transform_relative_l2"], "")
            self.assertEqual(row["radix16_speedup"], "")
            result = assembly.assemble([output], root / "assembled", "smoke")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["collected"]["timing"], ["register_uniform"])
            self.assertEqual(result["accuracy_status"], "unchecked")
            self.assertEqual(result["ignored_parts"], [])
            self.assertTrue(result["unit_kernel_comparison"]["available"])
            self.assertEqual((output / "kernel_comparison.csv").read_text(),
                             (root / "assembled" / "kernel_comparison.csv").read_text())

    def test_collection_defaults_available_variants_without_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", THUNDERFFT_COLLECTION="fixture")
            env.pop("THUNDERFFT_TABLE7_VARIANTS", None)
            env.pop("THUNDERFFT_TABLE7_PHASE", None)
            env.pop("THUNDERFFT_TABLE7_NCU", None)
            env.pop("THUNDERFFT_TABLE7_TIMING_ONLY", None)
            result = subprocess.run([sys.executable, str(HERE / "measure.py"), "--check-selection", "--output", directory], env=env,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            coverage = json.loads((Path(directory) / "coverage.json").read_text())
            self.assertEqual(coverage["requested"], "register_branch,register_uniform,shared_stage_radix4,historical_e4aa1b0,historical_652808b")
            self.assertEqual(coverage["source_missing"], ["radix16", "radix4_compute"])

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
                   "THUNDERFFT_TABLE7_TIMING_ONLY": "0",
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
            self.assertEqual(row["radix16_speedup"], "")
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertEqual(metadata["phase"], "timing")
            self.assertEqual(json.loads((output / "coverage.json").read_text()), metadata["coverage"])
            self.assertEqual(metadata["coverage"]["measured_variants"], ["register_uniform"])

    def test_full_measurement_exports_speedup_and_refreshed_coverage(self):
        # Native output fixtures only: this test never launches a GPU workload.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary, build, output = root / "fake-binary", root / "build", root / "output"
            binary.write_text("CPU test fixture")
            build.mkdir()
            times = {"radix16": 8, "radix4_compute": 12, "register_branch": 7, "register_uniform": 4}
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                variant = command[1]
                self.assertEqual(command, [str(binary), variant, "64", "128", "1"])
                stream = io.StringIO()
                writer = csv.DictWriter(stream, fieldnames=measure.RAW_FIELDS)
                writer.writeheader()
                writer.writerows(fixture_rows(variant, times[variant]))
                kwargs["stdout"].write(stream.getvalue())
                kwargs["stderr"].write(GPU)
                return subprocess.CompletedProcess(command, 0)
            env = {"THUNDERFFT_TABLE7_TIMING_ONLY": "0", "THUNDERFFT_TABLE7_BATCH": "64", "THUNDERFFT_TABLE7_REPEATS": "128", "THUNDERFFT_TABLE7_TRIALS": "1"}
            argv = ["measure.py", "--output", str(output), "--binary", str(binary), "--build-dir", str(build)]
            with future_recovered_variants(), patch.dict(os.environ, env, clear=True), patch.object(sys, "argv", argv), \
                    patch.object(measure, "capture", return_value={}), \
                    patch.object(measure.subprocess, "run", side_effect=run), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(measure.main(), 0)
            self.assertEqual([command[1] for command in calls], list(times))
            metadata = json.loads((output / "metadata.json").read_text())
            coverage = json.loads((output / "coverage.json").read_text())
            self.assertEqual(coverage, metadata["coverage"])
            self.assertEqual(coverage["measured_variants"], list(times))
            self.assertIn("passed", coverage["gpu_validation"])
            self.assertIn("ablation/kernels/radix16.cuh", metadata["source_sha256"])
            self.assertIn("ablation/kernels/radix4_shared.cuh", metadata["source_sha256"])
            with (output / "table.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            for row in rows:
                self.assertAlmostEqual(float(row["radix16_speedup"]), 8 / times[row["variant"]])
            with future_recovered_variants():
                result = assembly.assemble([output], root / "assembled", "smoke")
            self.assertEqual(result["status"], "complete")
            with (root / "assembled" / "table.csv").open() as stream:
                assembled_rows = list(csv.DictReader(stream))
            self.assertEqual([{field: row[field] for field in measure.TABLE_FIELDS}
                              for row in assembled_rows], rows)
            self.assertTrue(all(row[field] == "" for row in assembled_rows for field in assembly.ESTIMATE_FIELDS))


if __name__ == "__main__":
    unittest.main()
