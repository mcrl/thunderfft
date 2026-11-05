#!/usr/bin/env python3
"""CPU-only contracts for collecting and combining independent Table 5 phases."""

import copy
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

_spec = importlib.util.spec_from_file_location("table5_assemble_test_target", Path(__file__).with_name("assemble.py"))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
assemble = _module.assemble
summary = _module._summary

ENVIRONMENT = {"gpu_uuid": "GPU-12345678-1234-5678-1234-123456789abc", "gpu_name": "synthetic GPU", "local_cuda_device": 0, "pci_bus_id": "synthetic bus",
               "compute_capability": "8.0", "total_global_memory_bytes": 1234,
               "cuda_driver_api_version": 12090, "cuda_runtime_version": 12080,
               "cuda_compile_version": 12090, "cufft_version": 11303}


def make_part(path, phases=None, sizes=None, profile="smoke", batch=64, seed=1234):
    phases, sizes = list(summary.PHASES if phases is None else phases), sizes or [64]
    path.mkdir()
    ranges = ({"fp16": [8, 13], "tf32": [120, 127], "fp32": [120, 127]} if profile == "smoke"
              else {p: [0, 127] for p in summary.RANGE_PRECISIONS})
    metadata = {
        "evaluation": "table5", "profile": profile, "N": sizes, "batch": batch, "seed": seed,
        "selected_phases": phases, "range_exponents": ranges,
        "input": "complex Gaussian; per-FFT L2 norm / sqrt(N) = alpha",
        "reference": "cuFFT Z2Z (FP64) applied to the generated FP32 input",
        "implementations": dict(summary.IMPLEMENTATIONS),
        "source": {"sha256": {"main.cu": "a" * 64}},
        "build": {"binary_sha256": "b" * 64, "cmake_cache": {"CMAKE_CUDA_ARCHITECTURES": "80"},
                  "cuda_compiler_version": {"exit_code": 0, "stdout": "synthetic compiler"}},
        "measured_environment": copy.deepcopy(ENVIRONMENT),
    }
    if "accuracy_fp16_cufftdx" in phases:
        metadata["build"]["cufftdx"] = {"binary_sha256": "d" * 64, "target_files": {"flags.make": "-O3"}}
    (path / "metadata.json").write_text(json.dumps(metadata))
    (path / "status.json").write_text(json.dumps({"evaluation": "table5", "profile": profile,
                                                  "status": "completed", "exit_code": 0}))
    for phase in phases:
        rows = []
        precisions = (sorted(summary.ACCURACY_PRECISIONS[phase]) if phase in summary.ACCURACY_PRECISIONS
                      else [next(p for p, name in summary.RANGE_PHASE.items() if name == phase)])
        for n in sizes:
            for precision in precisions:
                accuracy = phase.startswith("accuracy_")
                alphas = [1.0] if accuracy else [2.0**exponent for exponent in range(ranges[precision][0], ranges[precision][1] + 1)]
                for alpha in alphas:
                    error = 1e-7 if precision == "fp32" else 1e-3 if precision == "fp16_cufftdx" else 1e-4
                    values = [n, batch, "gaussian", alpha, precision,
                              error if accuracy else "nan", error if accuracy else "nan", error if accuracy else "nan",
                              int(accuracy or alpha <= 2.0**(11 if precision == "fp16" else 123))]
                    rows.append(dict(zip(summary.RAW_FIELDS, values)))
        summary.write_csv(path / f"{phase}.csv", summary.RAW_FIELDS, rows)
        (path / f"{phase}.environment.json").write_text(json.dumps(ENVIRONMENT))
    return path


def mutate_json(path, function):
    value = json.loads(path.read_text())
    function(value)
    path.write_text(json.dumps(value))


class AssembleChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="thunderfft-table5-assemble-")
        self.root = Path(self.temp.name)
        self.output = self.root / "output"

    def tearDown(self):
        self.temp.cleanup()

    def read_table(self):
        with (self.output / "table.csv").open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_empty_collection_is_pending(self):
        result = assemble([], self.output, "paper")
        self.assertEqual(result["status"], "pending")
        self.assertEqual(len(result["missing"]), 24)
        self.assertEqual(len(self.read_table()), 16)
        self.assertTrue(all(row["largest_finite_exponent_observed"] == "" and
                            row["range_status"] == ("not_measured" if row["precision"] == "fp16_cufftdx" else "pending")
                            for row in self.read_table()))

    def test_selected_accuracy_collects_successfully_with_pending_ranges(self):
        part = make_part(self.root / "accuracy", ["accuracy_reduced"])
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("summarize.py")), str(part)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        result = assemble([part], self.output, "smoke")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing"]), 5)
        self.assertEqual([row["accuracy_status"] for row in self.read_table()], ["complete", "complete", "pending", "pending"])
        self.assertTrue(all(row["largest_finite_exponent_observed"] == "" for row in self.read_table()))

    def test_six_independent_phases_complete_without_changing_parts(self):
        parts = [make_part(self.root / phase, [phase]) for phase in summary.PHASES]
        original = {str(path): path.read_bytes() for part in parts for path in part.iterdir()}
        result = assemble(parts, self.output, "smoke")
        self.assertEqual(result, {"status": "complete", "missing": [], "full_paper_comparison": False})
        self.assertEqual([row["largest_finite_exponent_observed"] for row in self.read_table()], ["11", "123", "123", ""])
        self.assertEqual(self.read_table()[3]["range_status"], "not_measured")
        self.assertEqual(original, {str(path): path.read_bytes() for part in parts for path in part.iterdir()})

    def test_split_paper_sizes_complete(self):
        parts = [make_part(self.root / f"n{n}", sizes=[n], profile="paper") for n in [64, 256, 1024, 4096]]
        self.assertEqual(assemble(parts, self.output, "paper")["status"], "complete")
        self.assertEqual(len(self.read_table()), 16)

    def test_one_paper_size_leaves_other_sizes_pending(self):
        part = make_part(self.root / "one", profile="paper")
        result = assemble([part], self.output, "paper")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(result["missing"]), 18)

    def test_extra_sizes_must_be_complete(self):
        base = make_part(self.root / "base")
        extra = make_part(self.root / "extra", ["accuracy_reduced"], sizes=[64, 2048])
        result = assemble([base, extra], self.output, "smoke")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(sorted(item["item"] for item in result["missing"]),
                         sorted(f"{phase}:N=2048" for phase in summary.PHASES if phase != "accuracy_reduced"))
        full = make_part(self.root / "full", sizes=[64, 2048])
        self.assertEqual(assemble([base, full], self.output, "smoke")["status"], "complete")
        self.assertEqual(len(self.read_table()), 8)

    def test_identical_duplicate_is_deduplicated(self):
        parts = [make_part(self.root / name) for name in ["first", "second"]]
        result = assemble(parts, self.output, "smoke")
        self.assertEqual(result["status"], "complete")
        with (self.output / "data.csv").open(newline="") as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 26)

    def test_conflicting_duplicate_rejected(self):
        parts = [make_part(self.root / name) for name in ["first", "second"]]
        path = parts[1] / "accuracy_reduced.csv"
        rows = summary.load_rows(path)
        rows[0]["rel_l2"] = "0.0002"
        summary.write_csv(path, summary.RAW_FIELDS, rows)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            assemble(parts, self.output, "smoke")

    def test_incomplete_range_rejected(self):
        part = make_part(self.root / "part", ["range_fp16"])
        path = part / "range_fp16.csv"
        summary.write_csv(path, summary.RAW_FIELDS, summary.load_rows(path)[:-1])
        with self.assertRaisesRegex(ValueError, "missing, duplicate, or out-of-range"):
            assemble([part], self.output, "smoke")
        self.assertFalse((self.output / "table.csv").exists())

    def test_incomplete_range_summary_never_reports_boundary(self):
        part = make_part(self.root / "part", ["range_fp16"])
        path = part / "range_fp16.csv"
        summary.write_csv(path, summary.RAW_FIELDS, summary.load_rows(path)[:-1])
        metadata = json.loads((part / "metadata.json").read_text())
        checked = summary.validate_bundle(part, metadata)
        self.assertTrue(checked["errors"])
        self.assertEqual(checked["ranges"], {})

    def set_finite(self, part, phase, finite):
        path = part / f"{phase}.csv"
        rows = summary.load_rows(path)
        for row in rows:
            row["all_finite"] = str(int(finite(round(math.log2(float(row["alpha"]))))))
        summary.write_csv(path, summary.RAW_FIELDS, rows)

    def read_csv(self, name):
        with (self.output / name).open(newline="") as stream:
            return list(csv.DictReader(stream))

    def test_scan_boundary_status(self):
        for name, finite, expected in (("observed", None, ("observed", "11", "12")),
                                       ("right", lambda k: True, ("right_censored", "13", "")),
                                       ("left", lambda k: False, ("left_censored", "", "8"))):
            with self.subTest(name):
                part = make_part(self.root / name, ["range_fp16"])
                if finite:
                    self.set_finite(part, "range_fp16", finite)
                assemble([part], self.output, "smoke")
                fp16 = next(row for row in self.read_table() if row["precision"] == "fp16")
                self.assertEqual((fp16["boundary_status"], fp16["largest_finite_exponent_observed"],
                                  fp16["first_nonfinite_exponent_observed"]), expected)
                comparison = next(row for row in self.read_csv("comparison.csv") if row["precision"] == "fp16")
                self.assertEqual(comparison["finite_exponent_delta"] != "", name == "observed")

    def test_cufftdx_part_combines_with_build_without_mathdx(self):
        parts = [make_part(self.root / "core", list(summary.DEFAULT_PHASES)),
                 make_part(self.root / "dx", ["accuracy_fp16_cufftdx"])]
        self.assertEqual([item["item"] for item in assemble(parts[:1], self.output, "smoke")["missing"]],
                         ["accuracy_fp16_cufftdx:N=64"])
        self.assertEqual(assemble(parts, self.output, "smoke")["status"], "complete")

    def test_different_cufftdx_builds_rejected(self):
        parts = [make_part(self.root / name, ["accuracy_fp16_cufftdx"], sizes=[n], profile="paper")
                 for name, n in (("first", 64), ("second", 256))]
        mutate_json(parts[1] / "metadata.json", lambda m: m["build"]["cufftdx"].update(binary_sha256="e" * 64))
        with self.assertRaisesRegex(ValueError, "incompatible cuFFTDx"):
            assemble(parts, self.output, "paper")

    def test_missing_selected_phase_rejected(self):
        part = make_part(self.root / "part")
        (part / "range_fp16.csv").unlink()
        with self.assertRaisesRegex(ValueError, "phase validation failed"):
            assemble([part], self.output, "smoke")

    def test_different_population_rejected(self):
        parts = [make_part(self.root / "first"), make_part(self.root / "second", batch=128)]
        with self.assertRaisesRegex(ValueError, "incompatible.*batch"):
            assemble(parts, self.output, "smoke")

    def test_different_seed_rejected(self):
        parts = [make_part(self.root / "first"), make_part(self.root / "second", seed=99)]
        with self.assertRaisesRegex(ValueError, "incompatible.*seed"):
            assemble(parts, self.output, "smoke")

    def test_changed_source_or_binary_rejected(self):
        for change, label in ((lambda m: m["source"]["sha256"].update({"main.cu": "c" * 64}), "source_sha256"),
                              (lambda m: m["build"].update(binary_sha256="c" * 64), "binary_sha256")):
            with self.subTest(label=label):
                parts = [make_part(self.root / f"first-{label}"), make_part(self.root / f"second-{label}")]
                mutate_json(parts[1] / "metadata.json", change)
                with self.assertRaisesRegex(ValueError, f"incompatible.*{label}"):
                    assemble(parts, self.output, "smoke")

    def test_changed_gpu_rejected(self):
        parts = [make_part(self.root / name) for name in ["first", "second"]]
        mutate_json(parts[1] / "metadata.json", lambda m: m["measured_environment"].update(gpu_name="different"))
        for phase in summary.PHASES:
            mutate_json(parts[1] / f"{phase}.environment.json", lambda e: e.update(gpu_name="different"))
        with self.assertRaisesRegex(ValueError, "incompatible.*measured_environment"):
            assemble(parts, self.output, "smoke")

    def remove_uuid(self, part):
        mutate_json(part / "metadata.json", lambda m: m["measured_environment"].pop("gpu_uuid"))
        for path in part.glob("*.environment.json"):
            mutate_json(path, lambda e: e.pop("gpu_uuid"))

    def test_changed_gpu_uuid_rejected(self):
        parts = [make_part(self.root / name) for name in ["first", "second"]]
        different = "GPU-22345678-1234-5678-1234-123456789abc"
        mutate_json(parts[1] / "metadata.json", lambda m: m["measured_environment"].update(gpu_uuid=different))
        for path in parts[1].glob("*.environment.json"):
            mutate_json(path, lambda e: e.update(gpu_uuid=different))
        with self.assertRaisesRegex(ValueError, "incompatible.*measured_environment"):
            assemble(parts, self.output, "smoke")

    def test_single_legacy_bundle_without_uuid_accepted(self):
        part = make_part(self.root / "legacy")
        self.remove_uuid(part)
        self.assertEqual(assemble([part], self.output, "smoke")["status"], "complete")

    def test_identical_legacy_duplicates_without_uuid_accepted(self):
        parts = [make_part(self.root / name) for name in ["first", "second"]]
        for part in parts:
            self.remove_uuid(part)
        self.assertEqual(assemble(parts, self.output, "smoke")["status"], "complete")

    def test_split_legacy_bundles_without_uuid_rejected(self):
        parts = [make_part(self.root / "first", ["accuracy_reduced", "accuracy_fp32_cufft"]),
                 make_part(self.root / "second", ["range_fp16", "range_tf32", "range_fp32_cufft"])]
        for part in parts:
            self.remove_uuid(part)
        with self.assertRaisesRegex(ValueError, "GPU UUID is required"):
            assemble(parts, self.output, "smoke")

    def test_invalid_or_zero_uuid_rejected(self):
        for index, invalid in enumerate(["", "not-a-UUID", "GPU-00000000-0000-0000-0000-000000000000"]):
            with self.subTest(uuid=invalid):
                part = make_part(self.root / f"invalid-{index}")
                mutate_json(part / "metadata.json", lambda m: m["measured_environment"].update(gpu_uuid=invalid))
                for path in part.glob("*.environment.json"):
                    mutate_json(path, lambda e: e.update(gpu_uuid=invalid))
                with self.assertRaisesRegex(ValueError, "valid nonzero CUDA device UUID"):
                    assemble([part], self.output, "smoke")

    def test_missing_provenance_rejected(self):
        part = make_part(self.root / "part")
        mutate_json(part / "metadata.json", lambda m: m.pop("source"))
        with self.assertRaisesRegex(ValueError, "missing source/build"):
            assemble([part], self.output, "smoke")

    def test_profile_mismatch_rejected(self):
        part = make_part(self.root / "part")
        with self.assertRaisesRegex(ValueError, "profile"):
            assemble([part], self.output, "paper")

    def test_failed_part_rejected(self):
        part = make_part(self.root / "part")
        mutate_json(part / "status.json", lambda s: s.update(status="failed", exit_code=1))
        with self.assertRaisesRegex(ValueError, "successfully collected"):
            assemble([part], self.output, "smoke")

    def test_unknown_phase_rejected(self):
        part = make_part(self.root / "part")
        mutate_json(part / "metadata.json", lambda m: m.update(selected_phases=["unknown"]))
        with self.assertRaisesRegex(ValueError, "selected_phases"):
            assemble([part], self.output, "smoke")


if __name__ == "__main__":
    unittest.main()
