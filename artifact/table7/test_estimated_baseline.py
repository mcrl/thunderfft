"""Exercise Table 6 evidence reuse and explicitly estimated Table 7 cells."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("table7_projection_fixture", HERE / "test_assemble.py")
fixture = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fixture)
assembly = fixture.assembly


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class EstimatedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.table6 = self.root / "table6"
        self.table6.mkdir()
        self.parts = []
        for variant, timing in (("register_branch_cached", 7.9), ("register_uniform", 4)):
            part = fixture.fixture_part(self.root, variant, variant, time_us=timing,
                                        timing_only=variant == "register_branch_cached")
            meta = json.loads((part / "metadata.json").read_text())
            meta["profile"] = "paper"
            meta["settings"].update(batch=65536, R=1000)
            (part / "metadata.json").write_text(json.dumps(meta))
            for name in ("data.csv", f"{variant}.csv"):
                rows = read_rows(part / name)
                for row in rows:
                    row.update(batch=65536, inside_repeats=1000, t_2R_us=100 + 1000 * timing)
                write_rows(part / name, rows)
            self.parts.append(part)
        self.make_table6()

    def make_table6(self, time_ms=.021763):
        # Synthetic CPU fixture with the same independently checked source
        # contract as a real Table 6 assembly; no GPU measurements performed.
        rows = [{"gpu": "A100", "N": 256, "batch": 65536, "library": "tcFFT",
                 "variant": "layer_256_0_A100", "selected_candidate": "layer_256_0_A100",
                 "status": "collected", "median_time_ms": time_ms},
                {"gpu": "A100", "N": 256, "library": "FlashFFTConv", "status": "pending"}]
        write_rows(self.table6 / "table.csv", rows)
        write_rows(self.table6 / "data.csv", [{"gpu": "A100", "N": 256, "batch": 65536,
                   "library": "tcFFT", "variant": "layer_256_0_A100", "candidate": "layer_256_0_A100",
                   "precision": "fp16", "input": "complex", "passed": True, "time_ms": time_ms}])
        self.hash_metadata()

    def hash_metadata(self):
        (self.table6 / "metadata.json").write_text(json.dumps({"evaluation": "table6", "profile": "paper",
            "assembly_schema": "table6-validated-parts-v1", "outputs": {
                name: hashlib.sha256((self.table6 / name).read_bytes()).hexdigest() for name in ("table.csv", "data.csv")}}))

    def test_tcfft_only_evidence_fills_estimate_and_preserves_measured_exports(self):
        original, estimated = self.root / "original", self.root / "estimated"
        before = {str(p): p.read_bytes() for part in self.parts + [self.table6] for p in part.rglob("*") if p.is_file()}
        assembly.assemble(self.parts, original, "paper")
        state = assembly.assemble(self.parts, estimated, "paper", table6=self.table6)
        rows = {row["variant"]: row for row in read_rows(estimated / "table.csv")}
        baseline = rows["radix16"]
        self.assertEqual(baseline["timing_status"], "estimated")
        self.assertEqual(baseline["accuracy_status"], "not_applicable_projection")
        self.assertAlmostEqual(float(baseline["estimated_time_ms"]), .008161125)
        self.assertAlmostEqual(float(baseline["estimated_utilization_pct"]), 63.25378312049267)
        self.assertEqual(float(baseline["speedup_vs_estimated_radix16"]), 1)
        for field in ("median_time_ms", "min_time_ms", "max_time_ms", "utilization_pct", "trials", "radix16_speedup"):
            self.assertEqual(baseline[field], "")
        self.assertAlmostEqual(float(rows["register_branch"]["speedup_vs_estimated_radix16"]), .008161125 / .0079)
        self.assertEqual(rows["register_branch"]["accuracy_status"], "unchecked")
        self.assertEqual(rows["register_branch"]["timing_status"], "collected_unvalidated")
        self.assertEqual(rows["register_branch"]["source_variant"], "register_branch_cached")
        self.assertAlmostEqual(float(rows["register_uniform"]["speedup_vs_estimated_radix16"]), .008161125 / .004)
        self.assertEqual(rows["register_uniform"]["radix16_speedup"], "")
        self.assertEqual(rows["radix4_compute"]["estimated_time_ms"], "")
        self.assertEqual({item["item"] for item in state["missing"]}, {"radix4_compute.timing", "radix4_compute.utilization"})
        self.assertFalse(state["full_paper_comparison"])
        for name in ("data.csv", "kernel_comparison.csv"):
            self.assertEqual((original / name).read_bytes(), (estimated / name).read_bytes())
        for path, data in before.items():
            self.assertEqual(Path(path).read_bytes(), data)
        for record in state["baseline_projection"]["inputs"]:
            self.assertFalse(Path(record["path"]).is_absolute())
            self.assertEqual(hashlib.sha256((estimated / record["path"]).read_bytes()).hexdigest(), record["sha256"])

    def test_projection_follows_source_timing_instead_of_paper_utilization(self):
        self.make_table6(.030)
        state = assembly.assemble(self.parts, self.root / "new", "paper", table6=self.table6)
        estimate = state["baseline_projection"]["estimate"]
        self.assertAlmostEqual(estimate["projected_time_us"], 11.25)
        self.assertAlmostEqual(estimate["projected_utilization_pct"], estimate["source_utilization_pct"])
        self.assertNotAlmostEqual(estimate["projected_utilization_pct"], 70)

    def test_missing_tcfft_never_falls_back_to_documented_or_paper_time(self):
        rows = read_rows(self.table6 / "table.csv")
        rows[0].update(status="pending", median_time_ms="")
        write_rows(self.table6 / "table.csv", rows)
        self.hash_metadata()
        state = assembly.assemble(self.parts, self.root / "missing", "paper", table6=self.table6)
        self.assertEqual(state["baseline_projection"]["status"], "unavailable")
        self.assertEqual(len(state["missing"]), 4)
        self.assertEqual(read_rows(self.root / "missing/table.csv")[0]["estimated_time_ms"], "")

    def test_changed_source_hash_or_mismatched_raw_timing_is_rejected(self):
        rows = read_rows(self.table6 / "data.csv")
        rows[0]["time_ms"] = .050
        write_rows(self.table6 / "data.csv", rows)
        with self.assertRaisesRegex(ValueError, "input hash"):
            assembly.assemble(self.parts, self.root / "tampered", "paper", table6=self.table6)
        self.hash_metadata()
        with self.assertRaisesRegex(ValueError, "disagrees with data.csv"):
            assembly.assemble(self.parts, self.root / "mismatch", "paper", table6=self.table6)

    def test_incompatible_source_workload_or_failed_trials_are_rejected(self):
        for key, value in (("N", 128), ("batch", 1024), ("precision", "tf32"), ("gpu", "H100"), ("passed", False)):
            with self.subTest(key=key):
                self.make_table6()
                rows = read_rows(self.table6 / "data.csv")
                rows[0][key] = value
                write_rows(self.table6 / "data.csv", rows)
                self.hash_metadata()
                with self.assertRaisesRegex(ValueError, "data.csv must confirm"):
                    assembly.assemble(self.parts, self.root / key, "paper", table6=self.table6)


if __name__ == "__main__":
    unittest.main()
