"""CPU checks for optional tcFFT evidence reuse and honest pipeline coverage."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import collection, full_pipeline, paper_tables, runner


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


T6_MEASURE = module("table7_integration_table6_measure_fixture", ROOT / "artifact/table6/test_measure.py")
with patch.dict(sys.modules, {"test_measure": T6_MEASURE}):
    T6 = module("table7_integration_table6_assembly_fixture", ROOT / "artifact/table6/test_assemble.py")


class Table7Integration(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "collection"
        self.calls = []
        self.adapter = collection.load_adapter("table6")
        self.registry = json.loads((ROOT / "artifact/registry.json").read_text())["evaluations"]
        self.entries = {entry["id"]: entry for entry in self.registry}
        self.addCleanup(patch.stopall)
        patch.object(paper_tables, "render", return_value={}).start()
        patch.object(collection, "load_adapter", side_effect=self.load_adapter).start()

    def load_adapter(self, evaluation):
        if evaluation == "table6":
            return self.adapter
        self.assertEqual(evaluation, "table7")
        return SimpleNamespace(assemble=self.assemble_table7)

    def assemble_table7(self, sources, output, profile, table6=None):
        self.calls.append(table6)
        if table6 is not None:
            metadata = json.loads((table6 / "metadata.json").read_text())
            self.assertEqual(metadata["assembly_schema"], "table6-validated-parts-v1")
            self.assertEqual(metadata["profile"], profile)
        (output / "data.csv").write_text("variant,time\n")
        return {"status": "partial", "missing": [{"item": "radix4_compute.timing",
                "source_variant": "shared_stage_radix4", "reason": "fixture did not collect the selected source"}],
                "full_paper_comparison": False}

    def collect_tcfft(self):
        source = T6.bundle(self.root / "tcfft", "tcfft")
        (source / "status.json").write_text(json.dumps({
            "evaluation": "table6", "profile": "smoke", "status": "completed", "exit_code": 0}))
        return collection.add_part(self.directory, "table6", "tcfft", source, "smoke")

    def test_table7_only_without_table6_does_not_create_or_schedule_it(self):
        selected = runner.select_tasks(["table7"], self.registry, argparse.ArgumentParser())
        self.assertEqual([entry["id"] for entry in selected], ["table7"])
        states = collection.assemble_collection(self.directory, selected, "smoke")
        self.assertEqual(self.calls, [None])
        self.assertEqual([state["evaluation"] for state in states], ["table7"])
        self.assertFalse((self.directory / "assembled/table6").exists())

    def test_table7_only_revalidates_existing_table6_parts_without_measuring(self):
        record = self.collect_tcfft()
        before = collection.file_hashes(self.directory / "parts")
        with patch.object(runner, "execute", side_effect=AssertionError("GPU work requested")):
            states = collection.assemble_collection(self.directory, [self.entries["table7"]], "smoke")
        self.assertEqual([state["evaluation"] for state in states], ["table7"])
        self.assertEqual(self.calls[0].parts[-3:], ("table7", "inputs", "table6"))
        self.assertTrue((self.directory / "assembled/table7/inputs/table6/table.csv").is_file())
        self.assertFalse((self.directory / "assembled/table6").exists())
        self.assertEqual(collection.file_hashes(self.directory / "parts"), before)
        metadata = json.loads((self.directory / "assembled/table7/inputs/table6/metadata.json").read_text())
        self.assertEqual(metadata["sources"][0]["path"], str(self.directory / record["path"]))

    def test_selected_table6_orders_before_table7_and_is_reused(self):
        self.collect_tcfft()
        selected = runner.select_tasks(["table7", "table6"], self.registry, argparse.ArgumentParser())
        self.assertEqual([entry["id"] for entry in selected], ["table6", "table7"])
        collection.assemble_collection(self.directory, selected, "smoke")
        self.assertEqual(self.calls[0].name, "table6")
        self.assertNotEqual(self.calls[0].parent.name, "inputs")
        self.assertFalse((self.directory / "assembled/table7/inputs/table6").exists())

    def test_later_table6_assembly_preserves_and_refreshes_existing_table7(self):
        collection.assemble_collection(self.directory, [self.entries["table7"]], "smoke")
        self.collect_tcfft()
        states = collection.assemble_collection(self.directory, [self.entries["table6"]], "smoke")
        self.assertEqual([state["evaluation"] for state in states], ["table6", "table7"])
        self.assertIsNone(self.calls[0])
        self.assertEqual(self.calls[1].name, "table6")
        self.assertNotEqual(self.calls[1].parent.name, "inputs")

    def test_invalid_optional_evidence_retains_prior_assembly(self):
        collection.assemble_collection(self.directory, [self.entries["table7"]], "smoke")
        before = collection.file_hashes(self.directory / "assembled")
        self.collect_tcfft()
        with patch.object(self.adapter, "assemble", side_effect=ValueError("invalid raw evidence")), \
                self.assertRaisesRegex(ValueError, "invalid raw evidence"):
            collection.assemble_collection(self.directory, [self.entries["table7"]], "smoke")
        self.assertEqual(collection.file_hashes(self.directory / "assembled"), before)


class Table7Coverage(unittest.TestCase):
    def check(self, projection, *, full_paper_comparison=False, missing_shared_stage=False):
        # All available selected stages are collected by default. Only an
        # unavailable projection leaves the unrecovered radix-16 row empty.
        missing = ([] if projection.get("status") == "estimated" else [
            {"item": "radix16.timing", "source_variant": "radix16", "reason": "source unavailable"}])
        if missing_shared_stage:
            missing.append({"item": "radix4_compute.timing", "source_variant": "shared_stage_radix4",
                            "reason": "selected reconstruction has not been measured"})
        state = {"evaluation": "table7", "status": "partial" if missing else "complete",
                 "full_paper_comparison": full_paper_comparison, "missing": missing,
                 "baseline_projection": projection}
        return full_pipeline.assembly_result([state], Path("/no-files-required"),
            [{"id": "table7", "required_outputs": {"paper": []}}], "paper", ROOT)

    def test_estimate_remains_an_explicit_coverage_limit(self):
        projection = {"status": "estimated", "assumption": "retained tcFFT utilization",
                      "formula": "time = tcFFT_time * 0.375"}
        result = self.check(projection)
        self.assertTrue(result["outputs_complete"])
        self.assertFalse(result["full_paper_comparison"])
        estimate = next(item for item in result["coverage_limits"] if item["item"] == "radix16.projection")
        self.assertEqual(estimate["status"], "estimated")
        self.assertEqual(estimate["assumption"], projection["assumption"])
        self.assertEqual(estimate["formula"], projection["formula"])

    def test_estimate_without_assumption_or_claimed_full_measurement_fails(self):
        projection = {"status": "estimated", "assumption": "retained tcFFT utilization",
                      "formula": "time = tcFFT_time * 0.375"}
        self.assertFalse(self.check(projection, full_paper_comparison=True)["outputs_complete"])
        self.assertFalse(self.check(dict(projection, assumption=None))["outputs_complete"])
        self.assertFalse(self.check(dict(projection, status="measured"))["outputs_complete"])

    def test_unavailable_projection_preserves_reason_as_a_limit(self):
        result = self.check({"status": "unavailable", "reason": "no compatible tcFFT evidence"})
        self.assertTrue(result["outputs_complete"])
        self.assertTrue(any(item.get("status") == "unavailable" and
                            item["reason"] == "no compatible tcFFT evidence"
                            for item in result["coverage_limits"]))

    def test_valid_baseline_estimate_does_not_hide_missing_selected_reconstruction(self):
        projection = {"status": "estimated", "assumption": "retained tcFFT utilization",
                      "formula": "time = tcFFT_time * 0.375"}
        result = self.check(projection, missing_shared_stage=True)
        self.assertFalse(result["outputs_complete"])
        self.assertEqual([item["item"] for item in result["coverage_limits"]], ["radix16.projection"])
        self.assertEqual([item["item"] for item in result["unresolved"]], ["radix4_compute.timing"])


if __name__ == "__main__":
    unittest.main()
