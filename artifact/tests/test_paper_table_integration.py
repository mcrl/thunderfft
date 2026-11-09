"""CPU-only checks that visual tables publish with their source collection."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact import collection, full_pipeline


TABLES = tuple(f"table{number}" for number in range(2, 8))
PRESENTATIONS = ("table.pdf", "table.png", "table.html", "table-render.json")


class PaperTableIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "collection"

    @staticmethod
    def assemble_stub(sources, output, profile):
        (output / "data.csv").write_text("value\n1\n")
        (output / "table.csv").write_text("value\n1\n")
        return {"status": "partial", "full_paper_comparison": False,
                "missing": [{"item": "example", "reason": "not measured"}]}

    def render_stub(self, evaluation, output):
        self.assertTrue((output / "table.csv").is_file())
        self.assertFalse((output / "status.json").exists())
        for filename in PRESENTATIONS:
            (output / filename).write_text("rendered " + evaluation)
        return {"schema_version": 1, "outputs": {name: "fixture hash" for name in PRESENTATIONS[:3]}}

    def test_every_table_renders_with_coverage_status_preserved(self):
        adapter = SimpleNamespace(assemble=self.assemble_stub)
        with mock.patch.object(collection, "load_adapter", return_value=adapter), \
                mock.patch("artifact.paper_tables.render", side_effect=self.render_stub) as render:
            states = collection.assemble_collection(self.directory, [{"id": name} for name in TABLES], "paper")
        self.assertEqual([call.args[0] for call in render.call_args_list], list(TABLES))
        for state in states:
            self.assertEqual(state["status"], "partial")
            self.assertFalse(state["full_paper_comparison"])
            self.assertEqual(state["missing"], [{"item": "example", "reason": "not measured"}])
            output = self.directory / "assembled" / state["evaluation"]
            self.assertEqual(json.loads((output / "status.json").read_text())["presentation"], state["presentation"])
            self.assertTrue(all((output / filename).is_file() for filename in PRESENTATIONS))
        manifest = json.loads((self.directory / "assembled/manifest.json").read_text())
        self.assertFalse(manifest["new_gpu_measurements"])
        self.assertFalse(manifest["complete"])

    def test_render_failure_retains_previous_csvs_presentations_and_status(self):
        adapter = SimpleNamespace(assemble=self.assemble_stub)
        with mock.patch.object(collection, "load_adapter", return_value=adapter):
            with mock.patch("artifact.paper_tables.render", side_effect=self.render_stub):
                collection.assemble_collection(self.directory, [{"id": "table2"}], "paper")
            previous = collection.file_hashes(self.directory / "assembled")
            with mock.patch("artifact.paper_tables.render", side_effect=RuntimeError("renderer failed")), \
                    self.assertRaisesRegex(RuntimeError, "renderer failed"):
                collection.assemble_collection(self.directory, [{"id": "table2"}], "paper")
        self.assertEqual(collection.file_hashes(self.directory / "assembled"), previous)
        self.assertFalse(list(self.directory.glob(".assemble-*")))
        error = json.loads((self.directory / "assembly-error.json").read_text())
        self.assertTrue(error["previous_assembly_retained"])
        self.assertIn("renderer failed", error["error"])

    def test_non_table_evaluation_does_not_invoke_table_renderer(self):
        adapter = SimpleNamespace(assemble=self.assemble_stub)
        with mock.patch.object(collection, "load_adapter", return_value=adapter), \
                mock.patch("artifact.paper_tables.render", side_effect=AssertionError("table renderer used")):
            states = collection.assemble_collection(self.directory, [{"id": "validation"}], "paper")
        self.assertNotIn("presentation", states[0])

    def test_full_workflow_rejects_missing_or_empty_declared_presentations(self):
        output = self.directory / "assembled"
        table = output / "table2"
        table.mkdir(parents=True)
        (table / "data.csv").write_text("value\n1\n")
        entries = [{"id": "table2", "required_outputs": {"paper": ["data.csv"]}}]
        states = [{"evaluation": "table2", "status": "complete", "missing": [], "presentation": {}}]
        result = full_pipeline.assembly_result(states, output, entries, "paper", ROOT)
        self.assertFalse(result["outputs_complete"])
        self.assertEqual(set(result["missing_files"]), {f"table2/{name}" for name in PRESENTATIONS})
        for filename in PRESENTATIONS:
            (table / filename).write_text("present")
        self.assertTrue(full_pipeline.assembly_result(states, output, entries, "paper", ROOT)["outputs_complete"])
        (table / "table.pdf").write_text("")
        result = full_pipeline.assembly_result(states, output, entries, "paper", ROOT)
        self.assertFalse(result["outputs_complete"])
        self.assertEqual(result["missing_files"], ["table2/table.pdf"])


if __name__ == "__main__":
    unittest.main()
