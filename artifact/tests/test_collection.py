#!/usr/bin/env python3
"""CPU-only coverage of immutable imports, incomplete assembly and no-GPU routing."""

import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("collection_runner_test", HERE / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
collection = runner.collection


class CollectionChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "collections" / "test"

    def bundle(self, label="source", evaluation="table2", profile="smoke"):
        root = self.root / label
        root.mkdir()
        (root / "status.json").write_text(json.dumps({"evaluation": evaluation,
            "profile": profile, "status": "completed", "exit_code": 0}))
        (root / "data.csv").write_text("key,value\na,1\n")
        (root / "table.csv").write_text("key,value\na,1\n")
        (root / "metadata.json").write_text('{"origin":"CPU fixture"}\n')
        return root

    def main(self, *arguments):
        with patch("sys.argv", ["run.sh", "--result-dir", str(self.root), *arguments]), \
             patch.object(runner, "capture", side_effect=AssertionError("offline operation queried hardware")), \
             patch.object(runner, "execute", side_effect=AssertionError("offline operation executed GPU adapter")):
            return runner.main()

    def test_import_snapshot_survives_original_edits_and_prevents_overwrite(self):
        source = self.bundle()
        record = collection.add_part(self.directory, "table2", "first", source, "smoke")
        source.joinpath("data.csv").write_text("modified original\n")
        copied = self.directory / record["path"]
        self.assertEqual(copied.joinpath("data.csv").read_text(), "key,value\na,1\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            collection.add_part(self.directory, "table2", "first", source, "smoke")

    def test_profile_and_unsuccessful_subset_are_rejected(self):
        source = self.bundle()
        with self.assertRaisesRegex(ValueError, "successful paper"):
            collection.add_part(self.directory, "table2", "first", source, "paper")
        state = json.loads((source / "status.json").read_text())
        state["status"] = "failed"
        (source / "status.json").write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "successful"):
            collection.add_part(self.directory, "table2", "first", source, "smoke")
        collection.initialize(self.directory, "smoke")
        with self.assertRaisesRegex(ValueError, "profile"):
            collection.initialize(self.directory, "paper")

    def test_evidence_tampering_blocks_assembly(self):
        record = collection.add_part(self.directory, "table2", "first", self.bundle(), "smoke")
        (self.directory / record["path"] / "data.csv").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "changed or removed"):
            collection.assemble_collection(self.directory, [{"id": "table2"}], "smoke")

    def test_symlinks_and_unsafe_names_are_rejected(self):
        source = self.bundle()
        (source / "outside.csv").symlink_to(source / "data.csv")
        with self.assertRaisesRegex(ValueError, "symlink"):
            collection.add_part(self.directory, "table2", "first", source, "smoke")
        for value in ("..", "../escape", "/absolute", "a/b", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                collection.name(value)

    def test_conflict_retains_previous_assembly(self):
        one = self.bundle("one")
        collection.add_part(self.directory, "table2", "one", one, "smoke")
        collection.assemble_collection(self.directory, [{"id": "table2"}], "smoke")
        prior = (self.directory / "assembled/table2/data.csv").read_bytes()
        two = self.bundle("two")
        (two / "data.csv").write_text("key,value\na,2\n")
        collection.add_part(self.directory, "table2", "two", two, "smoke")
        with self.assertRaisesRegex(ValueError, "conflict"):
            collection.assemble_collection(self.directory, [{"id": "table2"}], "smoke")
        self.assertEqual((self.directory / "assembled/table2/data.csv").read_bytes(), prior)
        self.assertTrue((self.directory / "assembly-error.json").exists())

    def test_cli_import_and_assemble_never_execute_or_query_gpu(self):
        source = self.bundle(profile="paper")
        self.assertEqual(self.main("--collection", "test", "--profile", "paper",
                         "--import-result", f"table2={source}", "--part", "model"), 0)
        self.assertEqual(self.main("--collection", "test", "--assemble", "table2"), 2)
        assembled = self.directory / "assembled"
        self.assertTrue((assembled / "table2/table.csv").is_file())
        self.assertIn("cuFFTDx", (assembled / "missing.csv").read_text())
        self.assertFalse(json.loads((assembled / "manifest.json").read_text())["new_gpu_measurements"])

    def test_empty_collection_records_pending_without_measurement(self):
        self.assertEqual(self.main("--collection", "test", "--assemble", "table4"), 2)
        state = json.loads((self.directory / "assembled/table4/status.json").read_text())
        self.assertEqual(state["status"], "pending")
        self.assertFalse((self.directory / "assembled/table4/data.csv").exists())

    def test_dry_run_does_not_create_collection(self):
        self.assertEqual(self.main("--collection", "test", "--assemble", "--dry-run"), 0)
        self.assertFalse(self.directory.exists())

    def test_selected_update_preserves_previously_assembled_evaluations(self):
        collection.add_part(self.directory, "table2", "model", self.bundle(), "smoke")
        collection.assemble_collection(self.directory, [{"id": "table2"}], "smoke")
        collection.assemble_collection(self.directory, [{"id": "table4"}], "smoke")
        self.assertTrue((self.directory / "assembled/table2/table.csv").is_file())
        self.assertTrue((self.directory / "assembled/table4/status.json").is_file())
        statuses = json.loads((self.directory / "assembled/manifest.json").read_text())["evaluations"]
        self.assertEqual({state["evaluation"] for state in statuses}, {"table2", "table4"})

    def test_terminal_checkpoint_recovery_and_figure_after_original_removed(self):
        spec = importlib.util.spec_from_file_location("collection_table3_fixture", HERE / "table3/test_assemble.py")
        fixture = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture)
        source = fixture.make_part(self.root / "interrupted", status="running")
        status = {"evaluation": "table3", "profile": "smoke", "status": "running", "exit_code": None}
        (source / "status.json").write_text(json.dumps(status))
        with self.assertRaisesRegex(ValueError, "successful"):
            collection.add_part(self.directory, "table3", "unfinished", source, "smoke")
        status.update(status="interrupted", exit_code=130)
        (source / "status.json").write_text(json.dumps(status))
        record = collection.add_part(self.directory, "table3", "checkpoint", source, "smoke")
        self.assertEqual(record["acceptance"], "validated_terminal_checkpoint")
        self.assertEqual(record["original_status"], "interrupted")
        shutil.rmtree(source)
        states = collection.assemble_collection(self.directory, [{"id": "table3"}, {"id": "figure2"}], "smoke")
        self.assertEqual(states[0]["collected_configurations"], 1)
        self.assertTrue((self.directory / "assembled/figure2/figure.pdf").is_file())
        metadata = json.loads((self.directory / "assembled/figure2/metadata.json").read_text())
        self.assertEqual(metadata["source"]["source_result_dir"], "../table3")
        # A later update to an unrelated table must retain and refresh the figure.
        collection.assemble_collection(self.directory, [{"id": "table4"}], "smoke")
        self.assertTrue((self.directory / "assembled/figure2/figure.pdf").is_file())


if __name__ == "__main__":
    unittest.main()
