#!/usr/bin/env python3
"""CPU checks of task selection and fail-closed result handling."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("artifact_runner", Path(__file__).parents[1] / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerChecks(unittest.TestCase):
    def test_available_default_omits_unfinished_and_orders_dependencies(self):
        entries = [{"id": "figure", "available": True, "depends_on": ["data"]},
                   {"id": "external", "available": False},
                   {"id": "data", "available": True}]
        parser = argparse.ArgumentParser()
        self.assertEqual([e["id"] for e in runner.select_tasks([], entries, parser)], ["data", "figure"])
        self.assertEqual(len(runner.select_tasks(["all"], entries, parser)), 3)

    def test_optional_dependency_orders_only_when_selected(self):
        entries = [{"id": "model", "available": True, "optional_depends_on": ["data"]},
                   {"id": "data", "available": True}]
        parser = argparse.ArgumentParser()
        self.assertEqual([e["id"] for e in runner.select_tasks(["model"], entries, parser)], ["model"])
        self.assertEqual([e["id"] for e in runner.select_tasks([], entries, parser)], ["data", "model"])

    def run_adapter(self, script, required=("data.csv",)):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            task = root / "artifact" / "test"
            task.mkdir(parents=True)
            (task / "run.sh").write_text(script)
            output = root / "result"
            output.mkdir()
            entry = {"id": "test", "title": "Test", "implementation": "ready_subset",
                     "coverage": "explicit subset", "full_paper_comparison": False,
                     "required_outputs": {"smoke": list(required)}}
            args = SimpleNamespace(profile="smoke", build_dir=root / "build", device=0,
                                   jobs=1, run_id="test-run", result_dir=output, timeout=2)
            with patch.object(runner, "ROOT", root):
                state = runner.execute(entry, output, args)
            self.assertEqual(json.loads((output / "status.json").read_text()), state)
            return state

    def test_success_requires_output_and_keeps_subset_scope(self):
        state = self.run_adapter('printf "key,value\\na,1\\n" > "$THUNDERFFT_RESULT_DIR/data.csv"\n')
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["full_paper_comparison"])
        self.assertEqual(state["coverage"], "explicit subset")

    def test_zero_exit_without_output_fails(self):
        self.assertEqual(self.run_adapter("exit 0\n")["status"], "failed")

    def test_native_failure_is_not_completion(self):
        self.assertEqual(self.run_adapter("exit 1\n")["status"], "failed")

    def test_pending_keeps_its_own_exit_code(self):
        state = self.run_adapter("exit 78\n")
        self.assertEqual((state["status"], state["exit_code"]), ("pending", 78))


if __name__ == "__main__":
    unittest.main()
