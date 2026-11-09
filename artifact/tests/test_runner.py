#!/usr/bin/env python3
"""CPU checks of task selection and fail-closed result handling."""
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("artifact_runner", Path(__file__).parents[1] / "runner.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerChecks(unittest.TestCase):
    def dry_plan(self, *arguments):
        output = io.StringIO()
        with patch("sys.argv", ["run.sh", *arguments, "--dry-run"]), redirect_stdout(output), \
             patch.object(runner, "capture", side_effect=AssertionError("dry plan queried hardware")), \
             patch.object(runner, "execute", side_effect=AssertionError("dry plan executed a benchmark")), \
             patch.object(runner.cloud, "run", side_effect=AssertionError("dry plan dispatched a rental")), \
             patch.object(runner.full_pipeline, "preflight", side_effect=AssertionError("dry plan queried provider")), \
             patch.object(runner.full_pipeline, "inventory", side_effect=AssertionError("dry plan queried GPU")):
            self.assertEqual(runner.main(), 0)
        return output.getvalue()

    def test_no_arguments_and_all_dispatch_complete_paper_pipeline(self):
        for arguments in ((), ("all",), ("--backend", "all")):
            with self.subTest(arguments=arguments):
                output = self.dry_plan(*arguments)
                self.assertIn("Complete pipeline; profile: paper", output)
                self.assertIn("H100 PCIe 80GB: table3, table4; B200: table3, table4", output)
                self.assertIn("Then assemble all tables and figures automatically", output)
                for evaluation in ("table5", "table6", "table7", "figure3", "figure4", "figure5"):
                    self.assertIn(evaluation, output)

    def test_explicit_local_all_keeps_single_machine_smoke_default(self):
        output = self.dry_plan("all", "--backend", "local")
        self.assertNotIn("Complete pipeline", output)
        self.assertIn("Profile: smoke; device: 0; sequential execution", output)
        self.assertIn("table6", output)
        self.assertIn("figure5", output)

    def test_named_tasks_keep_local_smoke_default(self):
        output = self.dry_plan("validation", "table3")
        self.assertNotIn("Complete pipeline", output)
        self.assertIn("Profile: smoke; device: 0; sequential execution", output)
        self.assertIn("validation", output)
        self.assertIn("table3", output)
        self.assertNotIn("figure5", output)

    def test_cufftdx_mode_local_plan_default_environment_and_cli_precedence(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIn("cuFFTDx mode: preset", self.dry_plan("table3"))
        with patch.dict(os.environ, {runner.CUFFTDX_MODE_ENV: "all"}):
            self.assertIn("cuFFTDx mode: all", self.dry_plan("table3"))
            self.assertIn("cuFFTDx mode: preset", self.dry_plan("table3", "--cufftdx-mode", "preset"))

    def test_fast_default_and_full_shorthand_report_both_effective_modes(self):
        with patch.dict(os.environ, {}, clear=True):
            output = self.dry_plan("figure4")
            self.assertIn("cuFFTDx mode: preset", output)
            self.assertIn("cuDNN mode: fast", output)
            output = self.dry_plan("figure4", "--cufftdx-mode", "all")
            self.assertIn("cuFFTDx mode: all", output)
            self.assertIn("cuDNN mode: fast", output)
        with patch.dict(os.environ, {runner.CUFFTDX_MODE_ENV: "preset", runner.CUDNN_MODE_ENV: "fast"}):
            for arguments in (("--full",), ("figure4", "--full")):
                output = self.dry_plan(*arguments)
                self.assertIn("cuFFTDx mode: all", output)
                self.assertIn("cuDNN mode: full", output)

    def test_full_conflicts_fail_before_any_dispatch(self):
        for options in (("--cufftdx-mode", "preset"), ("--cudnn-mode", "fast")):
            with self.subTest(options=options), patch("sys.argv", ["run.sh", "--full", *options]), \
                    patch.object(runner.full_pipeline, "run", side_effect=AssertionError("pipeline dispatched")), \
                    patch.object(runner.cloud, "run", side_effect=AssertionError("cloud dispatched")), \
                    redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                runner.main()
            self.assertEqual(raised.exception.code, 2)

    def test_full_cloud_dispatch_carries_both_explicit_overrides(self):
        with patch("sys.argv", ["run.sh", "figure4", "--backend", "vast", "--full"]), \
                patch.object(runner.cloud, "run", return_value=0) as cloud_run:
            self.assertEqual(runner.main(), 0)
        args = cloud_run.call_args.args[0]
        self.assertEqual((args.cufftdx_mode, args.cudnn_mode), ("all", "full"))

    def test_cudnn_environment_and_explicit_cli_precedence(self):
        with patch.dict(os.environ, {runner.CUDNN_MODE_ENV: "full"}):
            self.assertIn("cuDNN mode: full", self.dry_plan("figure5"))
            self.assertIn("cuDNN mode: fast", self.dry_plan("figure5", "--cudnn-mode", "fast"))

    def test_cufftdx_mode_manifest_records_effective_cli_value(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = Path(temporary) / "result"
            argv = ["run.sh", "table3", "--cufftdx-mode", "preset", "--cudnn-mode", "full", "--result-dir", str(result)]
            state = {"evaluation": "table3", "status": "completed", "exit_code": 0}
            with patch("sys.argv", argv), patch.dict(os.environ, {runner.CUFFTDX_MODE_ENV: "all"}), \
                    patch.object(runner, "capture", return_value={}), \
                    patch.object(runner, "execute", return_value=state), redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
            manifest = json.loads(next((result / "runs").glob("*/manifest.json")).read_text())
            self.assertEqual(manifest["cufftdx_mode"], "preset")
            self.assertEqual(manifest["configuration"][runner.CUFFTDX_MODE_ENV], "preset")
            self.assertEqual(manifest["cudnn_mode"], "full")
            self.assertEqual(manifest["configuration"][runner.CUDNN_MODE_ENV], "full")

    def test_cloud_worker_expanded_full_task_list_does_not_start_nested_pipeline(self):
        registry = json.loads(runner.REGISTRY.read_text())
        # The coordinator expands 'all' before passing the named tasks to the
        # worker. The worker intentionally has no cloud credentials/backend flag.
        tasks = [entry["id"] for entry in runner.select_tasks(["all"], registry["evaluations"],
                                                              argparse.ArgumentParser())
                 if entry["id"] != "figure2"]
        with patch.object(runner.full_pipeline, "run", side_effect=AssertionError("worker attempted nested pipeline")):
            output = self.dry_plan(*tasks, "--profile", "paper", "--collection", "worker-fixture", "--part", "a100")
        self.assertIn("Profile: paper; device: 0; sequential execution", output)
        for evaluation in tasks:
            self.assertIn(evaluation, output)

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

    def run_adapter(self, script, required=("data.csv",), mode=None, convolution_mode=None):
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
                                   jobs=1, run_id="test-run", result_dir=output, timeout=2,
                                   cufftdx_mode=mode, cudnn_mode=convolution_mode)
            with patch.object(runner, "ROOT", root):
                state = runner.execute(entry, output, args)
            self.assertEqual(json.loads((output / "status.json").read_text()), state)
            return state

    def test_success_requires_output_and_keeps_subset_scope(self):
        state = self.run_adapter('printf "key,value\\na,1\\n" > "$THUNDERFFT_RESULT_DIR/data.csv"\n')
        self.assertEqual(state["status"], "completed")
        self.assertFalse(state["full_paper_comparison"])
        self.assertEqual(state["coverage"], "explicit subset")

    def test_adapter_receives_effective_mode_without_changing_parent_environment(self):
        with patch.dict(os.environ, {runner.CUFFTDX_MODE_ENV: "all"}):
            state = self.run_adapter('test "$THUNDERFFT_TABLE3_CUFFTDX_MODE" = preset || exit 9\n'
                                     'printf "fixture\\n" > "$THUNDERFFT_RESULT_DIR/data.csv"\n', mode="preset")
            self.assertEqual(os.environ[runner.CUFFTDX_MODE_ENV], "all")
        self.assertEqual((state["status"], state["cufftdx_mode"]), ("completed", "preset"))

    def test_adapter_receives_cudnn_mode_without_changing_parent_environment(self):
        with patch.dict(os.environ, {runner.CUDNN_MODE_ENV: "fast"}):
            state = self.run_adapter('test "$THUNDERFFT_CUDNN_MODE" = full || exit 9\n'
                                     'printf "fixture\\n" > "$THUNDERFFT_RESULT_DIR/data.csv"\n', convolution_mode="full")
            self.assertEqual(os.environ[runner.CUDNN_MODE_ENV], "fast")
        self.assertEqual((state["status"], state["cudnn_mode"]), ("completed", "full"))

    def test_zero_exit_without_output_fails(self):
        self.assertEqual(self.run_adapter("exit 0\n")["status"], "failed")

    def test_native_failure_is_not_completion(self):
        self.assertEqual(self.run_adapter("exit 1\n")["status"], "failed")

    def test_pending_keeps_its_own_exit_code(self):
        state = self.run_adapter("exit 78\n")
        self.assertEqual((state["status"], state["exit_code"]), ("pending", 78))


if __name__ == "__main__":
    unittest.main()
