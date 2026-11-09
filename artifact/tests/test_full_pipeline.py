"""CPU orchestration checks: one budget, safe cleanup, and complete CLI routing."""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import collection, full_pipeline as pipeline, runner
from artifact.cloud.config import CUFFTDX_MODE_ENV, CUDNN_MODE_ENV, DEFAULTS


class PipelineTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.args = SimpleNamespace(evaluations=[], part=None, assemble=False, import_result=[],
            cloud_batch=None, cloud_recover=None, profile=None, dry_run=False, cloud_plan=False,
            cloud_config=None, a100_backend="auto", local_wait_seconds=0, device=0,
            cloud_stage_seconds=1200, a100_stage_seconds=3600,
            result_dir=self.directory / "result", build_dir=self.directory / "build",
            collection=None, jobs=2, timeout=0, backend="all", gpu="A100", cloud_allow_dirty=False,
            cufftdx_mode=None, cudnn_mode=None)
        self.config = copy.deepcopy(DEFAULTS)
        self.config["budget_usd"] = 10.0
        self.entries = [{"id": name, "required_outputs": {"paper": ["data.csv"]}}
                        for name in ("table3", "table4", "figure2")]
        self.registry = {"evaluations": self.entries}
        self.parser = argparse.ArgumentParser()
        self.calls, self.allocated, self.cloud_parts = [], [], []
        self.stage_modes = []
        self.stage_cudnn_modes = []
        self.stage_codes = {}
        self.stage_costs = {"A100": .75, "H100": 1.25, "B200": 2.0}
        self.plan_costs = {"A100": 1.0, "H100": 2.0, "B200": 3.0}
        self.cleanup = True
        self.a100_backend = "local"
        self.raise_in_cloud = None
        self.no_evidence = False

    @staticmethod
    def select(requested, entries, parser):
        return list(entries) if requested == ["all"] else [entry for entry in entries if entry["id"] in requested]

    def fake_plan(self, client, config, gpu):
        self.assertGreaterEqual(config["budget_usd"], self.plan_costs[gpu])
        return {"gpu": gpu, "cost": {"planned_total_usd": self.plan_costs[gpu]},
                "account_before": {"balance_usd": 100}}

    def fake_local(self, *args, **kwargs):
        self.calls.append("A100")
        self.stage_modes.append(("A100", args[0].cufftdx_mode))
        self.stage_cudnn_modes.append(("A100", args[0].cudnn_mode))
        return self.stage_codes.get("A100", 0)

    def fake_cloud(self, args, entries, *unused):
        self.calls.append(args.gpu)
        configured = json.loads(args.cloud_config.read_text())
        self.allocated.append((args.gpu, configured["budget_usd"]))
        self.stage_modes.append((args.gpu, args.cufftdx_mode))
        self.stage_cudnn_modes.append((args.gpu, args.cudnn_mode))
        self.assertEqual(configured["environment"][CUFFTDX_MODE_ENV], args.cufftdx_mode)
        self.assertEqual(configured["environment"][CUDNN_MODE_ENV], args.cudnn_mode)
        self.cloud_parts.append((args.gpu, args.part, [entry["id"] for entry in entries]))
        code = self.stage_codes.get(args.gpu, 0)
        if not self.no_evidence:
            args.cloud_execution_summary = {"cleanup_confirmed": self.cleanup,
                "reserved_total_usd": self.stage_costs[args.gpu], "exit_code": code,
                "attempts": [{"reserved_total_usd": .25},
                             {"reserved_total_usd": self.stage_costs[args.gpu] - .25}]}
        if self.raise_in_cloud:
            raise self.raise_in_cloud
        return code

    def fake_assembly(self, destination, entries, profile):
        self.calls.append("assemble")
        states = []
        for entry in entries:
            output = destination / "assembled" / entry["id"]
            output.mkdir(parents=True, exist_ok=True)
            (output / "data.csv").write_text("source,value\ncpu-fixture,1\n")
            states.append({"evaluation": entry["id"], "status": "complete", "missing": [],
                           "full_paper_comparison": True})
        return states

    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(pipeline, "load_config", return_value=self.config))
        stack.enter_context(mock.patch.object(pipeline, "choose_a100", side_effect=lambda *a, **k:
            (self.a100_backend, {"available": True, "uuid": "GPU-a100"})))
        stack.enter_context(mock.patch.object(pipeline, "VastClient", return_value=object()))
        stack.enter_context(mock.patch.object(pipeline.cloud, "plan", side_effect=self.fake_plan))
        stack.enter_context(mock.patch.object(pipeline, "validate_private_key"))
        stack.enter_context(mock.patch.object(pipeline, "run_local", side_effect=self.fake_local))
        stack.enter_context(mock.patch.object(pipeline.cloud, "run", side_effect=self.fake_cloud))
        stack.enter_context(mock.patch.object(collection, "assemble_collection", side_effect=self.fake_assembly))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def execute(self):
        return pipeline.run(self.args, self.registry, ROOT, collection, self.select, self.parser, mock.Mock())

    def summary(self):
        paths = list((self.args.result_dir / "pipelines").glob("*/pipeline.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text())

    def test_complete_pipeline_reserves_later_gpu_then_debits_all_prior_attempts(self):
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.calls, ["A100", "H100", "B200", "assemble"])
        self.assertEqual(self.allocated, [("H100", 7.0), ("B200", 8.75)])
        self.assertEqual(self.cloud_parts, [("H100", "h100", ["table3", "table4"]),
                                           ("B200", "b200", ["table3", "table4"])])
        summary = self.summary()
        self.assertEqual(summary["reserved_total_usd"], 3.25)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["profile"], "paper")
        self.assertTrue(summary["assembly"]["outputs_complete"])

    def test_cloud_a100_includes_all_evaluations_and_preserves_later_gpu_budgets(self):
        self.a100_backend = "vast"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.allocated, [("A100", 5.0), ("H100", 6.25), ("B200", 8.0)])
        self.assertEqual(self.cloud_parts[0], ("A100", "a100", ["table3", "table4"]))
        self.assertEqual(self.summary()["reserved_total_usd"], 4.0)

    def test_configured_all_mode_reaches_local_and_cloud_stages_and_metadata(self):
        self.config["environment"][CUFFTDX_MODE_ENV] = "all"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_modes, [(gpu, "all") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.summary()["cufftdx_mode"], "all")
        self.assertEqual(self.summary()["preflight"]["cufftdx_mode"], "all")

    def test_explicit_mode_overrides_config_for_every_stage(self):
        self.args.cufftdx_mode = "preset"
        self.config["environment"][CUFFTDX_MODE_ENV] = "all"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_modes, [(gpu, "preset") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.config["environment"][CUFFTDX_MODE_ENV], "all")

    def test_configured_full_cudnn_mode_reaches_every_stage_and_metadata(self):
        self.config["environment"][CUDNN_MODE_ENV] = "full"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_cudnn_modes, [(gpu, "full") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.summary()["cudnn_mode"], "full")
        self.assertEqual(self.summary()["preflight"]["cudnn_mode"], "full")

    def test_explicit_fast_cudnn_mode_overrides_full_config(self):
        self.args.cudnn_mode = "fast"
        self.config["environment"][CUDNN_MODE_ENV] = "full"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_cudnn_modes, [(gpu, "fast") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.config["environment"][CUDNN_MODE_ENV], "full")

    def test_total_preflight_budget_failure_creates_no_results_or_stages(self):
        self.config["budget_usd"] = 4
        with self.patches(), self.assertRaisesRegex(ValueError, "shared"):
            self.execute()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.result_dir.exists())

    def test_failed_stage_assembles_partial_evidence_without_next_rental(self):
        self.stage_codes["H100"] = 1
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, ["A100", "H100", "assemble"])
        summary = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["stages"][-1]["status"], "failed")
        self.assertEqual(summary["reserved_total_usd"], 1.25)

    def test_unknown_cleanup_blocks_next_gpu_and_preserves_terminal_audit(self):
        self.cleanup = False
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertNotIn("B200", self.calls)
        summary = self.summary()
        self.assertFalse(summary["stages"][-1]["cloud"]["cleanup_confirmed"])
        self.assertNotEqual(summary["stages"][-1]["status"], "running")
        self.assertIn("assemble", self.calls)

    def test_missing_cloud_cost_evidence_blocks_next_gpu(self):
        self.no_evidence = True
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertNotIn("B200", self.calls)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["reserved_total_usd"], 7.0)
        self.assertTrue(self.summary()["stages"][-1]["accounting_unconfirmed"])

    def test_assembly_error_keeps_completed_gpu_costs_and_failed_summary(self):
        with self.patches(), mock.patch.object(collection, "assemble_collection", side_effect=ValueError("conflicting raw evidence")):
            self.assertEqual(self.execute(), 1)
        summary = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["assembly_error"], "conflicting raw evidence")
        self.assertEqual(summary["reserved_total_usd"], 3.25)
        self.assertEqual([stage["status"] for stage in summary["stages"]], ["completed"] * 3)
        self.assertEqual(self.calls, ["A100", "H100", "B200"])

    def test_interrupt_preserves_available_cleanup_cost_and_assembles_prior_parts(self):
        self.raise_in_cloud = KeyboardInterrupt()
        with self.patches():
            self.assertEqual(self.execute(), 130)
        summary = self.summary()
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["reserved_total_usd"], 1.25)
        self.assertEqual(summary["stages"][-1]["cloud"]["reserved_total_usd"], 1.25)
        self.assertNotEqual(summary["stages"][-1]["status"], "running")
        self.assertNotIn("B200", self.calls)
        self.assertIn("assemble", self.calls)

    def test_dry_run_never_queries_gpu_credentials_or_provider(self):
        self.args.dry_run = True
        with mock.patch.object(pipeline, "load_config", side_effect=AssertionError("configuration read")), \
                mock.patch.object(pipeline, "inventory", side_effect=AssertionError("GPU queried")), \
                mock.patch.object(pipeline, "VastClient", side_effect=AssertionError("provider queried")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.execute(), 0)
        self.assertFalse(self.args.result_dir.exists())

    def test_dry_run_reports_configured_mode_without_policy_or_credential_access(self):
        self.args.dry_run = True
        self.args.cloud_config = self.directory / "preview.json"
        self.args.cloud_config.write_text(json.dumps({"budget_usd": 50,
            "api_key_file": "/unreadable/key", "environment": {CUFFTDX_MODE_ENV: "all", CUDNN_MODE_ENV: "full"}}))
        output = io.StringIO()
        with mock.patch.object(pipeline, "load_config", side_effect=AssertionError("full config read")), \
                mock.patch.object(pipeline, "VastClient", side_effect=AssertionError("provider queried")), \
                mock.patch.object(pipeline, "inventory", side_effect=AssertionError("GPU queried")), \
                redirect_stdout(output):
            self.assertEqual(self.execute(), 0)
        self.assertIn("cuFFTDx mode: all (A100/H100/B200)", output.getvalue())
        self.assertIn("cuDNN mode: full", output.getvalue())
        self.assertFalse(self.args.result_dir.exists())

    def test_historical_table7_absence_is_explicit_but_missing_measurements_fail(self):
        output = self.directory / "assembled"
        task = output / "table7"
        task.mkdir(parents=True)
        (task / "data.csv").write_text("fixture\n")
        entries = [{"id": "table7", "required_outputs": {"paper": ["data.csv"]}}]
        states = [{"evaluation": "table7", "status": "partial", "missing": [{"item": "radix16.time"}]}]
        result = pipeline.assembly_result(states, output, entries, "paper", ROOT)
        self.assertTrue(result["outputs_complete"])
        self.assertEqual(len(result["coverage_limits"]), 1)
        states[0]["missing"].append({"item": "register_uniform.time"})
        self.assertFalse(pipeline.assembly_result(states, output, entries, "paper", ROOT)["outputs_complete"])

    def test_failed_or_unknown_assembly_status_cannot_pass_with_existing_files(self):
        output = self.directory / "assembled"
        task = output / "table3"
        task.mkdir(parents=True)
        (task / "data.csv").write_text("fixture\n")
        entries = [{"id": "table3", "required_outputs": {"paper": ["data.csv"]}}]
        for status in ("failed", "pending", "running", "unknown", "completed"):
            with self.subTest(status=status):
                states = [{"evaluation": "table3", "status": status, "missing": []}]
                result = pipeline.assembly_result(states, output, entries, "paper", ROOT)
                self.assertFalse(result["outputs_complete"])
                self.assertTrue(result["unresolved"])


class LocalInventoryTests(unittest.TestCase):
    INVENTORY = "0, GPU-v100, Tesla V100, 32768\n1, GPU-a100, NVIDIA A100-PCIE-80GB, 81920\n"

    def inventory(self, mask, processes="", device=0):
        responses = [subprocess.CompletedProcess([], 0, self.INVENTORY, ""),
                     subprocess.CompletedProcess([], 0, processes, "")]
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": mask}), mock.patch.object(
                pipeline.subprocess, "run", side_effect=responses):
            return pipeline.inventory(device)

    def test_logical_visibility_selects_a100_by_index_or_uuid(self):
        for mask in ("1", "GPU-a100", "1,0"):
            with self.subTest(mask=mask):
                observed = self.inventory(mask)
                self.assertTrue(observed["available"])
                self.assertEqual(observed["uuid"], "GPU-a100")
        self.assertFalse(self.inventory("1,0", device=1)["available"])
        self.assertFalse(self.inventory("0")["available"])

    def test_compute_process_blocks_gpu_even_without_utilization_query(self):
        self.assertFalse(self.inventory("1", "GPU-a100, 31337\n")["available"])
        self.assertTrue(self.inventory("1", "GPU-v100, 31337\n")["available"])

    def test_malformed_process_inventory_cannot_be_mistaken_for_an_idle_gpu(self):
        for processes in ("unsupported query\n", "GPU-a100\n", "GPU-v100, N/A\n",
                          "GPU-v100, -1\n", "GPU-v100, 0\n", "GPU-v100, 42, extra\n"):
            with self.subTest(processes=processes):
                self.assertFalse(self.inventory("1", processes)["available"])

    def test_hidden_or_unknown_visible_devices_fail_closed(self):
        for mask in ("", "-1", "9", "GPU-unknown"):
            with self.subTest(mask=mask):
                self.assertFalse(self.inventory(mask)["available"])

    def test_auto_falls_back_but_explicit_local_fails_on_busy_a100(self):
        args = SimpleNamespace(a100_backend="auto", local_wait_seconds=0, device=0)
        with mock.patch.object(pipeline, "inventory", return_value={"available": False, "reason": "busy"}):
            self.assertEqual(pipeline.choose_a100(args)[0], "vast")
            args.a100_backend = "local"
            with self.assertRaisesRegex(RuntimeError, "busy"):
                pipeline.choose_a100(args)

    def test_local_child_is_pinned_to_checked_uuid_with_logical_device_zero(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = SimpleNamespace(profile="paper", collection="test", result_dir=folder / "result",
                build_dir=folder / "build", device=1, jobs=2, timeout=0, cufftdx_mode="preset", cudnn_mode="full")
            process = mock.Mock(returncode=0)
            process.poll.return_value = 0
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0", CUFFTDX_MODE_ENV: "all", CUDNN_MODE_ENV: "fast"}), mock.patch.object(
                    pipeline.subprocess, "Popen", return_value=process) as launch:
                self.assertEqual(pipeline.run_local(args, ROOT, folder, mock.Mock(), gpu_uuid="GPU-a100"), 0)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,0")
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--device") + 1], "0")
            self.assertEqual(command[command.index("--backend") + 1], "local")
            self.assertEqual(command[command.index("--cufftdx-mode") + 1], "preset")
            self.assertEqual(command[command.index("--cudnn-mode") + 1], "full")
            self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-a100")
            self.assertEqual(launch.call_args.kwargs["env"][CUFFTDX_MODE_ENV], "preset")
            self.assertEqual(launch.call_args.kwargs["env"][CUDNN_MODE_ENV], "full")

    def test_local_child_requires_a_previously_checked_uuid(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(pipeline.subprocess, "Popen") as launch:
            with self.assertRaises((ValueError, RuntimeError)):
                pipeline.run_local(SimpleNamespace(), ROOT, Path(temporary), mock.Mock())
            launch.assert_not_called()


class DefaultRoutingTests(unittest.TestCase):
    def test_no_arguments_and_all_route_to_complete_pipeline(self):
        for argv in ([], ["all"], ["--backend", "all"]):
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", ["run.sh", *argv]), \
                    mock.patch.object(runner.full_pipeline, "run", return_value=17) as full:
                self.assertEqual(runner.main(), 17)
                full.assert_called_once()

    def test_explicit_local_and_named_evaluation_keep_scoped_behavior(self):
        for argv in (["all", "--backend", "local"], ["table3"]):
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", ["run.sh", *argv, "--dry-run"]), \
                    mock.patch.object(runner.full_pipeline, "run", side_effect=AssertionError("unexpected full pipeline")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)

    def test_full_a100_cloud_collection_accepts_all_measured_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            config.write_text('{"budget_usd": 1.5}\n')
            result = Path(temporary) / "result"
            argv = ["run.sh", "all", "--backend", "vast", "--gpu", "A100", "--collection", "cloud-all",
                    "--result-dir", str(result), "--cloud-config", str(config), "--dry-run"]
            output = io.StringIO()
            with mock.patch.object(sys, "argv", argv), redirect_stdout(output):
                self.assertEqual(runner.main(), 0)
            proposal = json.loads(output.getvalue())
            self.assertFalse(proposal["api_calls"])
            self.assertFalse(result.exists())
            self.assertTrue({"table5", "table6", "table7", "figure5"} <= set(proposal["jobs"][0]["evaluations"]))
            self.assertNotIn("figure2", proposal["jobs"][0]["evaluations"])


if __name__ == "__main__":
    unittest.main()
