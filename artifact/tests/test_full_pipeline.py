"""CPU orchestration checks: one budget, safe cleanup, and complete CLI routing."""
import argparse
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import collection, full_pipeline as pipeline, runner
from artifact.cloud.config import CUFFTDX_MODE_ENV, CUDNN_MODE_ENV, TABLE5_RANGE_MODE_ENV, DEFAULTS
from artifact.cloud.provider import ProviderError


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
            cufftdx_mode=None, cudnn_mode=None, table5_range_mode=None)
        self.config = copy.deepcopy(DEFAULTS)
        self.config["budget_usd"] = 10.0
        self.entries = [{"id": name, "required_outputs": {"paper": ["data.csv"]}}
                        for name in ("table3", "table4", "figure2")]
        self.registry = {"evaluations": self.entries}
        self.parser = argparse.ArgumentParser()
        self.calls, self.allocated = [], []
        self.stage_modes = []
        self.stage_cudnn_modes = []
        self.stage_range_modes = []
        self.stage_codes = {}
        self.stage_costs = {"A100": .75, "H100": 1.25, "B200": 2.0}
        self.plan_costs = {"A100": 1.0, "H100": 2.0, "B200": 3.0}
        self.cleanup = True
        self.a100_backend = "local"
        self.no_evidence = False
        self.group_ready = False
        self.prepare_error = False
        self.interrupted = False
        self.plan_calls = []
        self.plan_errors = {}
        self.group_request = None

    @staticmethod
    def select(requested, entries, parser):
        return list(entries) if requested == ["all"] else [entry for entry in entries if entry["id"] in requested]

    def fake_plan(self, client, config, gpu):
        self.plan_calls.append(gpu)
        if gpu in self.plan_errors:
            raise ProviderError(self.plan_errors[gpu])
        self.assertGreaterEqual(config["budget_usd"], self.plan_costs[gpu])
        return {"gpu": gpu, "cost": {"planned_total_usd": self.plan_costs[gpu]},
                "account_before": {"balance_usd": 100}}

    def fake_local(self, *args, **kwargs):
        self.assertTrue(self.group_ready, "A100 started before cloud resources were ready")
        self.calls.append("A100")
        self.stage_modes.append(("A100", args[0].cufftdx_mode))
        self.stage_cudnn_modes.append(("A100", args[0].cudnn_mode))
        self.stage_range_modes.append(("A100", args[0].table5_range_mode))
        return self.stage_codes.get("A100", 0)

    def fake_group(self, args, registry, root, collection_module, select_tasks, parser,
                   configs, proposals, *, local_run=None, stage_update=None, ready_check=None):
        self.group_request = {"args": args, "configs": copy.deepcopy(configs),
                              "proposals": copy.deepcopy(proposals)}
        stages, reserved = [], 0.0
        self.allocated = [(gpu, config["budget_usd"]) for gpu, config in configs.items()]
        self.assertLessEqual(sum(value for _, value in self.allocated), self.config["budget_usd"] + 1e-9)
        for gpu, configured in configs.items():
            self.assertGreaterEqual(configured["budget_usd"], self.plan_costs[gpu])
            self.assertEqual(configured["environment"][CUFFTDX_MODE_ENV], args.cufftdx_mode)
            self.assertEqual(configured["environment"][CUDNN_MODE_ENV], args.cudnn_mode)
            self.assertEqual(configured["environment"][TABLE5_RANGE_MODE_ENV], args.table5_range_mode)
        failure = self.prepare_error
        if not failure and ready_check is not None:
            try:
                ready_check()
            except (ValueError, RuntimeError):
                failure = True
        if not failure:
            self.group_ready = True
        if local_run is not None:
            code = 1 if failure else local_run(threading.Event())
            stages.append({"gpu": "A100", "backend": "local", "exit_code": code,
                           "status": "completed" if code == 0 else "failed", "elapsed_seconds": .001})
        for gpu, configured in configs.items():
            code = 130 if self.interrupted else 1 if failure else self.stage_codes.get(gpu, 0)
            if not failure:
                self.calls.append(gpu)
                self.stage_modes.append((gpu, args.cufftdx_mode))
                self.stage_cudnn_modes.append((gpu, args.cudnn_mode))
                self.stage_range_modes.append((gpu, args.table5_range_mode))
            charge = configured["budget_usd"] if self.no_evidence else self.stage_costs[gpu]
            reserved += charge
            cloud = None if self.no_evidence else {"cleanup_confirmed": self.cleanup,
                "reserved_total_usd": charge, "exit_code": code,
                "attempts": [{"reserved_total_usd": charge}]}
            stage = {"gpu": gpu, "backend": "vast", "exit_code": code, "elapsed_seconds": .001,
                     "status": "interrupted" if code == 130 else "completed" if code == 0 else "failed",
                     "cloud": cloud}
            if self.no_evidence:
                stage["accounting_unconfirmed"] = True
            stages.append(stage)
        if stage_update:
            for stage in stages:
                stage_update(stage["gpu"], dict(stage))
        code = 130 if self.interrupted else int(any(stage["exit_code"] for stage in stages))
        return {"exit_code": code, "stages": stages, "reserved_total_usd": reserved,
                "cleanup_confirmed": self.cleanup and not self.no_evidence,
                "billing_settlement_verified": False, "audit_folder": str(self.directory / "group-audit")}

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
        stack.enter_context(mock.patch.object(pipeline, "inventory", return_value={"available": True, "uuid": "GPU-a100"}))
        stack.enter_context(mock.patch.object(pipeline.cloud, "plan", side_effect=self.fake_plan))
        stack.enter_context(mock.patch.object(pipeline, "validate_private_key"))
        stack.enter_context(mock.patch.object(pipeline, "run_local", side_effect=self.fake_local))
        stack.enter_context(mock.patch.object(pipeline.cloud, "run", side_effect=AssertionError("serial cloud path used")))
        stack.enter_context(mock.patch.object(pipeline.cloud_group, "run", side_effect=self.fake_group))
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

    def test_complete_pipeline_allocates_one_budget_and_assembles_after_group_execution(self):
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.calls, ["A100", "H100", "B200", "assemble"])
        self.assertEqual(self.allocated, [("H100", 4.5), ("B200", 5.5)])
        self.assertEqual(self.plan_calls, ["H100", "B200"])
        self.assertEqual(set(self.group_request["configs"]), {"H100", "B200"})
        summary = self.summary()
        self.assertEqual(summary["reserved_total_usd"], 3.25)
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["profile"], "paper")
        self.assertTrue(summary["assembly"]["outputs_complete"])

    def test_cloud_a100_joins_group_with_one_shared_fixed_budget(self):
        self.a100_backend = "vast"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual([gpu for gpu, _ in self.allocated], ["A100", "H100", "B200"])
        self.assertAlmostEqual(sum(value for _, value in self.allocated), 10.0)
        for gpu, value in self.allocated:
            self.assertAlmostEqual(value, self.plan_costs[gpu] + 4.0 / 3.0)
        self.assertEqual(self.plan_calls, ["A100", "H100", "B200"])
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

    def test_configured_full_range_mode_reaches_every_stage_and_metadata(self):
        self.config["environment"][TABLE5_RANGE_MODE_ENV] = "full"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_range_modes, [(gpu, "full") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.summary()["table5_range_mode"], "full")
        self.assertEqual(self.summary()["preflight"]["table5_range_mode"], "full")

    def test_explicit_boundary_range_mode_overrides_full_config(self):
        self.args.table5_range_mode = "boundary"
        self.config["environment"][TABLE5_RANGE_MODE_ENV] = "full"
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.stage_range_modes, [(gpu, "boundary") for gpu in ("A100", "H100", "B200")])
        self.assertEqual(self.config["environment"][TABLE5_RANGE_MODE_ENV], "full")

    def test_total_preflight_budget_failure_creates_no_results_or_stages(self):
        self.config["budget_usd"] = 4
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.plan_calls, ["H100", "B200"])
        self.assertIsNone(self.group_request)
        self.assertFalse(self.args.result_dir.exists())

    def test_missing_offers_scan_all_required_gpus_before_stopping_without_work(self):
        for backend, failures in (("local", {"H100": "H100 unavailable"}),
                                  ("local", {"B200": "B200 unavailable"}),
                                  ("local", {"H100": "H100 unavailable", "B200": "B200 unavailable"}),
                                  ("vast", {"A100": "A100 unavailable", "H100": "H100 unavailable"})):
            with self.subTest(backend=backend, failures=failures):
                self.a100_backend, self.plan_errors = backend, failures
                self.plan_calls.clear()
                output = io.StringIO()
                with self.patches(), redirect_stderr(output):
                    self.assertEqual(self.execute(), 1)
                self.assertEqual(self.plan_calls, (["A100"] if backend == "vast" else []) + ["H100", "B200"])
                self.assertIsNone(self.group_request)
                self.assertEqual(self.calls, [])
                self.assertFalse(self.args.result_dir.exists())
                for detail in failures.values():
                    self.assertIn(detail, output.getvalue())
                self.assertIn("--cloud-plan", output.getvalue())
                self.assertIn("--backend local", output.getvalue())

    def test_preflight_error_keeps_all_failures_and_successful_proposals(self):
        self.plan_errors = {"H100": "fixture H100 unavailable"}
        with self.patches(), self.assertRaises(pipeline.PreflightUnavailable) as raised:
            pipeline.preflight(self.config, "vast", self.args, ROOT)
        self.assertEqual(self.plan_calls, ["A100", "H100", "B200"])
        self.assertEqual(raised.exception.failures, self.plan_errors)
        self.assertEqual([proposal["gpu"] for proposal in raised.exception.proposals], ["A100", "B200"])

    def test_unavailable_commands_preserve_modes_paths_device_and_cloud_settings(self):
        self.args.profile = "smoke"
        self.args.cufftdx_mode, self.args.cudnn_mode, self.args.table5_range_mode = "all", "full", "full"
        self.args.result_dir = self.directory / "results with ' quotes"
        self.args.build_dir = self.directory / "build with spaces"
        self.args.cloud_config = self.directory / "cloud $(literal) settings.json"
        self.args.device, self.args.jobs, self.args.timeout = 1, 8, 91
        self.args.a100_backend = "local"
        self.args.local_wait_seconds = 45
        self.args.cloud_stage_seconds, self.args.a100_stage_seconds = 700, 1500
        self.args.collection = "custom-evaluation"
        self.args.cloud_allow_dirty = True
        self.plan_errors = {"H100": "fixture no H100"}
        output = io.StringIO()
        with self.patches(), redirect_stderr(output):
            self.assertEqual(self.execute(), 1)
        commands = [shlex.split(line.strip()) for line in output.getvalue().splitlines()
                    if line.strip().startswith("./run.sh")]
        self.assertEqual(len(commands), 3)
        check, retry, local = commands
        self.assertIn("--cloud-plan", check)
        self.assertNotIn("--cloud-plan", retry)
        self.assertEqual(local[1:4], ["all", "--backend", "local"])
        expected_common = {"--profile": "smoke", "--cufftdx-mode": "all", "--cudnn-mode": "full",
            "--table5-range-mode": "full", "--result-dir": str(self.args.result_dir),
            "--build-dir": str(self.args.build_dir), "--device": "1", "--jobs": "8", "--timeout": "91"}
        for command in commands:
            for flag, value in expected_common.items():
                self.assertEqual(command[command.index(flag) + 1], value)
        expected_cloud = {"--cloud-config": str(self.args.cloud_config), "--a100-backend": "local",
            "--local-wait-seconds": "45", "--cloud-stage-seconds": "700",
            "--a100-stage-seconds": "1500", "--collection": "custom-evaluation"}
        for command in (check, retry):
            for flag, value in expected_cloud.items():
                self.assertEqual(command[command.index(flag) + 1], value)
            self.assertIn("--cloud-allow-dirty", command)
        self.assertNotIn("--cloud-config", local)
        self.assertNotIn("--cloud-allow-dirty", local)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.result_dir.exists())

    def test_cloud_plan_checks_all_offers_but_never_acquires_or_starts_work(self):
        self.args.cloud_plan = True
        with self.patches():
            self.assertEqual(self.execute(), 0)
        self.assertEqual(self.plan_calls, ["H100", "B200"])
        self.assertIsNone(self.group_request)
        self.assertEqual(self.calls, [])
        self.assertFalse(self.args.result_dir.exists())

    def test_fresh_offer_disappearance_preserves_audit_and_suggests_new_collection(self):
        self.args.collection = "initialized-before-fresh-offer-check"
        output = io.StringIO()
        with self.patches(), mock.patch.object(pipeline.cloud_group, "run", side_effect=
                pipeline.cloud_group.PreflightUnavailable("H100", "H100 became unavailable")), redirect_stderr(output):
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, ["assemble"])
        summary = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["reserved_total_usd"], 0.0)
        self.assertIn("assembly", summary)
        self.assertIn("H100 became unavailable", output.getvalue())
        self.assertIn("No instance has been created", output.getvalue())
        self.assertIn("new collection", output.getvalue().lower())
        commands = [shlex.split(line.strip()) for line in output.getvalue().splitlines()
                    if line.strip().startswith("./run.sh")]
        self.assertEqual(len(commands), 3)
        self.assertIn("--cloud-plan", commands[0])
        self.assertEqual(commands[2][1:4], ["all", "--backend", "local"])
        self.assertTrue(all("--collection" not in command for command in commands))
        retained = collection.read_manifest(collection.location(self.args.result_dir, self.args.collection))
        self.assertEqual(retained["parts"], [])

    def test_group_prepare_failure_starts_no_local_or_remote_measurements(self):
        self.prepare_error = True
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, ["assemble"])
        self.assertFalse(self.group_ready)
        summary = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["reserved_total_usd"], 3.25)
        self.assertIn("assembly", summary)

    def test_local_busy_or_identity_change_after_preparation_aborts_ready_gate(self):
        for current in ({"available": False, "uuid": "GPU-a100", "reason": "became busy"},
                        {"available": True, "uuid": "GPU-different"}):
            with self.subTest(current=current):
                self.calls.clear()
                self.group_ready = False
                # Use separate empty collections for each attempted execution.
                self.args.collection = "race-" + str(len(list(self.args.result_dir.glob("pipelines/*"))))
                with self.patches(), mock.patch.object(pipeline, "inventory", return_value=current):
                    self.assertEqual(self.execute(), 1)
                self.assertEqual(self.calls, ["assemble"])
                self.assertFalse(self.group_ready)

    def test_custom_paths_config_budget_and_stage_limits_reach_one_group(self):
        self.args.cloud_config = self.directory / "private settings.json"
        self.args.result_dir = self.directory / "custom results"
        self.args.build_dir = self.directory / "custom build"
        self.args.profile = "smoke"
        self.args.jobs, self.args.timeout = 8, 91
        self.args.cloud_stage_seconds = 700
        self.config["budget_usd"] = 6.25
        original = copy.deepcopy(self.config)
        with self.patches():
            self.assertEqual(self.execute(), 0)
        passed = self.group_request["args"]
        self.assertEqual(passed.cloud_config, self.directory / "private settings.json")
        self.assertEqual((passed.profile, passed.jobs, passed.timeout), ("smoke", 8, 91))
        self.assertEqual(passed.result_dir, self.directory / "custom results")
        self.assertEqual(passed.build_dir, self.directory / "custom build")
        self.assertEqual(self.allocated, [("H100", 2.625), ("B200", 3.625)])
        self.assertTrue(all(config["execution_seconds"] == 700 for config in self.group_request["configs"].values()))
        self.assertEqual(self.config, original)

    def test_failed_parallel_stage_preserves_other_completed_gpu_results_and_costs(self):
        self.stage_codes["H100"] = 1
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.calls, ["A100", "H100", "B200", "assemble"])
        summary = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(next(stage for stage in summary["stages"] if stage["gpu"] == "H100")["status"], "failed")
        self.assertEqual(summary["reserved_total_usd"], 3.25)

    def test_unknown_group_cleanup_fails_pipeline_and_preserves_terminal_audit(self):
        self.cleanup = False
        with self.patches():
            self.assertEqual(self.execute(), 1)
        summary = self.summary()
        self.assertFalse(summary["stages"][-1]["cloud"]["cleanup_confirmed"])
        self.assertNotEqual(summary["stages"][-1]["status"], "running")
        self.assertIn("assemble", self.calls)

    def test_missing_cloud_evidence_reserves_all_group_allowances_and_fails(self):
        self.no_evidence = True
        with self.patches():
            self.assertEqual(self.execute(), 1)
        self.assertEqual(self.summary()["status"], "failed")
        self.assertEqual(self.summary()["reserved_total_usd"], 10.0)
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
        self.interrupted = True
        with self.patches():
            self.assertEqual(self.execute(), 130)
        summary = self.summary()
        self.assertEqual(summary["status"], "interrupted")
        self.assertEqual(summary["reserved_total_usd"], 3.25)
        self.assertEqual(summary["stages"][-1]["cloud"]["reserved_total_usd"], 2.0)
        self.assertNotEqual(summary["stages"][-1]["status"], "running")
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
            "api_key_file": "/unreadable/key", "environment": {CUFFTDX_MODE_ENV: "all", CUDNN_MODE_ENV: "full",
                                                                   TABLE5_RANGE_MODE_ENV: "full"}}))
        output = io.StringIO()
        with mock.patch.object(pipeline, "load_config", side_effect=AssertionError("full config read")), \
                mock.patch.object(pipeline, "VastClient", side_effect=AssertionError("provider queried")), \
                mock.patch.object(pipeline, "inventory", side_effect=AssertionError("GPU queried")), \
                redirect_stdout(output):
            self.assertEqual(self.execute(), 0)
        self.assertIn("cuFFTDx mode: all (A100/H100/B200)", output.getvalue())
        self.assertIn("cuDNN mode: full", output.getvalue())
        self.assertIn("Table 5 range mode: full", output.getvalue())
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
                build_dir=folder / "build", device=1, jobs=2, timeout=0, cufftdx_mode="preset", cudnn_mode="full",
                table5_range_mode="full")
            process = mock.Mock(returncode=0)
            process.poll.return_value = 0
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0", CUFFTDX_MODE_ENV: "all", CUDNN_MODE_ENV: "fast",
                                             TABLE5_RANGE_MODE_ENV: "boundary"}), mock.patch.object(
                    pipeline.subprocess, "Popen", return_value=process) as launch:
                self.assertEqual(pipeline.run_local(args, ROOT, folder, mock.Mock(), gpu_uuid="GPU-a100"), 0)
                self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1,0")
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--device") + 1], "0")
            self.assertEqual(command[command.index("--backend") + 1], "local")
            self.assertEqual(command[command.index("--cufftdx-mode") + 1], "preset")
            self.assertEqual(command[command.index("--cudnn-mode") + 1], "full")
            self.assertEqual(command[command.index("--table5-range-mode") + 1], "full")
            self.assertEqual(launch.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "GPU-a100")
            self.assertEqual(launch.call_args.kwargs["env"][CUFFTDX_MODE_ENV], "preset")
            self.assertEqual(launch.call_args.kwargs["env"][CUDNN_MODE_ENV], "full")
            self.assertEqual(launch.call_args.kwargs["env"][TABLE5_RANGE_MODE_ENV], "full")

    def test_local_child_requires_a_previously_checked_uuid(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(pipeline.subprocess, "Popen") as launch:
            with self.assertRaises((ValueError, RuntimeError)):
                pipeline.run_local(SimpleNamespace(), ROOT, Path(temporary), mock.Mock())
            launch.assert_not_called()

    def test_cancelled_local_stage_does_not_launch_new_work(self):
        cancel = threading.Event()
        cancel.set()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = SimpleNamespace(profile="paper", collection="test", result_dir=folder / "result",
                build_dir=folder / "build", jobs=2, timeout=0, cufftdx_mode="preset", cudnn_mode="fast",
                table5_range_mode="boundary")
            with mock.patch.object(pipeline.subprocess, "Popen") as launch:
                self.assertEqual(pipeline.run_local(args, ROOT, folder, mock.Mock(),
                    gpu_uuid="GPU-a100", cancel_event=cancel), 130)
            launch.assert_not_called()

    def test_running_local_stage_stops_its_process_group_on_cancellation(self):
        cancel = threading.Event()
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = SimpleNamespace(profile="paper", collection="test", result_dir=folder / "result",
                build_dir=folder / "build", jobs=2, timeout=0, cufftdx_mode="preset", cudnn_mode="fast",
                table5_range_mode="boundary")
            process = mock.Mock()
            process.poll.return_value = None

            def waiting(*, timeout):
                self.assertLessEqual(timeout, 1)
                cancel.set()
                raise subprocess.TimeoutExpired("local-child", timeout)

            process.wait.side_effect = waiting
            stop = mock.Mock()
            with mock.patch.object(pipeline.subprocess, "Popen", return_value=process), redirect_stdout(io.StringIO()):
                self.assertEqual(pipeline.run_local(args, ROOT, folder, stop,
                    gpu_uuid="GPU-a100", cancel_event=cancel), 130)
            stop.assert_called_once_with(process)


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
