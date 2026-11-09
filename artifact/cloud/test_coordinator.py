"""CPU-only end-to-end lifecycle tests with a fake provider and fake SSH.

These exercise coordinator ordering, exceptional cleanup, and real immutable
collection imports without loading credentials or touching a cloud/GPU endpoint.
"""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import io
import json
import os
import time
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from artifact import collection
from artifact.cloud import coordinator, config, provider
from artifact.cloud import watchdog


class FakeClient:
    def __init__(self, events, *, ambiguous=False):
        self.events = events
        self.ambiguous = ambiguous
        self.rejected = False
        self.rows = []
        self.create_count = 0
        self.offers = [{"id": 12, "machine_id": 8, "num_gpus": 1,
                        "gpu_name": "A100 PCIE", "gpu_ram": 81920,
                        "dph_total": .6, "storage_cost": .1,
                        "inet_down_cost": .005, "inet_up_cost": .01}]
        self.created_offers = []

    def account_summary(self):
        self.events.append("account")
        return {"balance_usd": 10, "observed_at": 100}

    def billing_preflight(self):
        self.events.append("billing_preflight")
        return {"billing_read_verified": True, "observed_at": 100}

    def search_offers(self, gpu, disk):
        self.events.append("offers")
        return copy.deepcopy(self.offers)

    def instances(self):
        self.events.append("instances")
        return list(self.rows)

    def create_instance(self, offer_id, **kwargs):
        self.events.append("create")
        self.create_count += 1
        self.created_offers.append(offer_id)
        if self.rejected:
            raise provider.CreateRejected("Vast.ai rejected creation with HTTP 400; response body suppressed")
        self.rows = [{"id": 123, "label": kwargs["label"]}]
        if self.ambiguous:
            raise provider.CreateOutcomeUnknown("Lost acknowledgement; reconcile unique label")
        return 123

    def attach_ssh_key(self, instance_id, public):
        self.events.append("attach")

    def destroy_instance(self, instance_id):
        self.events.append("destroy")
        self.rows = [row for row in self.rows if row["id"] != instance_id]

    def billing_evidence(self, start, end, instance_id=None):
        self.events.append("billing")
        return {"instance_id": instance_id, "observed_charge_usd": .01,
                "settlement_verified": False, "account": self.account_summary()}


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.keys = self.root / "keys"
        self.keys.mkdir(mode=0o700)
        self.public = self.keys / "id_ed25519.pub"
        self.public.write_text("ssh-ed25519 QUJD fake-test-key\n")
        private = self.keys / "id_ed25519"
        private.write_text("offline fake private credential; ssh-keygen is mocked\n")
        private.chmod(0o600)
        self.settings = copy.deepcopy(config.DEFAULTS)
        self.settings.update(api_key_file=str(self.keys / "api_key"),
                             ssh_key_file=str(self.keys / "id_ed25519"),
                             ssh_public_key_file=str(self.public))
        self.args = SimpleNamespace(cloud_recover=None, assemble=False, import_result=[],
            device=0, cloud_config=None, cloud_batch=None, evaluations=[], part=None,
            collection=None, profile="smoke", gpu="A100", result_dir=self.root / "result",
            dry_run=False, cloud_plan=False, cloud_allow_dirty=False, jobs=2, timeout=600)
        self.entries = [{"id": "table2", "available": True}]
        self.registry = {"evaluations": self.entries}
        self.events = []
        self.client = FakeClient(self.events)
        self.worker_code = 0
        self.fetch_error = None
        self.downloaded = self.root / "downloaded/result"
        self.downloaded.mkdir(parents=True)
        self.parser = argparse.ArgumentParser()

    def selected(self, requested, entries, parser):
        return [row for row in entries if row["id"] in requested]

    def fake_source(self, root, destination, **kwargs):
        self.events.append("source")
        destination.write_bytes(b"fake source archive")
        return {"archive_sha256": "test", "files": {}}

    def fake_arm(self, folder, lock):
        self.events.append("watchdog")
        state = json.loads((folder / "state.json").read_text())
        self.assertFalse(state["create_attempted"])
        self.assertIsNone(state["instance_id"])
        self.assertEqual(self.client.rows, [])
        return mock.Mock(wait=mock.Mock(return_value=0))

    def fake_ssh(self, *args, **kwargs):
        instance = mock.Mock()
        def run(command, **kw):
            if command[0].endswith("/venv/bin/python"):
                self.events.append("worker")
                return subprocess.CompletedProcess(command, self.worker_code, "worker output", "")
            return subprocess.CompletedProcess(command, 0, "armed\n", "")
        instance.run.side_effect = run
        return instance

    def fake_fetch(self, *args, **kwargs):
        self.events.append("fetch")
        if self.fetch_error:
            raise self.fetch_error
        return self.downloaded

    def patches(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(coordinator, "load_config", return_value=self.settings))
        stack.enter_context(mock.patch.object(coordinator, "VastClient", return_value=self.client))
        stack.enter_context(mock.patch.object(coordinator, "SSH", side_effect=self.fake_ssh))
        stack.enter_context(mock.patch.object(coordinator, "build_source_bundle", side_effect=self.fake_source))
        stack.enter_context(mock.patch.object(coordinator, "arm_watchdog", side_effect=self.fake_arm))
        stack.enter_context(mock.patch.object(coordinator, "wait_ssh", side_effect=lambda *a, **k: self.fake_ssh()))
        stack.enter_context(mock.patch.object(coordinator, "bootstrap", side_effect=lambda *a, **k: self.events.append("bootstrap")))
        stack.enter_context(mock.patch.object(coordinator, "execute_worker", side_effect=self.fake_worker))
        stack.enter_context(mock.patch.object(coordinator, "stop_worker", side_effect=lambda *a, **k: self.events.append("stop_worker")))
        stack.enter_context(mock.patch.object(coordinator, "fetch_results", side_effect=self.fake_fetch))
        stack.enter_context(mock.patch.object(coordinator.subprocess, "run", return_value=
            subprocess.CompletedProcess(["ssh-keygen"], 0, "ssh-ed25519 QUJD\n", "")))
        stack.enter_context(mock.patch.object(coordinator.time, "sleep", return_value=None))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def fake_worker(self, *args, **kwargs):
        self.events.append("worker")
        return self.worker_code

    def run_coordinator(self):
        return coordinator.run(self.args, self.entries, self.registry, self.repo,
                               collection, self.selected, self.parser)

    def session(self):
        folders = list((self.args.result_dir / "cloud").iterdir())
        self.assertEqual(len(folders), 1)
        return folders[0]

    def test_dry_run_does_not_load_credentials_or_call_provider(self):
        self.args.dry_run = True
        with self.patches(), mock.patch.object(coordinator, "VastClient", side_effect=AssertionError("credential access")):
            self.assertEqual(self.run_coordinator(), 0)
        self.assertEqual(self.events, [])
        self.assertFalse(self.args.result_dir.exists())

    def test_plan_reads_account_and_offers_but_never_creates(self):
        self.args.cloud_plan = True
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        self.assertIn("account", self.events)
        self.assertIn("offers", self.events)
        self.assertTrue(set(self.events) <= {"account", "offers", "billing_preflight"})
        self.assertEqual(self.client.create_count, 0)
        self.assertFalse(self.args.result_dir.exists())

    def test_cufftdx_mode_resolves_config_batch_and_explicit_cli(self):
        key = config.CUFFTDX_MODE_ENV
        self.settings["environment"][key] = "all"
        with mock.patch.dict(os.environ, {key: "preset"}):
            jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual(jobs[0]["cufftdx_mode"], "all")
        self.assertEqual(jobs[0]["environment"][key], "all")
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([
            {"evaluations": ["table2"], "part": "preset", "environment": {key: "preset"}},
            {"evaluations": ["table2"], "part": "all"}]))
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["cufftdx_mode"] for job in jobs], ["preset", "all"])
        self.args.cufftdx_mode = "preset"
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["cufftdx_mode"] for job in jobs], ["preset", "preset"])
        self.assertEqual([job["environment"][key] for job in jobs], ["preset", "preset"])

    def test_invalid_batch_mode_fails_before_provider_access(self):
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([{"evaluations": ["table2"], "part": "bad",
            "environment": {config.CUFFTDX_MODE_ENV: "auto"}}]))
        with self.patches(), mock.patch.object(coordinator, "VastClient", side_effect=AssertionError("provider queried")):
            with self.assertRaisesRegex(ValueError, "preset or all"):
                self.run_coordinator()
        self.assertFalse(self.args.result_dir.exists())

    def test_readonly_plan_includes_effective_mode_for_each_job(self):
        self.args.cloud_plan = True
        self.args.cufftdx_mode = "preset"
        self.settings["environment"][config.CUFFTDX_MODE_ENV] = "all"
        output = io.StringIO()
        with self.patches(), redirect_stdout(output):
            self.assertEqual(self.run_coordinator(), 0)
        proposal, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(proposal["jobs"][0]["cufftdx_mode"], "preset")
        self.assertEqual(proposal["jobs"][0]["environment"][config.CUFFTDX_MODE_ENV], "preset")

    def test_request_retains_resolved_mode_for_worker_and_retries(self):
        self.args.cufftdx_mode = "all"
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        request = json.loads((self.session() / "request.json").read_text())
        self.assertEqual(request["jobs"][0]["cufftdx_mode"], "all")
        self.assertEqual(request["jobs"][0]["environment"][config.CUFFTDX_MODE_ENV], "all")

    def test_cudnn_modes_resolve_config_batch_and_explicit_cli(self):
        key = config.CUDNN_MODE_ENV
        self.settings["environment"][key] = "full"
        with mock.patch.dict(os.environ, {key: "fast"}):
            jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual(jobs[0]["cudnn_mode"], "full")
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([
            {"evaluations": ["table2"], "part": "fast", "environment": {key: "fast"}},
            {"evaluations": ["table2"], "part": "full"}]))
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["cudnn_mode"] for job in jobs], ["fast", "full"])
        self.args.cudnn_mode = "full"
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["cudnn_mode"] for job in jobs], ["full", "full"])
        self.assertEqual([job["environment"][key] for job in jobs], ["full", "full"])

    def test_table5_range_mode_resolves_config_batch_and_explicit_cli(self):
        key = config.TABLE5_RANGE_MODE_ENV
        self.settings["environment"][key] = "full"
        with mock.patch.dict(os.environ, {key: "boundary"}):
            jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual(jobs[0]["table5_range_mode"], "full")
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([
            {"evaluations": ["table2"], "part": "boundary", "environment": {key: "boundary"}},
            {"evaluations": ["table2"], "part": "full"}]))
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["table5_range_mode"] for job in jobs], ["boundary", "full"])
        self.args.table5_range_mode = "full"
        jobs = coordinator.make_jobs(self.args, self.entries, self.registry, self.settings, self.selected, self.parser)
        self.assertEqual([job["table5_range_mode"] for job in jobs], ["full", "full"])
        self.assertEqual([job["environment"][key] for job in jobs], ["full", "full"])

    def test_invalid_table5_range_batch_mode_fails_before_provider_access(self):
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([{"evaluations": ["table2"], "part": "bad",
            "environment": {config.TABLE5_RANGE_MODE_ENV: "all"}}]))
        with self.patches(), mock.patch.object(coordinator, "VastClient", side_effect=AssertionError("provider queried")):
            with self.assertRaisesRegex(ValueError, "boundary or full"):
                self.run_coordinator()
        self.assertFalse(self.args.result_dir.exists())

    def test_request_and_readonly_plan_retain_table5_range_mode(self):
        self.args.table5_range_mode = "full"
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        request = json.loads((self.session() / "request.json").read_text())
        self.assertEqual(request["jobs"][0]["table5_range_mode"], "full")
        self.assertEqual(request["jobs"][0]["environment"][config.TABLE5_RANGE_MODE_ENV], "full")
        self.args.cloud_plan = True
        output = io.StringIO()
        with self.patches(), redirect_stdout(output):
            self.assertEqual(self.run_coordinator(), 0)
        proposal, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(proposal["jobs"][0]["table5_range_mode"], "full")

    def test_invalid_cudnn_batch_mode_fails_before_provider_access(self):
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([{"evaluations": ["table2"], "part": "bad",
            "environment": {config.CUDNN_MODE_ENV: "all"}}]))
        with self.patches(), mock.patch.object(coordinator, "VastClient", side_effect=AssertionError("provider queried")):
            with self.assertRaisesRegex(ValueError, "fast or full"):
                self.run_coordinator()
        self.assertFalse(self.args.result_dir.exists())

    def test_cudnn_request_and_plan_record_effective_selection(self):
        self.args.cudnn_mode = "full"
        self.settings["environment"][config.CUDNN_MODE_ENV] = "fast"
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        request = json.loads((self.session() / "request.json").read_text())
        self.assertEqual(request["jobs"][0]["cudnn_mode"], "full")
        self.assertEqual(request["jobs"][0]["environment"][config.CUDNN_MODE_ENV], "full")
        self.args.cloud_plan = True
        output = io.StringIO()
        with self.patches(), redirect_stdout(output):
            self.assertEqual(self.run_coordinator(), 0)
        proposal, _ = json.JSONDecoder().raw_decode(output.getvalue())
        self.assertEqual(proposal["jobs"][0]["cudnn_mode"], "full")

    def test_watchdog_is_armed_before_create_and_cleanup_follows_fetch(self):
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        self.assertLess(self.events.index("watchdog"), self.events.index("create"))
        self.assertLess(self.events.index("source"), self.events.index("create"))
        self.assertLess(self.events.index("fetch"), self.events.index("destroy"))
        self.assertLess(self.events.index("destroy"), self.events.index("billing"))
        status = json.loads((self.session() / "status.json").read_text())
        self.assertTrue(status["cleanup_confirmed"])
        self.assertTrue(status["results_retrieved"])
        self.assertFalse(status["billing_settlement_verified"])

    def test_failed_watchdog_readiness_prevents_creation(self):
        with self.patches(), mock.patch.object(coordinator, "arm_watchdog", side_effect=RuntimeError("watchdog failed")):
            with self.assertRaisesRegex(RuntimeError, "watchdog failed"):
                self.run_coordinator()
        self.assertEqual(self.client.create_count, 0)
        self.assertNotIn("attach", self.events)

    def test_ambiguous_create_reconciles_by_label_without_rerenting(self):
        self.client.ambiguous = True
        with self.patches():
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 1)
        self.assertIn("destroy", self.events)
        self.assertNotIn("attach", self.events)
        self.assertEqual(self.client.rows, [])
        state = json.loads((self.session() / "state.json").read_text())
        self.assertEqual(state["instance_id"], 123)
        self.assertTrue(state["cleanup_confirmed"])

    def test_definitively_rejected_create_releases_cleanup_state_without_rerenting(self):
        self.client.rejected = True
        with self.patches():
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 1)
        self.assertNotIn("destroy", self.events)
        self.assertEqual(self.client.rows, [])
        state = json.loads((self.session() / "state.json").read_text())
        self.assertFalse(state["create_attempted"])
        self.assertTrue(state["finished"])
        self.assertTrue(state["cleanup_confirmed"])

    def test_worker_failure_still_fetches_and_destroys(self):
        self.worker_code = 1
        with self.patches():
            self.assertEqual(self.run_coordinator(), 1)
        self.assertIn("fetch", self.events)
        self.assertIn("destroy", self.events)
        self.assertEqual(self.client.rows, [])
        self.assertLess(self.events.index("stop_worker"), self.events.index("fetch"))

    def test_download_failure_still_destroys_and_preserves_error(self):
        self.fetch_error = OSError("simulated transfer failure")
        with self.patches():
            self.assertEqual(self.run_coordinator(), 1)
        self.assertIn("destroy", self.events)
        self.assertEqual(self.client.rows, [])
        self.assertTrue((self.session() / "download-error.json").is_file())
        self.assertFalse(json.loads((self.session() / "status.json").read_text())["results_retrieved"])

    def alternate_offer(self):
        self.client.offers.append(dict(self.client.offers[0], id=13, machine_id=9, dph_total=.61))

    def test_explicit_machine_exclusion_selects_an_alternate_without_renting(self):
        self.alternate_offer()
        self.settings["exclude_machine_ids"] = [8]
        proposal = coordinator.plan(self.client, self.settings, "A100")
        self.assertEqual(proposal["offer"]["machine_id"], 9)
        self.assertEqual(self.client.create_count, 0)

    def test_excluded_only_offer_is_availability_failure_not_price_rejection(self):
        self.settings["exclude_machine_ids"] = [8]
        with self.assertRaises(coordinator.OfferSelectionUnavailable) as caught:
            coordinator.plan(self.client, self.settings, "A100")
        error = caught.exception
        self.assertEqual(error.reason, "excluded_hosts_only")
        self.assertEqual(error.diagnostics["excluded_host_offers"], 1)
        self.assertEqual(error.diagnostics["rejected_cost_plans"], 0)
        self.assertEqual(error.diagnostics["cost_plan_errors"], [])
        self.assertIn("excluded/unavailable hosts", str(error))
        self.assertIn("rejected cost plans=0", str(error))
        self.assertEqual(self.client.create_count, 0)

    def test_empty_search_distinguishes_unavailable_hardware_from_budget(self):
        self.client.offers = []
        with self.assertRaises(coordinator.OfferSelectionUnavailable) as caught:
            coordinator.plan(self.client, self.settings, "A100")
        self.assertEqual(caught.exception.reason, "no_matching_offers")
        self.assertIn("no matching hardware offers were returned", str(caught.exception))
        self.assertEqual(caught.exception.diagnostics["rejected_cost_plans"], 0)

    def test_plan_reports_actual_cost_rejection_and_allocation(self):
        self.settings["budget_usd"] = .5
        with self.assertRaises(coordinator.OfferSelectionUnavailable) as caught:
            coordinator.plan(self.client, self.settings, "A100")
        error = caught.exception
        self.assertEqual(error.diagnostics["rejected_cost_plans"], 1)
        self.assertEqual(error.diagnostics["excluded_host_offers"], 0)
        self.assertIn("Conservative all-in plan USD", str(error))
        self.assertIn("exceeds budget USD 0.50", str(error))
        self.assertIn("allocation USD 0.50", str(error))

    def test_plan_reports_insufficient_balance_separately_from_cost_and_exclusion(self):
        with mock.patch.object(self.client, "account_summary", return_value={"balance_usd": .1}):
            with self.assertRaises(coordinator.OfferSelectionUnavailable) as caught:
                coordinator.plan(self.client, self.settings, "A100")
        error = caught.exception
        self.assertEqual(error.diagnostics["above_available_balance"], 1)
        self.assertEqual(error.diagnostics["rejected_cost_plans"], 0)
        self.assertGreater(error.diagnostics["lowest_over_balance_plan_usd"], .1)
        self.assertIn("available balance USD 0.10", str(error))

    def test_plan_preserves_missing_price_field_and_invalid_host_diagnostics(self):
        self.client.offers[0].pop("storage_cost")
        self.client.offers.append(dict(self.client.offers[0], id=13, machine_id=None))
        with self.assertRaises(coordinator.OfferSelectionUnavailable) as caught:
            coordinator.plan(self.client, self.settings, "A100")
        error = caught.exception
        self.assertEqual(error.diagnostics["invalid_host_ids"], 1)
        self.assertEqual(error.diagnostics["rejected_cost_plans"], 1)
        self.assertIn("Missing or invalid storage_cost", str(error))
        self.assertNotIn("exceeds budget", str(error))

    def test_unavailable_create_retains_original_reason_if_no_replacement_exists(self):
        proposal = coordinator.plan(self.client, self.settings, "A100")
        folder = self.root / "unavailable-create"
        folder.mkdir()
        now = time.time()
        state = {"created_at": now, "deadline": now + 1000, "label": "test-unavailable",
                 "create_attempted": False, "instance_id": None}
        rejection = provider.CreateRejected("Vast.ai returned HTTP 400 (no_such_ask)",
            code=400, reason="no_such_ask", retryable_offer=True)
        with mock.patch.object(self.client, "create_instance", side_effect=rejection) as create:
            with self.assertRaises(provider.ProviderError) as caught:
                coordinator.acquire_instance(self.client, self.settings, "A100", state, folder, proposal)
        self.assertEqual(create.call_count, 1)
        self.assertIn("no_such_ask", str(caught.exception))
        self.assertIn("was within allocation", str(caught.exception))
        self.assertIn("rejected cost plans=0", str(caught.exception))
        self.assertFalse(state["create_attempted"])
        self.assertIsNone(state["instance_id"])
        evidence = json.loads((folder / "replacement-error.json").read_text())
        self.assertEqual(evidence["reason"], "excluded_hosts_only")
        self.assertEqual(evidence["previous_create_rejection"]["reason"], "no_such_ask")
        self.assertEqual(evidence["previous_create_rejection"]["http_status"], 400)
        self.assertEqual(evidence["previous_create_rejection"]["planned_total_usd"],
                         proposal["cost"]["planned_total_usd"])
        self.assertEqual(evidence["diagnostics"]["rejected_cost_plans"], 0)

    def test_unavailable_create_can_still_select_an_alternate_host(self):
        self.alternate_offer()
        proposal = coordinator.plan(self.client, self.settings, "A100")
        folder = self.root / "alternate-create"
        folder.mkdir()
        now = time.time()
        state = {"created_at": now, "deadline": now + 1000, "label": "test-alternate",
                 "create_attempted": False, "instance_id": None}
        rejection = provider.CreateRejected("Vast.ai returned HTTP 400 (no_such_ask)",
            code=400, reason="no_such_ask", retryable_offer=True)
        with mock.patch.object(self.client, "create_instance", side_effect=[rejection, 456]) as create:
            coordinator.acquire_instance(self.client, self.settings, "A100", state, folder, proposal)
        self.assertEqual([call.args[0] for call in create.call_args_list], [12, 13])
        self.assertEqual(proposal["offer"]["machine_id"], 9)
        self.assertEqual(state["instance_id"], 456)
        self.assertFalse((folder / "replacement-error.json").exists())

    def test_startup_retry_destroys_before_rearming_and_preserves_one_budget(self):
        self.alternate_offer()
        with self.patches(), mock.patch.object(coordinator, "wait_ssh", side_effect=[
                coordinator.StartupUnavailable("public-key authentication unavailable"), self.fake_ssh()]):
            self.assertEqual(self.run_coordinator(), 0)
        self.assertEqual(self.client.created_offers, [12, 13])
        self.assertEqual([event for event in self.events if event in ("watchdog", "create", "destroy")],
                         ["watchdog", "create", "destroy", "watchdog", "create", "destroy"])
        summary = self.args.cloud_execution_summary
        first, second = summary["attempts"]
        self.assertTrue(summary["cleanup_confirmed"])
        self.assertTrue(first["startup_retryable"])
        self.assertFalse(second["startup_retryable"])
        self.assertGreater(first["reserved_total_usd"], self.settings["reserve_usd"])
        self.assertEqual(summary["reserved_total_usd"],
                         first["reserved_total_usd"] + second["reserved_total_usd"])
        retry_plan = json.loads((Path(second["folder"]) / "plan.json").read_text())
        self.assertAlmostEqual(retry_plan["cost"]["budget_usd"],
                               self.settings["budget_usd"] - first["reserved_total_usd"])
        for attempt in summary["attempts"]:
            self.assertTrue((Path(attempt["folder"]) / "cost.json").exists())
        self.assertEqual(json.loads((Path(summary["audit_folder"]) / "execution.json").read_text()), summary)

    def test_startup_retry_is_refused_when_remaining_budget_cannot_cover_new_plan(self):
        self.alternate_offer()
        self.settings["budget_usd"] = 1.3
        with self.patches(), mock.patch.object(coordinator, "wait_ssh", side_effect=
                coordinator.StartupUnavailable("startup deadline")):
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 1)
        self.assertEqual(self.client.rows, [])
        self.assertIn("retry_stopped", self.args.cloud_execution_summary)

    def test_startup_retry_never_rents_with_unconfirmed_cleanup(self):
        self.alternate_offer()
        with self.patches(), mock.patch.object(coordinator, "wait_ssh", side_effect=
                coordinator.StartupUnavailable("startup deadline")), mock.patch.object(
                coordinator, "cleanup_and_bill", return_value=False):
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 1)
        self.assertFalse(self.args.cloud_execution_summary["cleanup_confirmed"])
        self.assertEqual(len(self.args.cloud_execution_summary["attempts"]), 1)

    def test_startup_attempt_limit_excludes_each_failed_machine(self):
        self.alternate_offer()
        self.client.offers.append(dict(self.client.offers[0], id=14, machine_id=10))
        with self.patches(), mock.patch.object(coordinator, "wait_ssh", side_effect=
                coordinator.StartupUnavailable("startup deadline")):
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 2)
        self.assertEqual(len(set(self.client.created_offers)), 2)
        self.assertEqual(self.client.rows, [])

    def test_bootstrap_and_worker_failures_are_not_startup_retried(self):
        self.alternate_offer()
        with self.patches(), mock.patch.object(coordinator, "bootstrap", side_effect=TimeoutError("package setup")):
            self.assertEqual(self.run_coordinator(), 1)
        self.assertEqual(self.client.create_count, 1)
        self.assertFalse(self.args.cloud_execution_summary["attempts"][0]["startup_retryable"])

    def test_cost_bound_includes_lifetime_delay_network_and_larger_observed_charge(self):
        folder = self.root / "cost-evidence"
        folder.mkdir()
        (folder / "cleanup.json").write_text(json.dumps({"confirmed": True, "observed_at": 200}))
        proposal = coordinator.plan(self.client, self.settings, "A100")
        state = {"created_at": 100, "cleanup_confirmed": True, "create_attempted": True, "instance_id": 123}
        evidence = coordinator.attempt_cost(proposal, state, folder)
        cost = proposal["cost"]
        expected = ((cost["compute_usd_per_hour"] + cost["storage_usd_per_hour"]) * (100 + 360) / 3600
                    + cost["network_usd"] + cost["reserve_usd"])
        self.assertAlmostEqual(evidence["reserved_total_usd"], expected)
        (folder / "billing.json").write_text(json.dumps({"samples": [{"observed_charge_usd": 5}]}))
        self.assertEqual(coordinator.attempt_cost(proposal, state, folder)["reserved_total_usd"], 5.25)

    def test_publickey_failure_is_bounded_without_waiting_full_startup_timeout(self):
        self.client.rows = [{"id": 123, "ssh_host": "example.invalid", "ssh_port": 2222,
                             "actual_status": "running"}]
        ssh = mock.Mock()
        ssh.run.side_effect = subprocess.CalledProcessError(255, ["ssh"], stderr=
                "untrusted-host-message: Permission denied (publickey).")
        clock = [0]
        with mock.patch.object(coordinator, "SSH", return_value=ssh), mock.patch.object(
                coordinator.time, "time", side_effect=lambda: clock[0]), mock.patch.object(
                coordinator.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            with self.assertRaisesRegex(coordinator.StartupUnavailable, "public-key"):
                coordinator.wait_ssh(self.client, {"instance_id": 123}, self.settings, self.root, self.repo, 900)
        self.assertEqual(clock[0], 90)
        evidence = (self.root / "ssh-readiness.json").read_text()
        self.assertTrue(json.loads(evidence)["publickey_rejected"])
        self.assertNotIn("untrusted-host-message", evidence)

    def test_waiting_for_an_endpoint_is_bounded_by_startup_allowance(self):
        clock = [0]
        with mock.patch.object(coordinator.time, "time", side_effect=lambda: clock[0]), mock.patch.object(
                coordinator.time, "sleep", side_effect=lambda seconds: clock.__setitem__(0, clock[0] + seconds)):
            with self.assertRaisesRegex(coordinator.StartupUnavailable, "deadline"):
                coordinator.wait_ssh(self.client, {"instance_id": 123}, self.settings, self.root, self.repo, 20)
        self.assertEqual(clock[0], 20)

    def fixture_part(self, evaluation, label):
        source = self.root / ("fixture-" + label)
        source.mkdir()
        (source / "status.json").write_text(json.dumps({"evaluation": evaluation,
            "profile": "smoke", "status": "completed", "exit_code": 0}))
        (source / "data.csv").write_text("key,value\na,1\n")
        (source / "table.csv").write_text("key,value\na,1\n")
        (source / "metadata.json").write_text('{"origin":"synthetic CPU fixture"}\n')
        remote = collection.location(self.downloaded, "campaign")
        return collection.add_part(remote, evaluation, label, source, "smoke")

    def test_multipart_one_rental_automatically_imports_real_collection_evidence(self):
        self.args.collection = "campaign"
        self.registry["evaluations"] = [{"id": "table2", "available": True},
                                        {"id": "validation", "available": True}]
        self.args.cloud_batch = self.root / "batch.json"
        self.args.cloud_batch.write_text(json.dumps([
            {"evaluations": ["table2"], "part": "model"},
            {"evaluations": ["validation"], "part": "correctness"}]))
        self.fixture_part("table2", "model")
        self.fixture_part("validation", "correctness")
        with self.patches():
            self.assertEqual(self.run_coordinator(), 0)
        self.assertEqual(self.client.create_count, 1)
        manifest = collection.read_manifest(collection.location(self.args.result_dir, "campaign"))
        self.assertEqual({(p["evaluation"], p["part"]) for p in manifest["parts"]},
                         {("table2", "model"), ("validation", "correctness")})
        for record in manifest["parts"]:
            copied = collection.location(self.args.result_dir, "campaign") / record["path"]
            self.assertEqual(collection.file_hashes(copied), record["files_sha256"])
        request = json.loads((self.session() / "request.json").read_text())
        self.assertEqual(len(request["jobs"]), 2)

    def test_tampered_downloaded_part_is_not_imported(self):
        self.args.collection = "campaign"
        record = self.fixture_part("table2", "model")
        source = collection.location(self.downloaded, "campaign") / record["path"]
        (source / "data.csv").write_text("tampered\n")
        with self.assertRaisesRegex(ValueError, "hashes"):
            coordinator.register_results(self.args, [{"evaluations": ["table2"], "part": "model"}],
                                         self.downloaded, collection)
        self.assertIsNone(collection.read_manifest(collection.location(self.args.result_dir, "campaign")))

    def test_unrequested_remote_part_is_not_imported(self):
        self.args.collection = "campaign"
        self.fixture_part("table2", "model")
        with self.assertRaisesRegex(ValueError, "unexpected"):
            coordinator.register_results(self.args, [{"evaluations": ["table2"], "part": "requested"}],
                                         self.downloaded, collection)
        self.assertIsNone(collection.read_manifest(collection.location(self.args.result_dir, "campaign")))

    def test_ambiguous_absence_is_not_confirmed_and_unrelated_instances_survive(self):
        state = {"label": "thunderfft-ae-ours", "instance_id": None, "create_attempted": True}
        self.client.rows = [{"id": 999, "label": "someone-else"}]
        cleanup = watchdog.destroy_owned(self.client, state)
        self.assertTrue(cleanup["absent"])
        self.assertFalse(cleanup["confirmed"])
        self.assertNotIn("destroy", self.events)
        self.client.rows.append({"id": 123, "label": state["label"]})
        cleanup = watchdog.destroy_owned(self.client, state)
        self.assertTrue(cleanup["confirmed"])
        self.assertEqual(state["instance_id"], 123)
        self.assertEqual(self.client.rows, [{"id": 999, "label": "someone-else"}])

    def test_discovered_id_is_retained_when_delete_request_fails(self):
        state = {"label": "thunderfft-ae-ours", "instance_id": None, "create_attempted": True}
        self.client.rows = [{"id": 123, "label": state["label"]}]
        with mock.patch.object(self.client, "destroy_instance", side_effect=provider.ProviderError("transport failure")):
            with self.assertRaises(provider.ProviderError):
                watchdog.destroy_owned(self.client, state)
        self.assertEqual(state["instance_id"], 123)

    def test_recovery_uses_watchdog_discovered_id_after_ambiguous_creation(self):
        folder = self.root / "recovery"
        folder.mkdir()
        (folder / "state.json").write_text(json.dumps({"label": "thunderfft-ae-ours",
            "instance_id": None, "create_attempted": True, "created_at": 100,
            "api_key_file": str(self.keys / "api_key"), "cleanup_confirmed": False}))
        (folder / "watchdog-cleanup.json").write_text(json.dumps({"destroyed_ids": [123], "confirmed": True}))
        with self.patches():
            self.assertEqual(coordinator.recover(folder), 0)
        self.assertEqual(self.client.create_count, 0)
        state = json.loads((folder / "state.json").read_text())
        self.assertEqual(state["instance_id"], 123)
        self.assertTrue(state["cleanup_confirmed"])
        self.assertTrue((folder / "billing.json").is_file())



class WorkerProgressTests(unittest.TestCase):
    def snapshot(self):
        return {"schema_version": 1, "completed": 0, "total": 2,
                "evaluation": "table3", "status": "running", "updated_at": 123.0,
                "detail": {"schema_version": 1, "done": 36, "total": 50,
                           "unit": "cases", "detail": "cuFFTDx", "status": "running",
                           "updated_at": 123.0}}

    def reply(self, value):
        return subprocess.CompletedProcess([], 0, value, "")

    def test_progress_and_terminal_exit_use_one_ssh_call_per_poll(self):
        snapshot = self.snapshot()
        ssh = mock.Mock()
        ssh.run.side_effect = [self.reply(""),
            self.reply(json.dumps({"exit": None, "progress": snapshot})),
            self.reply(json.dumps({"exit": {"exit_code": 2}, "progress": snapshot}))]
        received = []
        with mock.patch.object(coordinator.time, "sleep") as delay:
            code = coordinator.execute_worker(ssh, "/workspace/fixture", {"execution_seconds": 30},
                                              progress_callback=received.append)
        self.assertEqual(code, 2)
        self.assertEqual(received, [snapshot, snapshot])
        self.assertEqual(ssh.run.call_count, 3)  # launch + two combined polls
        delay.assert_called_once()

    def test_legacy_worker_exit_poll_is_unchanged_without_progress_callback(self):
        ssh = mock.Mock()
        ssh.run.side_effect = [self.reply(""), self.reply('{"exit_code": 7}')]
        self.assertEqual(coordinator.execute_worker(ssh, "/workspace/fixture", {"execution_seconds": 30}), 7)
        self.assertIn("cat /workspace/fixture/worker.exit.json", ssh.run.call_args.args[0][2])

    def test_invalid_progress_is_ignored_but_worker_failure_is_returned(self):
        for progress in (None, {"completed": "36/50"}, {**self.snapshot(), "completed": 99}):
            with self.subTest(progress=progress):
                ssh, callback = mock.Mock(), mock.Mock()
                ssh.run.side_effect = [self.reply(""), self.reply(json.dumps(
                    {"exit": {"exit_code": 7}, "progress": progress}))]
                self.assertEqual(coordinator.execute_worker(ssh, "/workspace/fixture",
                    {"execution_seconds": 30}, progress_callback=callback), 7)
                callback.assert_not_called()

    def test_progress_callback_failure_keeps_polling_and_returns_terminal_exit(self):
        ssh = mock.Mock()
        ssh.run.side_effect = [self.reply(""),
            self.reply(json.dumps({"exit": None, "progress": self.snapshot()})),
            self.reply(json.dumps({"exit": {"exit_code": 0}, "progress": self.snapshot()}))]
        callback = mock.Mock(side_effect=OSError("display unavailable"))
        with mock.patch.object(coordinator.time, "sleep"):
            self.assertEqual(coordinator.execute_worker(ssh, "/workspace/fixture",
                {"execution_seconds": 30}, progress_callback=callback), 0)
        callback.assert_called_once()
        self.assertEqual(ssh.run.call_count, 3)
        self.assertTrue(all(call.args[0][0].endswith("/venv/bin/python")
                            for call in ssh.run.call_args_list[1:]))

    def test_malformed_terminal_evidence_is_not_ignored_as_telemetry(self):
        for envelope in ({"progress": self.snapshot()},
                         {"exit": {"exit_code": "invalid"}, "progress": None},
                         {"exit": {}, "progress": None}):
            with self.subTest(envelope=envelope):
                ssh = mock.Mock()
                ssh.run.side_effect = [self.reply(""), self.reply(json.dumps(envelope))]
                with self.assertRaises((ValueError, KeyError)):
                    coordinator.execute_worker(ssh, "/workspace/fixture", {"execution_seconds": 30},
                                               progress_callback=lambda value: None)

    def test_remote_transport_cancellation_is_not_swallowed(self):
        ssh = mock.Mock()
        ssh.run.side_effect = [self.reply(""), InterruptedError("cancelled transport")]
        with self.assertRaisesRegex(InterruptedError, "cancelled transport"):
            coordinator.execute_worker(ssh, "/workspace/fixture", {"execution_seconds": 30},
                                       progress_callback=lambda value: None)

    def test_real_remote_poll_script_ignores_bad_progress_and_bounds_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "source").symlink_to(Path(__file__).resolve().parents[2], target_is_directory=True)
            (root / "worker.exit.json").write_text('{"exit_code": 0}')
            for payload in (json.dumps(self.snapshot()), "{bad JSON", "x" * 70000):
                with self.subTest(size=len(payload)):
                    (root / "progress.json").write_text(payload)
                    responses = []
                    def run(command, **kwargs):
                        if command[0] == "bash":
                            return self.reply("")
                        response = subprocess.run([sys.executable, *command[1:]], text=True,
                                                  capture_output=True, check=True, timeout=5)
                        responses.append(response.stdout)
                        return response
                    received = []
                    code = coordinator.execute_worker(SimpleNamespace(run=run), str(root),
                        {"execution_seconds": 30}, progress_callback=received.append)
                    self.assertEqual(code, 0)
                    self.assertLess(len(responses[0].encode()), 65536)
                    self.assertEqual(len(received), int(payload.startswith('{"schema_version"')))


class DependencyBootstrapTests(unittest.TestCase):
    def test_bootstrap_scripts_match_cuda_major_and_pin_download_hashes(self):
        from artifact.cloud.config import CUDNN_ARCHIVES
        script = "\n".join(coordinator.cudnn_script())
        for major, (url, digest) in CUDNN_ARCHIVES.items():
            self.assertIn(major + ") url=", script)
            self.assertIn(url, script)
            self.assertIn(digest, script)
        self.assertIn("sha256sum -c -", script)
        self.assertIn("exit 1", script)
        completed = subprocess.run(["bash", "-n"], input="set -euo pipefail\n" + script,
                                   text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_bootstrap_only_installs_requested_dependencies(self):
        ssh = mock.Mock()
        coordinator.bootstrap(ssh, "/workspace/test run", time.time() + 900,
                              mathdx=False, cudnn=True, torch_headers=True)
        script = ssh.run.call_args.args[0][2]
        self.assertIn("cd -- '/workspace/test run'", script)
        self.assertIn("cudnn.tar.xz", script)
        self.assertIn("torch==2.8.0", script)
        self.assertIn("https://download.pytorch.org/whl/cpu", script)
        self.assertNotIn("mathdx.tar.gz", script)
        self.assertIn("python3-dev", script)
        self.assertIn("patch", script)
        completed = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

if __name__ == "__main__":
    unittest.main()
