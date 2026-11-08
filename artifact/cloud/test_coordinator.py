"""CPU-only end-to-end lifecycle tests with a fake provider and fake SSH.

These exercise coordinator ordering, exceptional cleanup, and real immutable
collection imports without loading credentials or touching a cloud/GPU endpoint.
"""
import argparse
from contextlib import ExitStack, redirect_stdout, redirect_stderr
import copy
import io
import json
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
