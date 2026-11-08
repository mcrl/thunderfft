"""Offline safety and API-contract tests. Never contact or create cloud resources."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.parse
from unittest import mock


SPEC = importlib.util.spec_from_file_location("thff_cloud_provider_test", Path(__file__).with_name("provider.py"))
provider = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(provider)


def offer(**overrides):
    return {"id": 12, "num_gpus": 1, "gpu_name": "A100 PCIE", "gpu_ram": 81920,
            "dph_total": .6, "storage_cost": .1, "inet_down_cost": .005,
            "inet_up_cost": .01, **overrides}


def budget(**overrides):
    return dict(disk_gb=32, startup_seconds=900, execution_seconds=600,
                download_seconds=120, cleanup_seconds=120,
                billing_delay_seconds=360, upload_gb=12, download_gb=1,
                **overrides)


class ProviderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.key = self.root / "vast_api_key"
        self.key.write_text("disposable-test-token\n")
        self.key.chmod(0o600)
        self.requests = []
        self.responses = []
        self.client = provider.VastClient(self.key, transport=self.transport)
        self.sleep_patch = mock.patch.object(provider.time, "sleep")
        self.mock_sleep = self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def tearDown(self):
        self.temp.cleanup()

    def transport(self, method, url, headers, payload, timeout):
        self.requests.append((method, url, headers, payload, timeout))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def test_credentials_fail_closed(self):
        self.key.chmod(0o644)
        with self.assertRaisesRegex(provider.ProviderError, "0600"):
            provider.read_secret_file(self.key)
        self.key.chmod(0o600)
        with self.assertRaisesRegex(provider.ProviderError, "outside"):
            provider.read_secret_file(self.key, repo_root=self.root)
        link = self.root / "key-link"
        link.symlink_to(self.key)
        with self.assertRaises(provider.ProviderError):
            provider.read_secret_file(link)
        self.key.write_text("bad\ntoken")
        with self.assertRaisesRegex(provider.ProviderError, "token"):
            provider.read_secret_file(self.key)
        self.root.chmod(0o755)
        with self.assertRaisesRegex(provider.ProviderError, "0700"):
            provider.read_secret_file(self.key)
        self.root.chmod(0o700)

    def test_credentials_check_owner_and_regular_file(self):
        with mock.patch.object(provider.os, "getuid", return_value=os.getuid() + 1):
            with self.assertRaisesRegex(provider.ProviderError, "owned"):
                provider.read_secret_file(self.key)
        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(provider.ProviderError, "regular"):
            provider.read_secret_file(fifo)

    def test_selectors_reject_other_paper_skus(self):
        self.assertTrue(provider.gpu_matches("A100", offer()))
        self.assertTrue(provider.gpu_matches("H100", offer(gpu_name="H100_PCIE")))
        self.assertTrue(provider.gpu_matches("B200", offer(gpu_name="B200", gpu_ram=196608)))
        for wrong in [offer(gpu_ram=40960), offer(gpu_name="A100 SXM4"),
                      offer(gpu_name="H100 NVL", gpu_ram=96256),
                      offer(num_gpus=8), offer(gpu_ram=None), offer(gpu_name="A100")]:
            self.assertFalse(provider.gpu_matches("A100", wrong))
        self.assertFalse(provider.gpu_matches("H100", offer(gpu_name="H100 SXM")))
        self.assertFalse(provider.gpu_matches("B200", offer(gpu_name="GB200", gpu_ram=196608)))

    def test_plan_all_costs_and_network_direction(self):
        plan = provider.plan_cost(offer(), **budget())
        self.assertEqual(plan["billed_seconds"], 2100)
        self.assertEqual(plan["active_seconds"], 1740)
        self.assertAlmostEqual(plan["compute_usd"], .35)
        self.assertAlmostEqual(plan["storage_usd"], .1 * 32 / 720 * 2100 / 3600)
        self.assertAlmostEqual(plan["network_usd"], .07)
        self.assertAlmostEqual(plan["planned_total_usd"], .35 + .07 + .25 + plan["storage_usd"])
        self.assertFalse(plan["provider_enforced_cap"])

    def test_plan_rejects_missing_prices_and_insufficient_budget(self):
        for field in ["dph_total", "storage_cost", "inet_up_cost", "inet_down_cost"]:
            for bad in [None, float("nan"), -1, True, "oops"]:
                with self.subTest(field=field, bad=bad), self.assertRaises(provider.ProviderError):
                    provider.plan_cost(offer(**{field: bad}), **budget())
        with self.assertRaisesRegex(provider.ProviderError, "exceeds budget"):
            provider.plan_cost(offer(dph_total=10), **budget())
        for key, value in [("budget_usd", 10.01), ("reserve_usd", .24),
                           ("billing_delay_seconds", 359), ("disk_gb", 0),
                           ("upload_gb", -1)]:
            args = budget()
            args[key] = value
            with self.subTest(key=key), self.assertRaises(provider.ProviderError):
                provider.plan_cost(offer(), **args)

    def test_search_is_header_auth_and_filters_skus_again(self):
        self.responses = [{"offers": [offer(), offer(gpu_name="A100 SXM4")]}]
        self.assertEqual(self.client.search_offers("A100", 32), [offer()])
        method, url, headers, payload, timeout = self.requests[0]
        self.assertEqual(method, "POST")
        self.assertEqual(url, "https://console.vast.ai/api/v0/bundles/")
        self.assertNotIn("disposable-test-token", url)
        self.assertEqual(headers["Authorization"], "Bearer disposable-test-token")
        self.assertEqual(payload["num_gpus"], {"eq": 1})
        self.assertEqual(payload["allocated_storage"], 32)
        self.assertEqual(payload["type"], "ondemand")
        self.assertEqual(payload["cuda_max_good"], {"gte": 12.9})

    def test_no_redirects(self):
        handler = provider._NoRedirect()
        self.assertIsNone(handler.redirect_request(None, None, 302, "", {}, "https://other.invalid"))

    def test_create_contract_and_no_ambiguous_retry(self):
        self.responses = [{"success": True, "new_contract": 123}]
        self.assertEqual(self.client.create_instance(12, image="nvidia/cuda:test",
                         disk_gb=32, label="thff-unique123"), 123)
        self.assertEqual(self.requests[-1][0:2], ("PUT", "https://console.vast.ai/api/v0/asks/12/"))
        self.assertTrue(self.requests[-1][3]["cancel_unavail"])
        self.assertEqual(self.requests[-1][3]["runtype"], "ssh_direc ssh_proxy")
        for failure in [TimeoutError("disposable-test-token"), {"success": True},
                        {"success": False, "msg": "disposable-test-token"}]:
            self.responses = [failure]
            count = len(self.requests)
            with self.assertRaises(provider.CreateOutcomeUnknown) as caught:
                self.client.create_instance(12, image="nvidia/cuda:test", disk_gb=32, label="unique")
            self.assertNotIn("disposable-test-token", str(caught.exception))
            self.assertEqual(len(self.requests), count + 1)

    def test_create_definitive_http_rejections_do_not_leave_ambiguous_outcome(self):
        for status in [400, 401, 403, 404, 405, 422]:
            self.responses = [provider.HTTPStatusError(status)]
            count = len(self.requests)
            with self.subTest(status=status), self.assertRaises(provider.CreateRejected):
                self.client.create_instance(12, image="nvidia/cuda:test", disk_gb=32, label="unique")
            self.assertEqual(len(self.requests), count + 1)
        for status in [410, 429, 500, 502, 503, 504]:
            self.responses = [provider.HTTPStatusError(status)]
            count = len(self.requests)
            with self.subTest(status=status), self.assertRaises(provider.CreateOutcomeUnknown):
                self.client.create_instance(12, image="nvidia/cuda:test", disk_gb=32, label="unique")
            self.assertEqual(len(self.requests), count + 1)

    def test_sanitized_errors_account_and_attach(self):
        self.responses = [RuntimeError("disposable-test-token")]
        with self.assertRaises(provider.ProviderError) as caught:
            self.client.instances()
        self.assertNotIn("disposable-test-token", str(caught.exception))
        self.responses = [{"balance": 4.5, "email": "private", "ssh_key": "private"}]
        summary = self.client.account_summary()
        self.assertEqual(set(summary), {"balance_usd", "observed_at"})
        self.assertEqual(self.requests[-1][1], "https://console.vast.ai/api/v0/users/current/")
        self.responses = [{"success": True}]
        self.client.attach_ssh_key(123, "ssh-ed25519 QUJD team-personal-comment")
        self.assertEqual(self.requests[-1][3], {"ssh_key": "ssh-ed25519 QUJD"})

    def test_destroy_acknowledgment_is_not_absence(self):
        self.responses = [{"success": True}, {"instances": [{"id": 123}], "next_token": None},
                          {"instances": [], "next_token": None}]
        self.client.destroy_instance(123)
        self.assertFalse(self.client.verify_absent(123))
        self.assertTrue(self.client.verify_absent(123))
        self.responses = [{}]
        with self.assertRaisesRegex(provider.ProviderError, "absence"):
            self.client.verify_absent(123)
        self.responses = [provider.HTTPStatusError(404)]
        self.client.destroy_instance(123)

    def test_instances_v1_reads_every_page_and_redacts_unrequested_fields(self):
        self.responses = [{"instances": [{"id": 123, "label": "first", "jupyter_token": "private"}],
                           "next_token": "second"},
                          {"instances": [{"id": 456, "ssh_host": "ssh.example", "ssh_port": 1234}],
                           "next_token": None}]
        with mock.patch.object(provider.time, "sleep"):
            rows = self.client.instances()
        self.assertEqual([row["id"] for row in rows], [123, 456])
        self.assertEqual(rows[1]["ssh_port"], 1234)
        self.assertNotIn("private", json.dumps(rows))
        first = urllib.parse.urlsplit(self.requests[0][1])
        self.assertEqual(first.path, "/api/v1/instances/")
        self.assertEqual(urllib.parse.parse_qs(first.query)["limit"], ["25"])
        self.assertIn("after_token=second", self.requests[1][1])

    def test_instances_missing_cursor_or_repeated_cursor_cannot_confirm_absence(self):
        self.responses = [{"instances": []}]
        with self.assertRaisesRegex(provider.ProviderError, "absence"):
            self.client.verify_absent(123)
        self.responses = [{"instances": [], "next_token": "same"}] * 2
        with self.assertRaisesRegex(provider.ProviderError, "pagination"):
            self.client.verify_absent(123)

    def test_instances_rate_limit_applies_between_separate_calls(self):
        self.responses = [{"instances": [], "next_token": None}] * 2
        with mock.patch.object(provider.time, "monotonic", return_value=100):
            self.client.instances()
            self.client.instances()
        self.mock_sleep.assert_called_once_with(1.05)

    def test_billing_preflight_validates_access_without_returning_rows(self):
        self.responses = [{"results": [{"description": "private"}]}]
        result = self.client.billing_preflight()
        self.assertTrue(result["billing_read_verified"])
        self.assertNotIn("private", json.dumps(result))
        self.assertIn("limit=1", self.requests[-1][1])

    def test_billing_paginates_filters_and_does_not_claim_settlement(self):
        row = {"type": "instance", "source": "instance-123", "amount": .3,
               "start": 100, "end": 200, "description": "private", "metadata": {"secret": "private"},
               "items": [{"type": "gpu", "amount": .25}]}
        other = dict(row, source="instance-999", amount=50)
        self.responses = [{"results": [other], "next_token": "next"},
                          {"results": [row], "next_token": None}, {"balance": 4.5}]
        evidence = self.client.billing_evidence(100, 200, 123)
        self.assertEqual(evidence["observed_charge_usd"], .3)
        self.assertTrue(evidence["pagination_complete"])
        self.assertFalse(evidence["settlement_verified"])
        self.assertNotIn("private", json.dumps(evidence))
        self.assertEqual(len(evidence["charge_rows"]), 1)
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.requests[0][1]).query)
        self.assertEqual(json.loads(query["select_filters"][0])["day"], {"gte": 0, "lte": 86400})
        self.assertIn("after_token=next", self.requests[1][1])
        self.assertNotIn("disposable-test-token", self.requests[1][1])

    def test_billing_rejects_cycles_and_malformed_rows(self):
        self.responses = [{"results": [], "next_token": "repeat"}] * 2
        with self.assertRaisesRegex(provider.ProviderError, "pagination"):
            self.client.billing_evidence(100, 200, 123)
        self.responses = [{"results": [{"source": "instance-123", "amount": float("nan")}]}]
        with self.assertRaisesRegex(provider.ProviderError, "amount"):
            self.client.billing_evidence(100, 200, 123)


if __name__ == "__main__":
    unittest.main()
