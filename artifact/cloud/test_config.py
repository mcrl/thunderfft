"""CPU checks for credentials-free previews and execution policy boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact.cloud import config


class ConfigurationTests(unittest.TestCase):
    def load(self, overrides):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(overrides))
            return config.load_config(path)

    def test_example_tracks_defaults(self):
        example = json.loads(Path(__file__).with_name("config.example.json").read_text())
        self.assertEqual(example, config.DEFAULTS)

    def test_rejects_over_budget_and_reduced_cleanup_reserves(self):
        for overrides in ({"budget_usd": 5.01}, {"reserve_usd": .24},
                          {"billing_delay_seconds": 359}, {"cleanup_seconds": 119},
                          {"execution_seconds": 59}, {"disk_gb": 0},
                          {"download_gb": float("inf")}, {"startup_seconds": True}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                self.load(overrides)

    def test_environment_does_not_forward_credentials_or_host_paths(self):
        for values in ({"VAST_API_KEY": "example"}, {"THUNDERFFT_CLOUD_KEY": "example"},
                       {"LD_PRELOAD": "/tmp/x"}, {"CUDACXX": "/host/nvcc"}):
            with self.assertRaises(ValueError):
                config.measurement_environment(values)
        with patch.dict(os.environ, {"VAST_API_KEY": "EXAMPLE_SENTINEL", "THUNDERFFT_TABLE3_PRECISIONS": "fp16"}):
            effective = config.inherited_environment({"environment": {}})
            self.assertNotIn("VAST_API_KEY", effective)
            self.assertEqual(effective["THUNDERFFT_TABLE3_PRECISIONS"], "fp16")

    def test_unknown_setting_and_multiline_override_rejected(self):
        with self.assertRaises(ValueError):
            self.load({"api_key": "must never be accepted"})
        with self.assertRaises(ValueError):
            self.load({"environment": {"THUNDERFFT_TABLE3_PRECISIONS": "fp16\nsecret"}})

    def test_cloud_preview_requires_no_credentials_and_creates_no_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "result"
            configuration = Path(folder) / "config.json"
            configuration.write_text(json.dumps({"api_key_file": str(Path(folder) / "absent")}))
            run = subprocess.run([str(ROOT / "run.sh"), "table3", "--backend", "vast", "--dry-run",
                                  "--result-dir", str(output), "--cloud-config", str(configuration)],
                                 text=True, capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertFalse(json.loads(run.stdout)["api_calls"])
            self.assertFalse(output.exists())

    def test_recover_cannot_be_hidden_under_preview_or_plan(self):
        for flag in ("--dry-run", "--cloud-plan"):
            run = subprocess.run([str(ROOT / "run.sh"), "--cloud-recover", "/nonexistent/session", flag],
                                 text=True, capture_output=True, timeout=10)
            self.assertEqual(run.returncode, 2)
            self.assertIn("--cloud-recover takes only", run.stderr)


if __name__ == "__main__":
    unittest.main()
