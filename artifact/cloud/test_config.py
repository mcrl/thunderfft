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
        for overrides in ({"budget_usd": 10.01}, {"reserve_usd": .24},
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
        with self.assertRaises(ValueError):
            self.load({"mathdx": "yes"})

    def test_table4_measurement_controls_forward_without_local_source_paths(self):
        values = {"THUNDERFFT_TABLE4_ITERATIONS": "1024", "THUNDERFFT_TABLE4_TRIALS": "5",
                  "THUNDERFFT_TABLE4_WARMUPS": "5", "THUNDERFFT_TABLE4_WAVES": "2",
                  "THUNDERFFT_TABLE4_TIMEOUT": "120", "THUNDERFFT_TABLE4_MODE": "measure"}
        self.assertEqual(config.measurement_environment(values), values)
        for key in ("THUNDERFFT_TABLE4_SOURCE_DIR", "THUNDERFFT_TABLE4_IDEAL_DIR"):
            with self.assertRaises(ValueError):
                config.measurement_environment({key: "/local/results"})

    def test_table7_timing_only_override_is_forwarded(self):
        values = {"THUNDERFFT_TABLE7_TIMING_ONLY": "1", "THUNDERFFT_TABLE7_TRIALS": "1"}
        self.assertEqual(config.measurement_environment(values), values)

    def test_library_and_figure_workloads_are_forwarded_without_host_paths(self):
        values = {"THUNDERFFT_TABLE6_LIBRARIES": "tcfft", "THUNDERFFT_TABLE6_TRIALS": "2",
                  "THUNDERFFT_FIGURE3_LIBRARIES": "copy,cufft", "THUNDERFFT_FIGURE3_BATCH": "1024",
                  "THUNDERFFT_FIGURE4_LIBRARIES": "FlashFFTConv", "THUNDERFFT_FIGURE4_FILTERS": "3,9",
                  "THUNDERFFT_FIGURE4_VALIDATE_CHANNELS": "16", "THUNDERFFT_FIGURE5_N": "128",
                  "THUNDERFFT_FIGURE5_LIBRARIES": "ThunderFFT", "THUNDERFFT_FIGURE5_TRIALS": "1"}
        self.assertEqual(config.measurement_environment(values), values)
        for key in ("THUNDERFFT_FIGURE4_BACKENDS", "THUNDERFFT_FIGURE5_BACKENDS", "FLASHFFTCONV_PYTHON"):
            with self.assertRaises(ValueError):
                config.measurement_environment({key: "/local/path"})

    def test_dependencies_follow_selected_jobs_not_other_baselines(self):
        def dependencies(evaluation, libraries, mathdx=True):
            return config.required_dependencies([{"evaluations": [evaluation], "environment": {
                "THUNDERFFT_" + evaluation.upper() + "_LIBRARIES": libraries}}], mathdx=mathdx)
        self.assertEqual(dependencies("table6", "tcfft"), dict(mathdx=False, cudnn=False, torch_headers=False))
        self.assertEqual(dependencies("table6", "flashfftconv"), dict(mathdx=False, cudnn=False, torch_headers=True))
        self.assertEqual(dependencies("figure4", "ThunderFFT"), dict(mathdx=False, cudnn=False, torch_headers=False))
        self.assertEqual(dependencies("figure4", "all"), dict(mathdx=False, cudnn=True, torch_headers=True))
        self.assertEqual(dependencies("figure4", "auto"), dependencies("figure4", "all"))
        self.assertEqual(dependencies("figure5", "all"), dict(mathdx=True, cudnn=True, torch_headers=False))
        self.assertEqual(dependencies("figure5", "auto"), dependencies("figure5", "all"))
        self.assertEqual(dependencies("table3", "ThunderFFT"), dict(mathdx=False, cudnn=False, torch_headers=False))
        self.assertFalse(dependencies("table3", "all", False)["mathdx"])

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
