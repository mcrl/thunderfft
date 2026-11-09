#!/usr/bin/env python3
"""Remote worker lifecycle tests using local fixtures and fake GPU probes."""
import hashlib
import importlib.util
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("cloud_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class InventoryTests(unittest.TestCase):
    def test_exact_paper_sku_and_memory(self):
        for gpu, name, memory in (("A100", "NVIDIA A100-PCIE-80GB", 81920),
                                  ("A100", "NVIDIA A100 80GB PCIe", 81052),
                                  ("H100", "NVIDIA H100 PCIe", 81559),
                                  ("B200", "NVIDIA B200", 183359)):
            with self.subTest(gpu=gpu, name=name):
                self.assertEqual(worker.validate_gpu_inventory(f"GPU-test, {name}, {memory} MiB, 580.1\n", gpu)["name"], name)

    def test_wrong_family_sxm_nvl_memory_multiple_devices_rejected(self):
        for gpu, name, memory in (("A100", "NVIDIA A100-SXM4-80GB", 81920),
                                  ("A100", "NVIDIA A100-PCIE-40GB", 40960),
                                  ("H100", "NVIDIA H100 NVL", 95830),
                                  ("H100", "NVIDIA H100 SXM", 81559),
                                  ("B200", "NVIDIA GB200", 183359)):
            with self.subTest(gpu=gpu, name=name), self.assertRaises(ValueError):
                worker.validate_gpu_inventory(f"GPU-test, {name}, {memory} MiB, 580.1\n", gpu)
        with self.assertRaises(ValueError):
            worker.validate_gpu_inventory("GPU-test,NVIDIA B200,183359 MiB,580\n" * 2, "B200")

    def test_provider_credentials_removed(self):
        self.assertEqual(worker.safe_environment({"PATH": "/bin", "CONTAINER_API_KEY": "fixture",
                          "VAST_API_KEY": "fixture", "AUTH_TOKEN": "fixture", "PASSWORD": "fixture"}), {"PATH": "/bin"})


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.parent = Path(self.tmp.name)
        self.root = self.parent / "source"
        self.root.mkdir()
        script = self.root / "run.sh"
        script.write_text("#!/bin/bash\nset -eu\nmkdir -p result/fixture\nprintf 'measured,fixture\\n' > result/fixture/data.csv\n")
        manifest = {"schema_version": 1, "git": {"head": "fixture", "dirty": False},
                    "files": {"run.sh": {"sha256": hashlib.sha256(script.read_bytes()).hexdigest()}}}
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request = {"gpu": "A100", "arch": "80", "execution_seconds": 60, "max_result_bytes": 1000000,
                        "build_jobs": 2, "task_timeout": 0, "collection": None, "profile": "smoke",
                        "jobs": [{"part": "test-part", "evaluations": ["fixture"], "environment": {}}]}
        self.request_file = self.parent / "request.json"
        self.write_request()
        self.probe_environments = []

    def write_request(self):
        self.request_file.write_text(json.dumps(self.request))

    def probe(self, argv, **kwargs):
        self.probe_environments.append(kwargs["env"])
        stdout = "GPU-test,NVIDIA A100-PCIE-80GB,81920 MiB,580.1\n" if argv[0] == "nvidia-smi" else "fixture version\n"
        return subprocess.CompletedProcess(argv, 0, stdout, "")

    def run_fixture(self):
        with patch.object(worker.subprocess, "run", side_effect=self.probe):
            return worker.main([str(self.root), str(self.request_file)])

    def test_success_returns_archive_manifest_and_terminal_marker(self):
        with patch.dict(os.environ, {"CONTAINER_API_KEY": "not-for-jobs"}):
            self.assertEqual(self.run_fixture(), 0)
        self.assertTrue(all("CONTAINER_API_KEY" not in env for env in self.probe_environments))
        marker = json.loads((self.parent / "worker.exit.json").read_text())
        self.assertEqual(marker["exit_code"], 0)
        self.assertTrue(marker["result_archive_ready"])
        self.assertEqual(int((self.parent / "worker.pid").read_text()), os.getpid())
        with tarfile.open(self.parent / "results.tar.gz") as archive:
            self.assertIn("result/fixture/data.csv", archive.getnames())
            self.assertIn("result/cloud-worker/source.json", archive.getnames())
            self.assertTrue(all(path == "result" or path.startswith("result/") for path in archive.getnames()))
        states = json.loads((self.root / "result/cloud-worker/jobs.json").read_text())
        self.assertEqual(states[0]["status"], "completed")

    def test_requested_mathdx_sets_root_or_fails_before_jobs(self):
        script = self.root / "run.sh"
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p result/fixture\nprintf "%s" "$MATHDX_ROOT" > result/fixture/data.csv\n')
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request["mathdx"] = True
        self.write_request()
        self.assertEqual(self.run_fixture(), 1)
        self.assertFalse((self.root / "result/fixture").exists())
        header = self.parent / "mathdx/nvidia/mathdx/25.06/include/cufftdx.hpp"
        header.parent.mkdir(parents=True)
        header.write_text("")
        self.assertEqual(self.run_fixture(), 0)
        self.assertEqual((self.root / "result/fixture/data.csv").read_text(), str(self.parent / "mathdx/nvidia/mathdx/25.06"))

    def test_all_worker_job_explicitly_selects_local_backend_without_recursive_rental(self):
        self.request["jobs"][0]["evaluations"] = ["all"]
        self.write_request()
        self.assertEqual(self.run_fixture(), 0)
        record = json.loads((self.root / "result/cloud-worker/jobs.log").read_text().splitlines()[0])
        command = record["command"]
        self.assertIn("all", command)
        self.assertEqual(command[command.index("--backend") + 1], "local")
        self.assertEqual(command[command.index("--cufftdx-mode") + 1], "preset")
        self.assertEqual(command[command.index("--cudnn-mode") + 1], "fast")
        self.assertEqual(command[command.index("--table5-range-mode") + 1], "boundary")

    def test_each_job_gets_aggregate_progress_path_and_resets_prior_job(self):
        self.request["jobs"].append({"part": "second-part", "evaluations": ["fixture", "other"],
                                     "environment": {}})
        self.write_request()
        snapshots = []
        def launch(command, **kwargs):
            path = Path(kwargs["env"]["THUNDERFFT_RUN_PROGRESS_FILE"])
            self.assertEqual(path, self.parent / "progress.json")
            snapshots.append(json.loads(path.read_text()))
            path.write_text('{"stale": "prior completed job"}')
            process = Mock()
            process.wait.return_value = 0
            return process
        with patch.object(worker.subprocess, "Popen", side_effect=launch):
            self.assertEqual(self.run_fixture(), 0)
        self.assertEqual([snapshot["total"] for snapshot in snapshots], [1, 2])
        self.assertTrue(all(snapshot["completed"] == 0 and snapshot["detail"] is None
                            and snapshot["evaluation"] == "" for snapshot in snapshots))

    def test_progress_write_failure_does_not_fail_measurements(self):
        original = worker.atomic_json
        def write(path, value):
            if path.name == "progress.json":
                raise OSError("optional telemetry unavailable")
            return original(path, value)
        with patch.object(worker, "atomic_json", side_effect=write):
            self.assertEqual(self.run_fixture(), 0)
        self.assertTrue((self.root / "result/fixture/data.csv").is_file())

    def test_worker_passes_requested_mode_to_cli_environment_and_job_metadata(self):
        key = "THUNDERFFT_TABLE3_CUFFTDX_MODE"
        script = self.root / "run.sh"
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p result/fixture\n'
                          'printf "%s" "$THUNDERFFT_TABLE3_CUFFTDX_MODE" > result/fixture/data.csv\n')
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request["jobs"][0].update(cufftdx_mode="all", environment={key: "all"})
        self.write_request()
        with patch.dict(os.environ, {key: "preset"}):
            self.assertEqual(self.run_fixture(), 0)
            self.assertEqual(os.environ[key], "preset")
        self.assertEqual((self.root / "result/fixture/data.csv").read_text(), "all")
        command = json.loads((self.root / "result/cloud-worker/jobs.log").read_text().splitlines()[0])["command"]
        self.assertEqual(command[command.index("--cufftdx-mode") + 1], "all")
        self.assertEqual(json.loads((self.root / "result/cloud-worker/jobs.json").read_text())[0]["cufftdx_mode"], "all")

    def test_worker_rejects_invalid_or_conflicting_mode_before_job_launch(self):
        for mode, environment in (("invalid", {}), ("all", {"THUNDERFFT_TABLE3_CUFFTDX_MODE": "preset"})):
            with self.subTest(mode=mode, environment=environment):
                self.request["jobs"][0].update(cufftdx_mode=mode, environment=environment)
                self.write_request()
                with patch.object(worker.subprocess, "Popen") as launch:
                    self.assertEqual(self.run_fixture(), 1)
                launch.assert_not_called()

    def test_worker_passes_cudnn_mode_to_cli_environment_and_metadata(self):
        key = "THUNDERFFT_CUDNN_MODE"
        script = self.root / "run.sh"
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p result/fixture\n'
                          'printf "%s" "$THUNDERFFT_CUDNN_MODE" > result/fixture/data.csv\n')
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request["jobs"][0].update(cudnn_mode="full", environment={key: "full"})
        self.write_request()
        with patch.dict(os.environ, {key: "fast"}):
            self.assertEqual(self.run_fixture(), 0)
            self.assertEqual(os.environ[key], "fast")
        self.assertEqual((self.root / "result/fixture/data.csv").read_text(), "full")
        command = json.loads((self.root / "result/cloud-worker/jobs.log").read_text().splitlines()[0])["command"]
        self.assertEqual(command[command.index("--cudnn-mode") + 1], "full")
        self.assertEqual(json.loads((self.root / "result/cloud-worker/jobs.json").read_text())[0]["cudnn_mode"], "full")

    def test_worker_rejects_invalid_or_conflicting_cudnn_mode_before_launch(self):
        for mode, environment in (("all", {}), ("full", {"THUNDERFFT_CUDNN_MODE": "fast"})):
            with self.subTest(mode=mode, environment=environment):
                self.request["jobs"][0].update(cudnn_mode=mode, environment=environment)
                self.write_request()
                with patch.object(worker.subprocess, "Popen") as launch:
                    self.assertEqual(self.run_fixture(), 1)
                launch.assert_not_called()

    def test_worker_passes_table5_range_mode_to_cli_environment_and_metadata(self):
        key = "THUNDERFFT_TABLE5_RANGE_MODE"
        script = self.root / "run.sh"
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p result/fixture\n'
                          'printf "%s" "$THUNDERFFT_TABLE5_RANGE_MODE" > result/fixture/data.csv\n')
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request["jobs"][0].update(table5_range_mode="full", environment={key: "full"})
        self.write_request()
        with patch.dict(os.environ, {key: "boundary"}):
            self.assertEqual(self.run_fixture(), 0)
            self.assertEqual(os.environ[key], "boundary")
        self.assertEqual((self.root / "result/fixture/data.csv").read_text(), "full")
        command = json.loads((self.root / "result/cloud-worker/jobs.log").read_text().splitlines()[0])["command"]
        self.assertEqual(command[command.index("--table5-range-mode") + 1], "full")
        self.assertEqual(json.loads((self.root / "result/cloud-worker/jobs.json").read_text())[0]["table5_range_mode"], "full")

    def test_worker_rejects_invalid_or_conflicting_table5_range_mode_before_launch(self):
        for mode, environment in (("all", {}), ("full", {"THUNDERFFT_TABLE5_RANGE_MODE": "boundary"})):
            with self.subTest(mode=mode, environment=environment):
                self.request["jobs"][0].update(table5_range_mode=mode, environment=environment)
                self.write_request()
                with patch.object(worker.subprocess, "Popen") as launch:
                    self.assertEqual(self.run_fixture(), 1)
                launch.assert_not_called()

    def test_source_tampering_prevents_any_job_and_preserves_error(self):
        (self.root / "run.sh").write_text("changed\n")
        self.assertEqual(self.run_fixture(), 1)
        self.assertFalse((self.root / "result/fixture").exists())
        self.assertIn("hash mismatch", (self.root / "result/cloud-worker/error.json").read_text())
        self.assertTrue(json.loads((self.parent / "worker.exit.json").read_text())["result_archive_ready"])

    def test_cudnn_and_flash_headers_environment_is_checked_before_jobs(self):
        script = self.root / "run.sh"
        script.write_text('#!/bin/bash\nset -eu\nmkdir -p result/fixture\nprintf "%s\\n%s\\n%s" "$CUDNN_HOME" "$LD_LIBRARY_PATH" "$FLASHFFTCONV_PYTHON" > result/fixture/data.csv\n')
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.request.update(cudnn=True, torch_headers=True)
        self.write_request()
        self.assertEqual(self.run_fixture(), 1)
        self.assertFalse((self.root / "result/fixture").exists())
        cudnn = self.parent / "cudnn"
        for name in ("include/cudnn.h", "lib/libcudnn.so"):
            path = cudnn / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        self.assertEqual(self.run_fixture(), 0)
        lines = (self.root / "result/fixture/data.csv").read_text().splitlines()
        self.assertEqual(lines[0], str(cudnn))
        self.assertEqual(lines[1].split(os.pathsep)[0], str(cudnn / "lib"))
        self.assertEqual(lines[2], sys.executable)

    def test_interrupt_stops_runner_records_terminal_state_and_packs(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [KeyboardInterrupt(), 130]
        with patch.object(worker.subprocess, "Popen", return_value=process):
            self.assertEqual(self.run_fixture(), 130)
        process.send_signal.assert_called_once_with(signal.SIGTERM)
        state = json.loads((self.root / "result/cloud-worker/jobs.json").read_text())[0]
        self.assertEqual((state["status"], state["exit_code"]), ("interrupted", 130))
        self.assertTrue(json.loads((self.parent / "worker.exit.json").read_text())["result_archive_ready"])

    def test_job_deadline_stops_runner_and_reserves_pack_time(self):
        process = Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("fixture", 35), 130]
        with patch.object(worker.subprocess, "Popen", return_value=process):
            self.assertEqual(self.run_fixture(), 124)
        timeout = process.wait.call_args_list[0].kwargs["timeout"]
        self.assertLessEqual(timeout, 35)
        self.assertGreater(timeout, 30)
        self.assertTrue(json.loads((self.parent / "worker.exit.json").read_text())["result_archive_ready"])

    def test_probe_setup_deadline_produces_terminal_timeout(self):
        with patch.object(worker.subprocess, "run", side_effect=subprocess.TimeoutExpired("probe", 20)):
            self.assertEqual(worker.main([str(self.root), str(self.request_file)]), 124)
        self.assertFalse((self.root / "result/fixture").exists())
        self.assertTrue((self.parent / "worker.exit.json").is_file())

    def test_failed_exit_code_and_unexpected_probe_error_still_finalize(self):
        (self.root / "run.sh").write_text("#!/bin/bash\nexit 7\n")
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256((self.root / "run.sh").read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        self.assertEqual(self.run_fixture(), 1)
        self.assertEqual(json.loads((self.root / "result/cloud-worker/jobs.json").read_text())[0]["exit_code"], 7)
        with patch.object(worker.subprocess, "run", side_effect=FileNotFoundError("fixture missing tool")):
            self.assertEqual(worker.main([str(self.root), str(self.request_file)]), 1)
        self.assertTrue((self.parent / "worker.exit.json").is_file())

    def test_too_short_execution_rejected_and_signal_handlers_restored(self):
        self.request["execution_seconds"] = 59
        self.write_request()
        before = signal.getsignal(signal.SIGTERM)
        self.assertEqual(self.run_fixture(), 1)
        self.assertIs(signal.getsignal(signal.SIGTERM), before)
        self.assertFalse((self.root / "result/fixture").exists())

    def test_result_bound_failure_not_claimed_ready(self):
        self.request["max_result_bytes"] = 1
        self.write_request()
        self.assertEqual(self.run_fixture(), 1)
        marker = json.loads((self.parent / "worker.exit.json").read_text())
        self.assertFalse(marker["result_archive_ready"])
        self.assertFalse((self.parent / "results.tar.gz").exists())

    def test_real_sigterm_finishes_child_before_result_archive(self):
        # Run only fixture shell code. The GPU/compiler probes are local scripts;
        # no CUDA program or cloud/network operation is launched.
        fakebin = self.parent / "bin"
        fakebin.mkdir()
        for name, text in (("nvidia-smi", "GPU-test,NVIDIA A100-PCIE-80GB,81920 MiB,580.1"),
                           ("nvcc", "fixture compiler"), ("cmake", "fixture cmake")):
            command = fakebin / name
            command.write_text("#!/bin/sh\nprintf '%s\\n' '" + text + "'\n")
            command.chmod(0o755)
        script = self.root / "run.sh"
        script.write_text("#!/bin/bash\nset -eu\n[ -z \"${CONTAINER_API_KEY+x}\" ]\n"
                          "mkdir -p result/fixture\n"
                          "trap 'echo terminal > result/fixture/terminal; exit 130' TERM\n"
                          "touch result/fixture/started\nwhile :; do sleep 0.1; done\n")
        manifest = json.loads((self.root / ".thunderfft-cloud-source.json").read_text())
        manifest["files"]["run.sh"]["sha256"] = hashlib.sha256(script.read_bytes()).hexdigest()
        (self.root / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
        environment = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ.get("PATH", ""),
                           CONTAINER_API_KEY="fixture-secret")
        process = subprocess.Popen([sys.executable, str(Path(worker.__file__).resolve()),
                                    str(self.root), str(self.request_file)],
                                   env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            until = time.monotonic() + 15
            while not (self.root / "result/fixture/started").exists():
                if process.poll() is not None or time.monotonic() > until:
                    self.fail("fixture worker did not start its job")
                time.sleep(0.05)
            process.send_signal(signal.SIGTERM)
            process.communicate(timeout=10)
            self.assertEqual(process.returncode, 130)
            with tarfile.open(self.parent / "results.tar.gz") as archive:
                self.assertEqual(archive.extractfile("result/fixture/terminal").read(), b"terminal\n")
                jobs = json.load(archive.extractfile("result/cloud-worker/jobs.json"))
            self.assertEqual(jobs[0]["status"], "interrupted")
        finally:
            if process.poll() is None:
                process.kill()
            process.communicate()


class PackingTests(unittest.TestCase):
    def test_links_and_special_files_rejected_and_no_archive_published(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = root / "result"
            result.mkdir()
            bad = result / "bad"
            for kind in ("link", "fifo"):
                with self.subTest(kind=kind):
                    bad.symlink_to("/etc/passwd") if kind == "link" else os.mkfifo(bad)
                    with self.assertRaises(ValueError):
                        worker.pack_results(result, root / "results.tar.gz", 100000, time.monotonic() + 5)
                    bad.unlink()
                    self.assertFalse((root / "results.tar.gz").exists())

    def test_hardlinked_results_encoded_as_ordinary_files(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            result = root / "result"
            result.mkdir()
            (result / "a").write_text("same content")
            os.link(result / "a", result / "b")
            worker.pack_results(result, root / "results.tar.gz", 100000, time.monotonic() + 5)
            with tarfile.open(root / "results.tar.gz") as archive:
                self.assertTrue(archive.getmember("result/a").isfile())
                self.assertTrue(archive.getmember("result/b").isfile())


class ModelRoundTripTests(unittest.TestCase):
    def test_real_table2_runner_collection_download_import_and_offline_assembly(self):
        """Actual CPU model pipeline; GPU identity/version output is synthetic."""
        repository = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as name:
            temporary = Path(name)
            source = temporary / "source"
            source.mkdir()
            selected = ["run.sh", "artifact/runner.py", "artifact/full_pipeline.py", "artifact/collection.py", "artifact/progress.py",
                        "artifact/common.sh", "artifact/registry.json"]
            selected += [str(path.relative_to(repository))
                         for directory in (repository / "artifact/cloud", repository / "artifact/table2",
                                           repository / "artifact/table3")
                         for path in directory.iterdir()
                         if path.is_file() and not path.name.startswith("test_")
                         and path.suffix in (".py", ".sh", ".json", ".csv")]
            for relative in selected:
                target = source / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(repository / relative, target)
            manifest = {"schema_version": 1,
                        "git": {"head": "synthetic-cpu-roundtrip-fixture", "dirty": False},
                        "files": {relative: {"sha256": hashlib.sha256((source / relative).read_bytes()).hexdigest()}
                                  for relative in selected}}
            (source / ".thunderfft-cloud-source.json").write_text(json.dumps(manifest))
            request = {"gpu": "A100", "arch": "80", "profile": "smoke", "execution_seconds": 60,
                       "max_result_bytes": 10_000_000, "build_jobs": 2, "task_timeout": 10,
                       "collection": "cpu-roundtrip",
                       "jobs": [{"evaluations": ["table2"], "part": "model-only", "environment": {}}]}
            request_file = temporary / "request.json"
            request_file.write_text(json.dumps(request))
            fakebin = temporary / "synthetic-probes"
            fakebin.mkdir()
            probe_log = temporary / "probe-calls.log"
            for tool, output in (("nvidia-smi", "GPU-synthetic,NVIDIA A100-PCIE-80GB,81920 MiB,580.1"),
                                 ("nvcc", "synthetic compiler version only"),
                                 ("cmake", "synthetic cmake version only")):
                executable = fakebin / tool
                executable.write_text("#!/bin/sh\nprintf '%s\\n' '" + tool + "' >> \"$FIXTURE_PROBE_LOG\"\n"
                                      + ("[ \"$#\" = 1 ] && [ \"$1\" = --version ] || exit 99\n" if tool != "nvidia-smi" else "")
                                      + "printf '%s\\n' '" + output + "'\n")
                executable.chmod(0o755)
            environment = dict(os.environ, PATH=str(fakebin) + os.pathsep + os.environ.get("PATH", ""),
                               FIXTURE_PROBE_LOG=str(probe_log), CONTAINER_API_KEY="not-a-real-key")
            completed = subprocess.run([sys.executable, str(source / "artifact/cloud/worker.py"),
                                        str(source), str(request_file)], env=environment,
                                       capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(json.loads((temporary / "worker.exit.json").read_text())["result_archive_ready"])

            # Run the real receiving/import contract with no API/SSH objects.
            with patch.object(sys, "path", [str(repository), *sys.path]):
                from artifact import collection
                from artifact.cloud import coordinator, transport
            fetched = transport.safe_extract_result(temporary / "results.tar.gz", temporary / "download") / "result"
            remote_manifest = json.loads((fetched / "collections/cpu-roundtrip/manifest.json").read_text())
            self.assertEqual(len(remote_manifest["parts"]), 1)
            part = remote_manifest["parts"][0]
            self.assertEqual((part["evaluation"], part["part"]), ("table2", "model-only"))
            remote_part = fetched / "collections/cpu-roundtrip" / part["path"]
            status = json.loads((remote_part / "status.json").read_text())
            self.assertEqual((status["status"], status["exit_code"]), ("completed", 0))
            with (remote_part / "data.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["origin"] for row in rows}, {"calculated"})
            run_manifests = list((fetched / "runs").glob("*/manifest.json"))
            self.assertEqual(len(run_manifests), 1)
            self.assertEqual(json.loads(run_manifests[0].read_text())["cloud_source"]["git"]["head"],
                             "synthetic-cpu-roundtrip-fixture")
            local_result = temporary / "local-result"
            args = SimpleNamespace(collection="cpu-roundtrip", result_dir=local_result, profile="smoke")
            imported = coordinator.register_results(args, request["jobs"], fetched, collection)
            self.assertEqual(imported, [{"evaluation": "table2", "part": "model-only"}])
            probes_before_assembly = probe_log.read_text()
            assembled = subprocess.run(["bash", str(source / "run.sh"), "table2", "--collection", "cpu-roundtrip",
                                        "--assemble", "--result-dir", str(local_result)], env=environment,
                                       capture_output=True, text=True, timeout=15)
            self.assertEqual(assembled.returncode, 2, assembled.stdout + assembled.stderr)
            self.assertEqual(probe_log.read_text(), probes_before_assembly)
            assembly_root = local_result / "collections/cpu-roundtrip/assembled"
            assembled_status = json.loads((assembly_root / "table2/status.json").read_text())
            self.assertEqual(assembled_status["status"], "partial")
            self.assertFalse(assembled_status["full_paper_comparison"])
            self.assertIn("cuFFTDx measured row", (assembly_root / "missing.csv").read_text())
            self.assertEqual((assembly_root / "table2/table.csv").read_bytes(), (remote_part / "table.csv").read_bytes())
            self.assertTrue((assembly_root / "status.csv").is_file())


if __name__ == "__main__":
    unittest.main()
