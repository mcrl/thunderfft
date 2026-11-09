#!/usr/bin/env python3
"""CPU-only tests: no network, credentials, GPU, or rented resources."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("cloud_transport", Path(__file__).with_name("transport.py"))
transport = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(transport)


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.git("init", "-q")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.invalid")
        (self.repo / "run.sh").write_text("#!/bin/sh\ntrue\n")
        (self.repo / "run.sh").chmod(0o755)
        (self.repo / ".gitignore").write_text("result/\nbuild/\n")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture")
        self.dest = self.root / "source.tar.gz"

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              capture_output=True, text=True).stdout.strip()

    def test_tracked_snapshot_has_provenance_and_no_history_or_results(self):
        (self.repo / "result").mkdir()
        (self.repo / "result" / "prior.csv").write_text("not current measurements")
        manifest = transport.build_source_bundle(self.repo, self.dest)
        self.assertEqual(manifest["git"]["head"], self.git("rev-parse", "HEAD"))
        self.assertFalse(manifest["git"]["dirty"])
        with tarfile.open(self.dest) as archive:
            self.assertEqual(set(archive.getnames()), {".gitignore", "run.sh", transport.SOURCE_MANIFEST})
            self.assertEqual(archive.getmember("run.sh").mode, 0o755)
            embedded = json.load(archive.extractfile(transport.SOURCE_MANIFEST))
        self.assertEqual(embedded["files"], manifest["files"])
        self.assertEqual(manifest["archive_sha256"], hashlib.sha256(self.dest.read_bytes()).hexdigest())

    def extract_distribution(self):
        transport.build_source_bundle(self.repo, self.dest)
        extracted = self.root / "extracted"
        with tarfile.open(self.dest) as archive:
            archive.extractall(extracted)
        return extracted

    def test_extracted_distribution_rebundles_without_git(self):
        extracted = self.extract_distribution()
        head = self.git("rev-parse", "HEAD")
        (extracted / "__pycache__").mkdir()
        (extracted / "__pycache__" / "x.pyc").write_bytes(b"generated")
        (extracted / "result").mkdir()
        (extracted / "result" / "run.csv").write_text("new measurement")
        self.assertTrue(transport.is_distribution(extracted))
        self.assertTrue(transport.distribution_status(extracted)["unmodified"])
        manifest = transport.build_source_bundle(extracted, self.root / "again.tar.gz")
        self.assertEqual(manifest["git"]["head"], head)
        self.assertFalse(manifest["git"]["dirty"])
        self.assertEqual(manifest["distribution"]["modified_files"], [])
        with tarfile.open(self.root / "again.tar.gz") as archive:
            self.assertEqual(set(archive.getnames()), {".gitignore", "run.sh", transport.SOURCE_MANIFEST})
            self.assertEqual(archive.getmember("run.sh").mode, 0o755)

    def test_extracted_distribution_rejects_edits_and_extra_files(self):
        extracted = self.extract_distribution()
        (extracted / "run.sh").write_text("#!/bin/sh\nfalse\n")
        with self.assertRaisesRegex(ValueError, "modified.*run.sh"):
            transport.build_source_bundle(extracted, self.root / "edited.tar.gz")
        manifest = transport.build_source_bundle(extracted, self.root / "edited.tar.gz", allow_dirty=True)
        self.assertTrue(manifest["git"]["dirty"])
        self.assertEqual(manifest["distribution"]["modified_files"], ["run.sh"])
        (extracted / "notes.txt").write_text("not packaged")
        self.assertEqual(transport.distribution_status(extracted)["untracked"], ["notes.txt"])
        with self.assertRaisesRegex(ValueError, "outside the distribution manifest"):
            transport.build_source_bundle(extracted, self.root / "extra.tar.gz", allow_dirty=True)
        (extracted / "notes.txt").unlink()
        (extracted / "run.sh").unlink()
        with self.assertRaisesRegex(ValueError, "missing.*run.sh"):
            transport.build_source_bundle(extracted, self.root / "missing.tar.gz", allow_dirty=True)

    def test_dirty_opt_in_archives_actual_bytes_and_labels_revision(self):
        (self.repo / "run.sh").write_text("changed working tree\n")
        with self.assertRaisesRegex(ValueError, "dirty"):
            transport.build_source_bundle(self.repo, self.dest)
        manifest = transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)
        self.assertTrue(manifest["git"]["dirty"])
        with tarfile.open(self.dest) as archive:
            self.assertEqual(archive.extractfile("run.sh").read(), b"changed working tree\n")

    def test_untracked_implementation_not_silently_omitted(self):
        (self.repo / "new_kernel.cu").write_text("new code\n")
        with self.assertRaisesRegex(ValueError, "untracked"):
            transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)

    def test_tracked_secret_paths_are_excluded(self):
        (self.repo / "cloud_key").mkdir()
        (self.repo / "cloud_key" / "secret").write_text("fixture secret")
        (self.repo / ".env").write_text("fixture secret")
        self.git("add", ".")
        self.git("commit", "-qm", "fixture secret paths")
        manifest = transport.build_source_bundle(self.repo, self.dest)
        self.assertEqual(set(manifest["excluded_files"]), {"cloud_key/secret", ".env"})
        with tarfile.open(self.dest) as archive:
            self.assertNotIn("cloud_key/secret", archive.getnames())

    def test_credential_content_rejected_without_disclosing_it(self):
        (self.repo / "run.sh").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\nsecret\n")
        with self.assertRaisesRegex(ValueError, "credential-like content") as error:
            transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)
        self.assertNotIn("secret", str(error.exception))
        self.assertFalse(self.dest.exists())

    def test_symlink_source_and_byte_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, "byte limit"):
            transport.build_source_bundle(self.repo, self.dest, max_bytes=1)
        (self.repo / "run.sh").unlink()
        (self.repo / "run.sh").symlink_to("/etc/passwd")
        with self.assertRaisesRegex(ValueError, "regular file"):
            transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)

    def test_existing_dirty_file_cannot_change_during_snapshot(self):
        source = self.repo / "run.sh"
        source.write_text("first dirty version\n")
        original_git = transport._git
        reads = 0

        def changing_git(repo, *args):
            nonlocal reads
            answer = original_git(repo, *args)
            if args[0] == "status":
                reads += 1
                if reads == 2:
                    source.write_text("second dirty version\n")
            return answer

        with patch.object(transport, "_git", side_effect=changing_git):
            with self.assertRaisesRegex(ValueError, "files changed"):
                transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)
        self.assertFalse(self.dest.exists())

    def test_api_token_in_ordinary_source_is_rejected(self):
        (self.repo / "run.sh").write_text('vast_api_key="' + "a" * 32 + '"\n')
        with self.assertRaisesRegex(ValueError, "credential-like"):
            transport.build_source_bundle(self.repo, self.dest, allow_dirty=True)


class SSHTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.key = self.root / "test-key"
        self.key.write_text("nonfunctional fixture")
        self.key.chmod(0o600)
        self.known = self.root / "known_hosts"
        self.ssh = transport.SSH("127.0.0.1", 2222, "root", self.key, self.known)

    def test_identity_and_host_pin_options_and_shell_quoting(self):
        argv = ["printf", "%s", "$(touch /tmp/should-not-run); ' x"]
        command = self.ssh._command(argv)
        self.assertIn("ForwardAgent=no", command)
        self.assertIn("IdentityAgent=none", command)
        self.assertIn("IdentitiesOnly=yes", command)
        self.assertIn("StrictHostKeyChecking=accept-new", command)
        self.assertEqual(shlex.split(command[-1]), argv)
        self.known.write_text("fixture pinned key\n")
        self.assertIn("StrictHostKeyChecking=yes", self.ssh._command(argv))

    def test_bad_permissions_symlink_and_in_repo_keys_rejected(self):
        self.key.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "0600"):
            self.ssh._command(["true"])
        self.key.chmod(0o600)
        with self.assertRaisesRegex(ValueError, "outside"):
            transport.validate_private_key(self.key, self.root)
        link = self.root / "link"
        link.symlink_to(self.key)
        with self.assertRaisesRegex(ValueError, "symlink"):
            transport.validate_private_key(link)

    def test_run_has_timeout_and_no_shell(self):
        with patch.object(transport.subprocess, "run") as run:
            self.ssh.run(["true"], timeout=4)
        self.assertEqual(run.call_args.kwargs["timeout"], 4)
        self.assertNotIn("shell", run.call_args.kwargs)

    def test_cancellable_run_preserves_text_input_output_and_returncode(self):
        self.ssh.cancel_event = threading.Event()
        command = [sys.executable, "-c", "import sys,time;time.sleep(.15);sys.stdout.write(sys.stdin.read());sys.stderr.write('error');sys.exit(3)"]
        with patch.object(self.ssh, "_command", return_value=command):
            result = self.ssh.run(["fixture"], input="input\n" * 50000, timeout=3, check=False)
            self.assertEqual(result.returncode, 3)
            self.assertEqual(result.stdout, "input\n" * 50000)
            self.assertEqual(result.stderr, "error")
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                self.ssh.run(["fixture"], input="small", timeout=3)
            self.assertEqual(failed.exception.output, "small")
            self.assertEqual(failed.exception.stderr, "error")

    def test_cancel_before_launch_does_not_create_process(self):
        self.ssh.cancel_event = threading.Event()
        self.ssh.cancel_event.set()
        with patch.object(transport.subprocess, "Popen", side_effect=AssertionError("cancelled command started")), \
                self.assertRaises(InterruptedError):
            self.ssh.run(["true"], timeout=5)

    def test_cancellation_interrupts_run_and_upload_and_reaps_process(self):
        local = self.root / "payload"
        local.write_bytes(b"payload")
        original = transport.subprocess.Popen
        for operation in (lambda: self.ssh.run(["fixture"], timeout=30),
                          lambda: self.ssh.upload(local, "/remote/payload", timeout=30)):
            with self.subTest(operation=operation):
                self.ssh.cancel_event = threading.Event()
                processes = []

                def start(*args, **kwargs):
                    process = original(*args, **kwargs)
                    processes.append(process)
                    return process

                timer = threading.Timer(.15, self.ssh.cancel_event.set)
                started = time.monotonic()
                timer.start()
                try:
                    command = [sys.executable, "-c", "import time;time.sleep(30)"]
                    with patch.object(self.ssh, "_command", return_value=command), \
                            patch.object(transport.subprocess, "Popen", side_effect=start), \
                            self.assertRaises(InterruptedError):
                        operation()
                finally:
                    timer.cancel()
                    timer.join()
                self.assertLess(time.monotonic() - started, 3)
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll())

    def test_cancellable_command_timeout_still_kills_and_reaps_process(self):
        self.ssh.cancel_event = threading.Event()
        original = transport.subprocess.Popen
        processes = []

        def start(*args, **kwargs):
            process = original(*args, **kwargs)
            processes.append(process)
            return process

        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import time;time.sleep(30)"]), \
                patch.object(transport.subprocess, "Popen", side_effect=start), \
                self.assertRaises(subprocess.TimeoutExpired):
            self.ssh.run(["fixture"], timeout=.05)
        self.assertEqual(len(processes), 1)
        self.assertIsNotNone(processes[0].poll())

    def test_download_cancellation_preserves_existing_file_and_removes_temporary(self):
        target = self.root / "download.tar"
        target.write_bytes(b"previous complete file")
        self.ssh.cancel_event = threading.Event()
        timer = threading.Timer(.15, self.ssh.cancel_event.set)
        timer.start()
        try:
            command = [sys.executable, "-c", "import os,time;os.close(1);os.close(2);time.sleep(30)"]
            with patch.object(self.ssh, "_command", return_value=command), self.assertRaises(InterruptedError):
                self.ssh.download("/remote/file", target, timeout=30)
        finally:
            timer.cancel()
            timer.join()
        self.assertEqual(target.read_bytes(), b"previous complete file")
        self.assertFalse(list(self.root.glob(".download-*")))

    def test_private_key_is_not_uploadable(self):
        with self.assertRaisesRegex(ValueError, "noncredential"):
            self.ssh.upload(self.key, "/remote/key")

    def test_known_hosts_special_file_fails_without_opening(self):
        fifo = self.root / "fifo"
        os.mkfifo(fifo, 0o600)
        with self.assertRaisesRegex(ValueError, "regular file"):
            transport.SSH("localhost", 22, "root", self.key, fifo)

    def test_download_streams_atomic_bounded_file(self):
        target = self.root / "download.tar"
        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'abc')"]):
            self.ssh.download("/remote/file", target, timeout=2, max_bytes=3)
        self.assertEqual(target.read_bytes(), b"abc")
        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import sys;sys.stdout.buffer.write(b'abcd')"]):
            with self.assertRaisesRegex(ValueError, "byte limit"):
                self.ssh.download("/remote/file", target, timeout=2, max_bytes=3)
        self.assertEqual(target.read_bytes(), b"abc")
        self.assertFalse(list(self.root.glob(".download-*")))

    def test_download_timeout_and_remote_errors_leave_no_partial_file(self):
        target = self.root / "download.tar"
        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import time;time.sleep(10)"]):
            with self.assertRaises(subprocess.TimeoutExpired):
                self.ssh.download("/remote/file", target, timeout=0.05)
        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import sys;sys.stderr.write('missing');sys.exit(2)"]):
            with self.assertRaises(subprocess.CalledProcessError):
                self.ssh.download("/remote/file", target, timeout=2)
        self.assertFalse(target.exists())
        self.assertFalse(list(self.root.glob(".download-*")))

    def test_download_flooded_stderr_is_bounded(self):
        with patch.object(self.ssh, "_command", return_value=[sys.executable, "-c", "import sys;sys.stderr.write('x'*1100000)"]):
            with self.assertRaisesRegex(ValueError, "stderr exceeds"):
                self.ssh.download("/remote/file", self.root / "download", timeout=2)


class ExtractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.root / "result.tar.gz"
        self.dest = self.root / "download"

    def bundle(self, entries):
        with tarfile.open(self.archive, "w:gz") as archive:
            for name, kind, data in entries:
                info = tarfile.TarInfo(name)
                info.type, info.mode = kind, 0o7777
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = "/etc/passwd"
                if kind == tarfile.REGTYPE:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
                else:
                    archive.addfile(info)

    def test_valid_results_published_without_executable_permissions(self):
        self.bundle([("result", tarfile.DIRTYPE, b""),
                     ("result/table3/run/data.csv", tarfile.REGTYPE, b"actual,csv\n")])
        self.assertEqual(transport.safe_extract_result(self.archive, self.dest), self.dest)
        file = self.dest / "result/table3/run/data.csv"
        self.assertEqual(file.read_bytes(), b"actual,csv\n")
        self.assertEqual(file.stat().st_mode & 0o7777, 0o600)
        with self.assertRaisesRegex(ValueError, "already exists"):
            transport.safe_extract_result(self.archive, self.dest)

    def test_paths_links_special_files_and_duplicates_rejected_atomically(self):
        bad_entries = [
            [("../escape", tarfile.REGTYPE, b"bad")],
            [("/result/absolute", tarfile.REGTYPE, b"bad")],
            [("result/../escape", tarfile.REGTYPE, b"bad")],
            [("result/a", tarfile.SYMTYPE, b"")],
            [("result/a", tarfile.LNKTYPE, b"")],
            [("result/a", tarfile.FIFOTYPE, b"")],
            [("other/data", tarfile.REGTYPE, b"bad")],
            [("result/a", tarfile.REGTYPE, b"one"), ("result/a", tarfile.REGTYPE, b"two")],
            [("result/a/b", tarfile.REGTYPE, b"one"), ("result/a", tarfile.REGTYPE, b"two")],
        ]
        for entries in bad_entries:
            with self.subTest(entries=entries):
                self.bundle(entries)
                with self.assertRaises(ValueError):
                    transport.safe_extract_result(self.archive, self.dest)
                self.assertFalse(self.dest.exists())
                self.assertFalse(list(self.root.glob(".extract-*")))

    def test_extraction_count_and_byte_limits(self):
        self.bundle([("result/a", tarfile.REGTYPE, b"abcd"), ("result/b", tarfile.REGTYPE, b"abc")])
        for limits in ({"max_bytes": 6}, {"max_files": 1}):
            with self.subTest(limits=limits), self.assertRaisesRegex(ValueError, "limits"):
                transport.safe_extract_result(self.archive, self.dest, **limits)
        self.assertFalse(self.dest.exists())


if __name__ == "__main__":
    unittest.main()
