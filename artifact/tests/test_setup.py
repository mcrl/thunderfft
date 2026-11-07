"""Exercise setup archive placement without installation, network, HOME edits or GPUs."""
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (ROOT / "artifact/setup.sh").read_text()
HELPERS = SOURCE[SOURCE.index("step() {"):SOURCE.index('step "Python >= 3.9"')]


class SetupInstallationTests(unittest.TestCase):
    def install(self, source, destination, digest="unused", suffix=".tar.gz"):
        # Run only the installer helper, never the setup entrypoint.
        script = "set -euo pipefail\n" + HELPERS + '\ninstall_archive fixture "$1" "$2" "$3" "$4"\n'
        return subprocess.run(["bash", "-c", script, "setup-test", "https://invalid.test/pinned" + suffix,
                               digest, str(source), str(destination)], capture_output=True, text=True)

    def archive(self, path, root_name, contents):
        with tarfile.open(path, "w:xz" if path.suffix == ".xz" else "w:gz") as archive:
            payload = contents.encode()
            member = tarfile.TarInfo(root_name + "/nvidia/mathdx/25.06/include/cufftdx.hpp")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_cuda_archives_with_same_root_use_distinct_destinations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for major in (12, 13):
                archive = root / f"cuda{major}.tar.gz"
                digest = self.archive(archive, "nvidia-mathdx-25.06.1", f"CUDA {major}")
                result = self.install(archive, root / f"mathdx-cuda{major}", digest)
                self.assertEqual(result.returncode, 0, result.stderr)
            for major in (12, 13):
                self.assertEqual((root / f"mathdx-cuda{major}/nvidia/mathdx/25.06/include/cufftdx.hpp").read_text(),
                                 f"CUDA {major}")
            self.assertFalse((root / "nvidia-mathdx-25.06.1").exists())

    def test_extracted_directory_override_does_not_nest_on_rerun(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "extracted"
            source.mkdir()
            (source / "marker").write_text("first")
            target = root / "destination"
            for value in ("first", "updated"):
                (source / "marker").write_text(value)
                result = self.install(source, target)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual((target / "marker").read_text(), value)
                self.assertFalse((target / "extracted").exists())

    def test_archive_digest_failure_prevents_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "fixture.tar.xz"
            self.archive(source, "pinned", "data")
            target = root / "destination"
            result = self.install(source, target, "0" * 64, ".tar.xz")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SHA-256 mismatch", result.stderr)
            self.assertFalse(target.exists())

    def test_install_root_contains_detected_cuda_major(self):
        statement = next(line for line in SOURCE.splitlines() if line.startswith("MATHDX_INSTALL="))
        roots = []
        for major in (12, 13):
            # Expands the real HOME for a path calculation only; no filesystem writes.
            script = 'MATHDX_VERSION=25.06.1\ncuda_major="$1"\n' + statement + '\nprintf "%s" "$MATHDX_INSTALL"'
            result = subprocess.run(["bash", "-c", script, "setup-test", str(major)],
                                    check=True, capture_output=True, text=True)
            roots.append(result.stdout)
            self.assertTrue(result.stdout.endswith(f"nvidia-mathdx-25.06.1-cuda{major}"))
        self.assertNotEqual(*roots)


if __name__ == "__main__":
    unittest.main()
