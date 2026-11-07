#!/usr/bin/env python3
"""Synthetic CPU tests for independent Figure 3 measurements and normalization."""
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_measure import module, native

HERE = Path(__file__).resolve().parent
A = module("figure3_test_assemble", HERE / "assemble.py")
UUID = "GPU-11111111-2222-3333-4444-555555555555"
NAME = "NVIDIA A100-SXM4-80GB"


def bundle(root, library, *, e2e=50.0):
    root.mkdir()
    (root / "raw").mkdir()
    precision = "fp32" if library == "cufft" else "tf32"
    raw = native(library, e2e=e2e, precision=precision)
    (root / "raw/data.csv").write_text(raw)
    (root / "raw/data.log").write_text(f"compiled_CUDART=12090 runtime=12090 driver=13000\nGPU={NAME} compute=8.0 batch=1024\nGPU_UUID={UUID}\n")
    meta = {"evaluation": "figure3", "profile": "smoke", "status": "completed", "numerical_and_timing_gates_passed": True,
            "settings": {"batch": 1024, "repeats": 64, "trials": 1},
            "selection": {"N": [64], "precisions": ["tf32"], "libraries": [library]},
            "gpu": "A100", "gpu_name": NAME, "gpu_uuid": UUID, "compute_capability": "8.0",
            "source_sha256": {"src.cu": "a" * 64}, "binary_sha256": "b" * 64,
            "gpu_inventory_before": {"exit_code": 0, "stdout": f"0, {UUID}, {NAME}, 580\n"},
            "commands": [{"argv": ["/binary", "--library", library, "64", precision, A.M.NATIVE[library][1], "64", "1", "1024"],
                          "exit_code": 0, "stdout": "raw/data.csv", "stderr": "raw/data.log"}]}
    (root / "metadata.json").write_text(json.dumps(meta))
    parsed = A.M.parse_native(raw, library=library, n=64, precision=precision, batch=1024,
                             repeats=64, trials=1, gpu="A100", raw_csv="raw/data.csv")
    A.M.TABLE3.write_csv(root / "data.csv", parsed, A.M.FIELDS)
    return root


def rows(path):
    return list(csv.DictReader(io.StringIO(path.read_text())))


class Assembly(unittest.TestCase):
    def test_collect_fft_before_communication(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fft = bundle(root / "fft", "thunderfft", e2e=100)
            state = A.assemble([fft], root / "before", "smoke")
            measured = [row for row in rows(root / "before/summary.csv") if row["status"] != "pending"]
            self.assertEqual(state["status"], "partial")
            self.assertEqual((measured[0]["normalized"], measured[0]["status"]), ("", "collected_unscaled"))
            comm = bundle(root / "comm", "copy", e2e=50)
            A.assemble([fft, comm], root / "after", "smoke")
            measured = [row for row in rows(root / "after/summary.csv") if row["library"] == "ThunderFFT" and row["N"] == "64"]
            self.assertEqual(float(measured[0]["normalized"]), 0.5)
            self.assertTrue((root / "after/figure.pdf").is_file())

    def test_output_cannot_overlap_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = bundle(root / "part", "copy")
            original = {path.relative_to(part).as_posix(): path.read_bytes() for path in part.rglob("*") if path.is_file()}
            alias = root / "alias"
            alias.symlink_to(part, target_is_directory=True)
            for output in (part, root, part / "nested-output", alias):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "output must be separate"):
                    A.assemble([part], output, "smoke")
            self.assertFalse((part / "nested-output").exists())
            self.assertEqual(original, {path.relative_to(part).as_posix(): path.read_bytes()
                                        for path in part.rglob("*") if path.is_file()})

    def test_reject_checkpoint_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            part = bundle(Path(directory) / "part", "copy")
            (part / "data.csv").write_text((part / "data.csv").read_text().replace("0.05,", "0.06,"))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                A.load_part(part, "smoke")

    def test_reject_overlapping_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "overlapping"):
                A.assemble([bundle(root / "a", "copy"), bundle(root / "b", "copy")], root / "out", "smoke")

    def test_reject_missing_native_log(self):
        with tempfile.TemporaryDirectory() as directory:
            part = bundle(Path(directory) / "part", "copy")
            (part / "raw/data.log").unlink()
            with self.assertRaisesRegex(ValueError, "evidence"):
                A.load_part(part, "smoke")

    def test_library_selection(self):
        with patch.dict(os.environ, {"THUNDERFFT_FIGURE3_LIBRARIES": "thunderfft"}):
            self.assertEqual(A.M.selected_libraries(False), ("thunderfft",))
        for value in ("cufftdx", "all", "copy,copy", ""):
            with self.subTest(value=value), patch.dict(os.environ, {"THUNDERFFT_FIGURE3_LIBRARIES": value}):
                with self.assertRaises(ValueError):
                    A.M.selected_libraries(False)
        with patch.dict(os.environ, {"THUNDERFFT_FIGURE3_LIBRARIES": "cufftdx"}):
            self.assertEqual(A.M.selected_libraries(True), ("cufftdx",))

    def test_plot_distinguishes_gpu_panels(self):
        with tempfile.TemporaryDirectory() as directory:
            base = {"gpu": "A100", "N": 64, "precision": "tf32", "library": "ThunderFFT", "normalized": 0.9}
            A.PLOT.make_plot([base, dict(base, gpu="H100", normalized=1.1)], Path(directory) / "figure.pdf")
            self.assertGreater((Path(directory) / "figure.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
