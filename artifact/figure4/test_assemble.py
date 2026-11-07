#!/usr/bin/env python3
"""CPU-only regression tests for library/filter collection assembly."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("figure4_assemble_test", HERE / "assemble.py")
ASSEMBLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLE)
UUID = "GPU-f4bb2c48-b43c-0f5a-8f59-75d6373ea4e6"


def part(root, name, library="ThunderFFT", filters=(3,), spec=None, **changes):
    spec = spec or ASSEMBLE.MEASURE.FIGURE4
    directory = root / name
    directory.mkdir()
    status = {"evaluation": spec["evaluation"], "profile": "smoke", "status": "completed", "exit_code": 0}
    metadata = dict(status, numerical_and_timing_gates_passed=True, operation=spec["operation"],
                    gpu="A100", gpu_name="NVIDIA A100", compute_capability="8.0", gpu_uuid=UUID,
                    settings={"trials": 1, "filters": list(filters)}, binary_sha256={library: "a" * 64},
                    source_sha256={"source.cu": "b" * 64})
    metadata.update(changes)
    metadata["gpu_inventory_before"] = {"exit_code": 0, "stdout": f"0, {metadata.get('gpu_uuid') or UUID}, NVIDIA A100, 580\n"}
    metadata["effective_CUDA_VISIBLE_DEVICES"] = metadata.get("gpu_uuid") or "0"
    (directory / "raw").mkdir()
    log = "GPU=NVIDIA A100 compute=8.0 batch=256 seed=42\n"
    (directory / "raw/output.log").write_text(log)
    if spec["evaluation"] == "figure4":
        command = {"ThunderFFT": ASSEMBLE.MEASURE.figure4_command,
                   "cuDNN": ASSEMBLE.MEASURE.figure4_cudnn_command,
                   "FlashFFTConv": ASSEMBLE.MEASURE.figure4_flash_command}[library]
        argv, _ = command(Path("/binary"), filters, "smoke", 1)
        validated = "1/256 sequences"
        raw = "".join(f"Filter size: {f}\nValidation N=4096 vs direct circular conv (raw/N): L2_rel=0.001\n"
                      f"Validation sequences: 1 of 256\nCircular convolution validation: PASS\n"
                      f"Native (trial=0) time: {1+f/100} ms\n" for f in filters)
    else:
        argv = ["/binary", "128", ",".join(map(str, filters)), library.lower(), "5", "20", "1"]
        validated = "1 sampled outputs"
        raw = "".join(f"Filter size: {f}\nValidation 2D ({library.lower()}) vs direct reference: L2_rel=0.001 samples=1\n"
                      f"Convolution validation: PASS\nNative (trial=0) time: {1+f/100} ms\n" for f in filters)
    metadata["commands"] = [{"library": library, "argv": argv, "stdout": "raw/output.out", "stderr": "raw/output.log", "exit_code": 0}]
    (directory / "raw/output.out").write_text(raw)
    (directory / "status.json").write_text(json.dumps(status))
    (directory / "metadata.json").write_text(json.dumps(metadata))
    rows = [{"gpu": "A100", "shape": "L=4096,B=4,D=64" if spec["evaluation"] == "figure4" else "N=128",
             "filter": size, "library": library, "precision": spec["precision"], "repeat": 0,
             "time_ms": 1 + size / 100, "filter_fft_included": "n/a" if library == "cuDNN" else "true",
             "rel_l2": 0.001, "validated": validated, "passed": "true", "operation": spec["operation"],
             "source": "raw/output.out"} for size in filters]
    ASSEMBLE.MEASURE.TABLE3.write_csv(directory / "data.csv", rows, ASSEMBLE.MEASURE.FIELDS)
    return directory


class Assembly(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plot = patch.object(ASSEMBLE.PLOT, "make_plot", return_value=[])
        self.plot.start()

    def tearDown(self):
        self.plot.stop()
        self.temp.cleanup()

    def test_split_libraries_and_filters_complete_grid(self):
        parts = [part(self.root, f"{lib}{f}", library=lib, filters=(f,))
                 for lib in ASSEMBLE.MEASURE.FIGURE4["libraries"] for f in (3, 9, 53)]
        state = ASSEMBLE.assemble(parts, self.root / "out", "smoke")
        self.assertEqual(state["status"], "complete")
        self.assertFalse(state["full_paper_comparison"])
        self.assertEqual(len(list(csv.DictReader((self.root / "out/data.csv").read_text().splitlines()))), 9)

    def test_partial_preserves_missing_cells_and_identical_duplicate(self):
        one = part(self.root, "one")
        duplicate = part(self.root, "duplicate")
        state = ASSEMBLE.assemble([one, duplicate], self.root / "out", "smoke")
        self.assertEqual(state["status"], "partial")
        self.assertEqual(len(state["missing"]), 8)
        self.assertEqual(len(list(csv.DictReader((self.root / "out/data.csv").read_text().splitlines()))), 1)

    def test_conflicting_measurement_rejected(self):
        one = part(self.root, "one")
        two = part(self.root, "two")
        text = (two / "data.csv").read_text().replace(",1.03,", ",2.03,")
        (two / "data.csv").write_text(text)
        with self.assertRaisesRegex(ValueError, "conflicting duplicate"):
            ASSEMBLE.assemble([one, two], self.root / "out", "smoke")

    def test_different_device_or_binary_rejected(self):
        one = part(self.root, "one")
        for name, changes in (("device", {"gpu_uuid": "GPU-00000000-0000-0000-0000-000000000001"}),
                              ("binary", {"binary_sha256": {"ThunderFFT": "c" * 64}})):
            other = part(self.root, name, filters=(9,), **changes)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "incompatible"):
                ASSEMBLE.assemble([one, other], self.root / (name + "out"), "smoke")

    def test_empty_collection_is_pending(self):
        state = ASSEMBLE.assemble([], self.root / "out", "smoke")
        self.assertEqual(state["status"], "pending")
        self.assertEqual(len(state["missing"]), 9)

    def test_missing_trials_cannot_complete_a_cell(self):
        one = part(self.root, "one", settings={"trials": 2, "filters": [3]})
        with self.assertRaisesRegex(ValueError, "incomplete trial"):
            ASSEMBLE.assemble([one], self.root / "out", "smoke")

    def test_source_ancestor_and_descendant_outputs_rejected(self):
        one = part(self.root, "one")
        for output in (self.root, one, one / "nested"):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, "source part"):
                ASSEMBLE.assemble([one], output, "smoke")

    def test_backend_snapshot_survives_move_and_original_removal(self):
        one = part(self.root, "one")
        with (one / "data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        external = self.root / "external.csv"
        imported = dict(rows[0], library="FlashFFTConv")
        ASSEMBLE.MEASURE.TABLE3.write_csv(external, [imported], ASSEMBLE.MEASURE.FIELDS)
        copied, hashes, origins = ASSEMBLE.MEASURE.snapshot_backends(
            [external], one, ASSEMBLE.MEASURE.FIGURE4, "A100", "L=4096,B=4,D=64", [3], ("ThunderFFT",))
        metadata = json.loads((one / "metadata.json").read_text())
        metadata.update(backend_sha256=hashes, backend_imports=origins)
        (one / "metadata.json").write_text(json.dumps(metadata))
        ASSEMBLE.MEASURE.TABLE3.write_csv(one / "data.csv", rows + copied, ASSEMBLE.MEASURE.FIELDS)
        external.unlink()
        moved = self.root / "moved"
        one.rename(moved)
        state = ASSEMBLE.assemble([moved], self.root / "out", "smoke")
        self.assertEqual((state["status"], len(state["missing"])), ("partial", 7))
        saved = moved / next(iter(hashes))
        saved.write_text(saved.read_text().replace("1.03", "2.03"))
        with self.assertRaisesRegex(ValueError, "backend CSV hash"):
            ASSEMBLE.assemble([moved], self.root / "tampered", "smoke")

    def test_modified_raw_timing_or_gate_rejected(self):
        for name, before, after in (("time", "1.03 ms", "2.03 ms"), ("gate", "PASS", "FAIL")):
            one = part(self.root, name)
            raw = one / "raw/output.out"
            raw.write_text(raw.read_text().replace(before, after))
            with self.subTest(name=name), self.assertRaises(ValueError):
                ASSEMBLE.assemble([one], self.root / (name + "out"), "smoke")

    def test_missing_log_and_failed_command_rejected(self):
        one = part(self.root, "one")
        (one / "raw/output.log").unlink()
        with self.assertRaisesRegex(ValueError, "raw evidence"):
            ASSEMBLE.assemble([one], self.root / "out", "smoke")
        two = part(self.root, "two")
        metadata = json.loads((two / "metadata.json").read_text())
        metadata["commands"][0]["exit_code"] = 1
        (two / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "successfully"):
            ASSEMBLE.assemble([two], self.root / "out2", "smoke")

    def test_invalid_correctness_rejected(self):
        one = part(self.root, "one", numerical_and_timing_gates_passed=False)
        with self.assertRaisesRegex(ValueError, "gates"):
            ASSEMBLE.assemble([one], self.root / "out", "smoke")

    def test_legacy_without_uuid_only_allowed_alone(self):
        one = part(self.root, "one", gpu_uuid=None)
        self.assertEqual(ASSEMBLE.assemble([one], self.root / "alone", "smoke")["status"], "partial")
        two = part(self.root, "two", filters=(9,), gpu_uuid=None)
        with self.assertRaisesRegex(ValueError, "UUID"):
            ASSEMBLE.assemble([one, two], self.root / "together", "smoke")


if __name__ == "__main__":
    unittest.main()
