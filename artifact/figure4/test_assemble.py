#!/usr/bin/env python3
"""CPU-only regression tests for library/filter collection assembly."""
import csv
import importlib.util
import json
import math
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


def with_policy(directory, spec, filters, selected, mode="fast"):
    metadata = json.loads((directory / "metadata.json").read_text())
    metadata["settings"].update(filters=list(filters), selected_libraries=list(selected), cudnn_mode=mode)
    metadata["source_sha256"]["artifact/figure4/policy.py"] = ASSEMBLE.digest(HERE / "policy.py")
    metadata["policy_skipped"] = ASSEMBLE.MEASURE.POLICY.skipped_cells(
        spec["evaluation"], filters, selected, mode)
    (directory / "metadata.json").write_text(json.dumps(metadata))
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

    def fast_parts(self):
        spec = ASSEMBLE.MEASURE.FIGURE4
        parts = [part(self.root, lib, library=lib, filters=(3, 9, 53))
                 for lib in ("ThunderFFT", "FlashFFTConv")]
        parts.append(with_policy(part(self.root, "cuDNN", library="cuDNN", filters=(3, 9)),
                                 spec, (3, 9, 53), ("cuDNN",)))
        return parts

    def test_fast_policy_completes_declared_cells_and_plots_tail_as_gap(self):
        parts = self.fast_parts()
        state = ASSEMBLE.assemble(parts, self.root / "out", "smoke")
        self.assertEqual((state["status"], state["missing"], state["cudnn_mode"]), ("complete", [], "fast"))
        self.assertFalse(state["coverage_complete"])
        self.assertEqual([item["item"] for item in state["policy_skipped"]], ["cuDNN:filter=53"])
        rows = ASSEMBLE.PLOT.make_plot.call_args.args[0]
        gap = next(row for row in rows if row["library"] == "cuDNN" and row["filter"] == 53)
        self.assertTrue(math.isnan(float(gap["time_ms"])))
        self.assertEqual(ASSEMBLE.PLOT.make_plot.call_args.kwargs["policy_skipped"], state["policy_skipped"])
        coverage = json.loads((self.root / "out/coverage.json").read_text())
        self.assertEqual(coverage["policy_skipped"], state["policy_skipped"])

    def test_fast_policy_does_not_hide_missing_required_libraries_or_early_filters(self):
        spec = ASSEMBLE.MEASURE.FIGURE4
        dx = with_policy(part(self.root, "dx", library="cuDNN", filters=(3,)),
                         spec, (3, 9, 53), ("cuDNN",))
        state = ASSEMBLE.assemble([dx], self.root / "out", "smoke")
        self.assertEqual(state["status"], "partial")
        missing = {item["item"] for item in state["missing"]}
        self.assertIn("cuDNN:filter=9", missing)
        self.assertIn("ThunderFFT:filter=53", missing)
        self.assertNotIn("cuDNN:filter=53", missing)

    def test_fabricated_policy_skip_or_missing_policy_hash_is_rejected(self):
        for name in ("early", "full", "hash"):
            with self.subTest(name=name):
                directory = with_policy(part(self.root, name, library="cuDNN", filters=(3, 9)),
                                        ASSEMBLE.MEASURE.FIGURE4, (3, 9, 53), ("cuDNN",))
                metadata = json.loads((directory / "metadata.json").read_text())
                if name == "early":
                    metadata["policy_skipped"][0]["filter"] = 9
                elif name == "full":
                    metadata["settings"]["cudnn_mode"] = "full"
                else:
                    metadata["source_sha256"].pop("artifact/figure4/policy.py")
                (directory / "metadata.json").write_text(json.dumps(metadata))
                with self.assertRaisesRegex(ValueError, "policy"):
                    ASSEMBLE.assemble([directory], self.root / (name + "out"), "smoke")

    def test_same_source_cannot_claim_a_measured_row_was_skipped(self):
        directory = with_policy(part(self.root, "dx", library="cuDNN", filters=(3, 9, 53)),
                                ASSEMBLE.MEASURE.FIGURE4, (3, 9, 53), ("cuDNN",))
        with self.assertRaisesRegex(ValueError, "overlap measured"):
            ASSEMBLE.assemble([directory], self.root / "out", "smoke")

    def test_conflicting_cudnn_modes_rejected_but_unrelated_full_parts_allowed(self):
        parts = self.fast_parts()
        parts.append(part(self.root, "full-tail", library="cuDNN", filters=(53,)))
        with self.assertRaisesRegex(ValueError, "incompatible cuDNN modes"):
            ASSEMBLE.assemble(parts, self.root / "out", "smoke")

    def test_historical_no_mode_keeps_full_coverage_requirement(self):
        directory = part(self.root, "old", library="cuDNN", filters=(3, 9))
        state = ASSEMBLE.assemble([directory], self.root / "out", "smoke")
        self.assertEqual(state["cudnn_mode"], "full")
        self.assertEqual(state["policy_skipped"], [])
        self.assertIn("cuDNN:filter=53", {item["item"] for item in state["missing"]})

    def test_validated_imported_tail_measurement_fills_another_parts_skip(self):
        parts = self.fast_parts()
        owner = with_policy(parts[0], ASSEMBLE.MEASURE.FIGURE4, (3, 9, 53), ("ThunderFFT",))
        with (owner / "data.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        tail = dict(next(row for row in rows if row["filter"] == "53"), library="cuDNN", filter_fft_included="n/a")
        external = self.root / "measured-tail.csv"
        ASSEMBLE.MEASURE.TABLE3.write_csv(external, [tail], ASSEMBLE.MEASURE.FIELDS)
        imported, hashes, origins = ASSEMBLE.MEASURE.snapshot_backends(
            [external], owner, ASSEMBLE.MEASURE.FIGURE4, "A100", "L=4096,B=4,D=64", (3, 9, 53), ("ThunderFFT",))
        metadata = json.loads((owner / "metadata.json").read_text())
        metadata.update(backend_sha256=hashes, backend_imports=origins)
        (owner / "metadata.json").write_text(json.dumps(metadata))
        ASSEMBLE.MEASURE.TABLE3.write_csv(owner / "data.csv", rows + imported, ASSEMBLE.MEASURE.FIELDS)
        result = ASSEMBLE.assemble(parts, self.root / "out", "smoke")
        self.assertEqual((result["status"], result["policy_skipped"]), ("complete", []))
        self.assertTrue(result["coverage_complete"])

    def test_policy_skip_does_not_mask_failed_native_command(self):
        parts = self.fast_parts()
        metadata = json.loads((parts[-1] / "metadata.json").read_text())
        metadata["commands"][0]["exit_code"] = 124
        (parts[-1] / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "successfully"):
            ASSEMBLE.assemble(parts, self.root / "out", "smoke")

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


class PlotPolicy(unittest.TestCase):
    def test_plot_keeps_nan_tail_and_explains_fast_coverage(self):
        import matplotlib.axes
        import matplotlib.figure
        original_plot = matplotlib.axes.Axes.plot
        original_text = matplotlib.figure.Figure.text
        lines, captions = [], []
        def capture_plot(axis, *args, **kwargs):
            lines.append((kwargs.get("label"), args[1]))
            return original_plot(axis, *args, **kwargs)
        def capture_text(figure, *args, **kwargs):
            captions.append(args[2])
            return original_text(figure, *args, **kwargs)
        rows = [{"library": "cuDNN", "filter": size, "time_ms": elapsed}
                for size, elapsed in ((3, 1.0), (9, 2.0))]
        skips = ASSEMBLE.MEASURE.POLICY.skipped_cells("figure4", (3, 9, 53), ("cuDNN",), "fast")
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(matplotlib.axes.Axes, "plot", capture_plot), \
                patch.object(matplotlib.figure.Figure, "text", capture_text):
            output = Path(temporary) / "figure.pdf"
            ASSEMBLE.PLOT.make_plot(rows, output, "figure4", policy_skipped=skips)
            self.assertTrue(output.read_bytes().startswith(b"%PDF"))
        self.assertTrue(math.isnan(next(values for label, values in lines if label == "cuDNN")[-1]))
        self.assertTrue(any("cuDNN tail omitted" in text and "--full" in text for text in captions))


if __name__ == "__main__":
    unittest.main()
