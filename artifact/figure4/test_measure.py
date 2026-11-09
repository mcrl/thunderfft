#!/usr/bin/env python3
"""CPU checks: native gate parsing, backend semantics and plotting (synthetic input)."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


MEASURE = module("figure4_measure", HERE / "measure.py")
PLOT = module("figure4_plot", HERE / "plot.py")
NATIVE = ("Filter size: 3\n"
          "Validation N=4096 vs direct circular conv (raw/N): L2_rel=0.00096 max_abs=1e-4\n"
          "Validation sequences: 1025 of 49152\n"
          "Circular convolution validation: PASS (raw/N relative L2 limit=0.005)\n"
          "ThunderFFT full FFT-Conv (N=4096, include_filter_fft=1, K=3, trial=0) time: 1.861 ms\n"
          "ThunderFFT full FFT-Conv (N=4096, include_filter_fft=1, K=3, trial=1) time: 1.862 ms\n")
SHAPE = "L=4096,B=64,D=768"


class NativeContract(unittest.TestCase):
    def test_parse_and_gate(self):
        self.assertEqual(MEASURE.figure4_parse(NATIVE), [(3, 0, 1.861, 0.00096, "1025/49152 sequences"),
                                                         (3, 1, 1.862, 0.00096, "1025/49152 sequences")])
        two = NATIVE + NATIVE.replace("Filter size: 3", "Filter size: 5").replace("K=3", "K=5")
        self.assertEqual([r[0] for r in MEASURE.figure4_parse(two)], [3, 3, 5, 5])
        for text in (NATIVE.replace("PASS", "FAIL"), NATIVE.replace(" time: 1.861 ms", "").replace(" time: 1.862 ms", ""),
                     NATIVE.replace("Validation sequences: 1025 of 49152\n", "")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MEASURE.figure4_parse(text)

    def test_cudnn_output_parses_with_same_gate(self):
        text = ("Filter size: 7\nValidation N=4096 vs direct circular conv (cuDNN): L2_rel=0.0004\n"
                "Validation sequences: 1025 of 49152\nCircular convolution validation: PASS\n"
                "cuDNN depthwise circular conv1d (N=4096, K=7, trial=0) time: 2.5 ms\n")
        self.assertEqual(MEASURE.figure4_parse(text), [(7, 0, 2.5, 0.0004, "1025/49152 sequences")])
        argv, shape = MEASURE.figure4_cudnn_command(Path("/bin/c"), [3, 7], "paper", 3)
        self.assertEqual((argv[1:6], argv[-1], shape), (["--figure4", "4096", "3,7", "64", "768"], "3", SHAPE))

    def test_library_selection_and_flash_contract(self):
        with patch.dict("os.environ", {"THUNDERFFT_FIGURE4_LIBRARIES": "FlashFFTConv"}):
            self.assertEqual(MEASURE.selected_libraries(MEASURE.FIGURE4, ("ThunderFFT", "FlashFFTConv")), ("FlashFFTConv",))
            with self.assertRaisesRegex(ValueError, "unavailable"):
                MEASURE.selected_libraries(MEASURE.FIGURE4, ("ThunderFFT",))
        argv, shape = MEASURE.figure4_flash_command(Path("/flash"), [3, 53], "paper", 3)
        self.assertEqual((argv[1:4], shape), (["3,53", "64", "768"], SHAPE))
        text = NATIVE.replace("(raw/N)", "(FlashFFTConv)")
        self.assertEqual(MEASURE.figure4_parse(text)[0][3], 0.00096)

    def test_paper_command_uses_paper_workload_and_gate(self):
        argv, shape = MEASURE.figure4_command(Path("/bin/x"), [53], "paper", 3)
        self.assertEqual(shape, SHAPE)
        self.assertEqual(argv[1:6] + argv[8:10] + argv[-1:], ["4096", "53", "0", "64", "768", "1", "1", "3"])


class DeviceIdentity(unittest.TestCase):
    def test_numeric_and_unset_visibility_use_cuda_probe(self):
        first = "GPU-11111111-2222-3333-4444-555555555555"
        second = "GPU-99999999-2222-3333-4444-555555555555"
        log = "GPU=NVIDIA A100 compute=8.0 batch=256 seed=42\n"
        metadata = {"gpu_inventory_before": {"exit_code": 0, "stdout":
                    f"0, {first}, NVIDIA A100, 580\n1, {second}, NVIDIA A100, 580\n"},
                    "device_uuid_probe": {"exit_code": 0, "stdout": f"GPU_UUID={second}\n"}}
        for mask in (None, "0", "1"):
            with self.subTest(mask=mask):
                metadata["effective_CUDA_VISIBLE_DEVICES"] = mask
                self.assertEqual(MEASURE.measured_identity(metadata, log)[3:], (second, "580"))
        metadata["device_uuid_probe"]["exit_code"] = 1
        with self.assertRaisesRegex(ValueError, "unambiguously"):
            MEASURE.measured_identity(metadata, log)

    def test_whitespace_library_selector(self):
        with patch.dict("os.environ", {"THUNDERFFT_FIGURE4_LIBRARIES": " ThunderFFT , FlashFFTConv "}):
            self.assertEqual(MEASURE.selected_libraries(MEASURE.FIGURE4, ("ThunderFFT", "FlashFFTConv")),
                             ("ThunderFFT", "FlashFFTConv"))
        for text in ("ThunderFFT,,FlashFFTConv", "Flash FFTConv", "ThunderFFT,"):
            with self.subTest(text=text), patch.dict("os.environ", {"THUNDERFFT_FIGURE4_LIBRARIES": text}):
                with self.assertRaises(ValueError):
                    MEASURE.selected_libraries(MEASURE.FIGURE4, MEASURE.FIGURE4["libraries"])


class Backends(unittest.TestCase):
    def write(self, path, **changes):
        row = {"gpu": "A100", "shape": SHAPE, "filter": 3, "library": "FlashFFTConv", "repeat": 0,
               "time_ms": 2.14, "rel_l2": 1e-3, "passed": "true", "operation": MEASURE.FIGURE4["operation"]}
        row.update(changes)
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_join_summary_and_plot(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.write(output / "flash.csv")
            rows, _ = MEASURE.read_backends([output / "flash.csv"], MEASURE.FIGURE4, "A100", SHAPE, (3, 5))
            ours = [{"gpu": "A100", "shape": SHAPE, "filter": size, "library": "ThunderFFT", "time_ms": time}
                    for size, time in ((3, 1.8), (5, 2.0))]
            summary = MEASURE.summarize(ours + rows)
            mean = [row for row in summary if row["library"] == "ThunderFFT" and row["filter"] == "mean(2)"]
            self.assertAlmostEqual(mean[0]["median_time_ms"], 1.9)
            MEASURE.TABLE3.write_csv(output / "summary.csv", summary, MEASURE.SUMMARY_FIELDS)
            try:
                libraries = PLOT.make_plot(PLOT.read_summary(output / "summary.csv"), output / "figure.pdf", "figure4")
            except PLOT.Pending:
                self.skipTest("matplotlib unavailable")
            self.assertEqual(libraries, ["FlashFFTConv", "ThunderFFT"])

    def test_divergence_rule_cuts_only_far_slower_library(self):
        rows = [{"filter": 19, "library": "cuFFTDx", "time_ms": 7.5}, {"filter": 19, "library": "cuFFT", "time_ms": 14.7},
                {"filter": 19, "library": "cuDNN", "time_ms": 76.5}, {"filter": 17, "library": "cuDNN", "time_ms": 900}]
        self.assertTrue(MEASURE.diverged(rows, "cuDNN", 19, 10))
        self.assertFalse(MEASURE.diverged(rows, "cuDNN", 19, 11))
        self.assertFalse(MEASURE.diverged(rows, "cuDNN", 19, 0))     # 0 disables truncation
        self.assertFalse(MEASURE.diverged(rows, "cuDNN", 17, 10))    # nothing to compare at f=17

    def test_rejects_mismatched_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b.csv"
            for changes in ({"passed": "false"}, {"gpu": "H100"}, {"shape": "L=4096,B=1,D=1"}, {"filter": 7},
                            {"operation": "cross-correlation"}, {"rel_l2": 0.01}, {"time_ms": "nan"},
                            {"library": "ThunderFFT"}):
                self.write(path, **changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    MEASURE.read_backends([path], MEASURE.FIGURE4, "A100", SHAPE, (3, 5))


class CudnnWorkSelection(unittest.TestCase):
    def collect(self, spec, mode=None, overrides=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        binary = directory / "native"
        binary.write_bytes(b"synthetic native fixture")
        calls = []
        def command(path, sizes, profile, trials):
            return [str(path), ",".join(map(str, sizes)), str(trials)], spec.get("shape", "N=16384")
        def native(argv, **options):
            calls.append(argv)
            values = [(int(size), trial, 1.0, 0.001, "checked")
                      for size in argv[1].split(",") for trial in range(int(argv[2]))]
            options["stdout"].write(json.dumps(values))
            options["stderr"].write("native fixture\n")
            return SimpleNamespace(returncode=0)
        runners = [(library, binary, command, json.loads, "n/a" if library == "cuDNN" else "true")
                   for library in spec["libraries"]]
        environment = dict(overrides or {})
        if mode is not None:
            environment["THUNDERFFT_CUDNN_MODE"] = mode
        with patch.dict("os.environ", environment, clear=True), \
                patch.object(MEASURE.TABLE3, "capture", return_value={}), \
                patch.object(MEASURE, "measured_identity", return_value=("A100", "NVIDIA A100", "8.0", "GPU-fixture", "580")), \
                patch.object(MEASURE.subprocess, "run", side_effect=native):
            result = MEASURE.collect(spec, runners, directory, directory / "result", "paper", [HERE / "policy.py"])
        self.assertEqual(result, 0)
        metadata = json.loads((directory / "result/metadata.json").read_text())
        with (directory / "result/data.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        return metadata, rows

    def test_default_batches_only_cudnn_prefix_for_each_figure(self):
        figure5 = module("fast_cudnn_figure5", HERE.parent / "figure5/measure.py").FIGURE5
        for spec, limit, count in ((MEASURE.FIGURE4, 25, 192), (figure5, 11, 249)):
            with self.subTest(evaluation=spec["evaluation"]):
                metadata, rows = self.collect(spec)
                self.assertEqual(metadata["settings"]["cudnn_mode"], "fast")
                self.assertEqual(len(rows), count)
                self.assertEqual(len(metadata["commands"]), len(spec["libraries"]))
                self.assertEqual({int(row["filter"]) for row in rows if row["library"] == "cuDNN"},
                                 set(range(3, limit + 1, 2)))
                self.assertEqual(metadata["policy_skipped"], MEASURE.POLICY.skipped_cells(
                    spec["evaluation"], spec["filters"]["paper"], spec["libraries"], "fast"))
                self.assertFalse(metadata["coverage_complete"])

    def test_full_measures_every_filter_and_disables_ratio_cutoff(self):
        metadata, rows = self.collect(MEASURE.FIGURE4, "full", {"THUNDERFFT_FIGURE4_SLOW_RATIO": "0.1"})
        self.assertEqual(len(rows), 234)
        self.assertEqual(len(metadata["commands"]), 3)
        self.assertEqual(metadata["policy_skipped"], [])
        self.assertEqual(metadata["settings"]["slow_ratio"], 0)
        self.assertTrue(metadata["coverage_complete"])

    def test_all_omitted_cudnn_filters_do_not_start_a_native_process(self):
        metadata, rows = self.collect(MEASURE.FIGURE4, overrides={"THUNDERFFT_FIGURE4_FILTERS": "53"})
        self.assertEqual({c["library"] for c in metadata["commands"]}, {"ThunderFFT", "FlashFFTConv"})
        self.assertNotIn("cuDNN", metadata["binary_sha256"])
        self.assertEqual(len(rows), 6)
        with self.assertRaisesRegex(ValueError, "skips every"):
            self.collect(MEASURE.FIGURE4, overrides={"THUNDERFFT_FIGURE4_FILTERS": "53",
                                                   "THUNDERFFT_FIGURE4_LIBRARIES": "cuDNN"})

    def test_invalid_modes_and_ratio_are_rejected_before_launch(self):
        for mode, extra in (("invalid", {}), ("fast", {"THUNDERFFT_FIGURE4_SLOW_RATIO": "nan"}),
                            ("fast", {"THUNDERFFT_FIGURE4_SLOW_RATIO": "-1"})):
            with self.subTest(mode=mode, extra=extra), self.assertRaises(ValueError):
                self.collect(MEASURE.FIGURE4, mode, extra)


if __name__ == "__main__":
    unittest.main()
