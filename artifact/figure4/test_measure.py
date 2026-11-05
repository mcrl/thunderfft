#!/usr/bin/env python3
"""CPU checks: native gate parsing, backend semantics and plotting (synthetic input)."""

import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


MEASURE = module("figure4_measure", HERE / "measure.py")
PLOT = module("figure4_plot", HERE / "plot.py")
NATIVE = ("Validation N=4096 vs direct circular conv (raw/N): L2_rel=0.00096 max_abs=1e-4\n"
          "Validation sequences: 1025 of 49152\n"
          "Circular convolution validation: PASS (raw/N relative L2 limit=0.005)\n"
          "ThunderFFT full FFT-Conv (N=4096, include_filter_fft=1) time: 1.861 ms\n")
SHAPE = "L=4096,B=64,D=768"


class NativeContract(unittest.TestCase):
    def test_parse_and_gate(self):
        self.assertEqual(MEASURE.figure4_parse(NATIVE), (1.861, 0.00096, "1025/49152 sequences"))
        for text in (NATIVE.replace("PASS", "FAIL"), NATIVE.replace("time: 1.861 ms\n", ""),
                     NATIVE.replace("Validation sequences: 1025 of 49152\n", "")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MEASURE.figure4_parse(text)

    def test_cudnn_output_parses_with_same_gate(self):
        text = ("Validation N=4096 vs direct circular conv (cuDNN): L2_rel=0.0004\n"
                "Validation sequences: 1025 of 49152\nCircular convolution validation: PASS\n"
                "cuDNN depthwise circular conv1d (N=4096) time: 2.5 ms\n")
        self.assertEqual(MEASURE.figure4_parse(text), (2.5, 0.0004, "1025/49152 sequences"))
        argv, shape = MEASURE.figure4_cudnn_command(Path("/bin/c"), 7, "paper")
        self.assertEqual((argv[1:6], shape), (["--figure4", "4096", "7", "64", "768"], SHAPE))

    def test_paper_command_uses_paper_workload_and_gate(self):
        argv, shape = MEASURE.figure4_command(Path("/bin/x"), 53, "paper")
        self.assertEqual(shape, SHAPE)
        self.assertEqual(argv[1:6] + argv[8:10], ["4096", "53", "0", "64", "768", "1", "1"])


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

    def test_rejects_mismatched_backends(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b.csv"
            for changes in ({"passed": "false"}, {"gpu": "H100"}, {"shape": "L=4096,B=1,D=1"}, {"filter": 7},
                            {"operation": "cross-correlation"}, {"rel_l2": 0.01}, {"time_ms": "nan"},
                            {"library": "ThunderFFT"}):
                self.write(path, **changes)
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    MEASURE.read_backends([path], MEASURE.FIGURE4, "A100", SHAPE, (3, 5))


if __name__ == "__main__":
    unittest.main()
