#!/usr/bin/env python3
"""CPU checks: native gates, cuFFTDx selection and normalization (synthetic input)."""

import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


MEASURE = module("figure3_measure", HERE / "measure.py")
PLOT = module("figure3_plot", HERE / "plot.py")


def native(library="copy", trials=(0,), e2e=50.0, rel=0.0, precision="tf32", **changes):
    fields = MEASURE.TABLE3.RAW_FIELDS + (("library",) if library != "thunderfft" else ())
    output = io.StringIO()
    writer = csv.DictWriter(output, fields)
    writer.writeheader()
    for trial in trials:
        row = dict(zip(MEASURE.TABLE3.RAW_FIELDS, (64, precision, MEASURE.NATIVE[library][1], 1024, 64, trial,
                                                   100, 200, 100 / 64, e2e, rel, 0)), library=library)
        row.update(changes)
        writer.writerow({key: row[key] for key in fields})
    return output.getvalue()


def parse(text, library="copy", trials=1):
    return MEASURE.parse_native(text, library=library, n=64, precision="tf32", batch=1024, repeats=64,
                                trials=trials, gpu="A100", raw_csv="raw/x.csv")


class NativeContract(unittest.TestCase):
    def test_valid_rows_and_bytes(self):
        row = parse(native())[0]
        self.assertEqual((row["library"], row["time_ms"], row["bytes"]), ("Comm. only", 0.05, 2 * 64 * 1024 * 8))
        self.assertEqual(parse(native("thunderfft", rel=0.001), "thunderfft")[0]["library"], "ThunderFFT")

    def test_failure_gates(self):
        for text, library, trials in ((native(rel=0.001), "copy", 1),             # copy must be exact
                                      (native("thunderfft", rel=0.006), "thunderfft", 1),
                                      (native(e2e=0), "copy", 1), (native(e2e="nan"), "copy", 1),
                                      (native(N=256), "copy", 1), (native(), "copy", 2),
                                      (native(trials=(0, 0)), "copy", 2),
                                      (native(library="copy-scalar"), "copy", 1)):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse(text, library, trials)


class Normalization(unittest.TestCase):
    def rows(self):
        rows = parse(native(e2e=50.0)) + parse(native("thunderfft", e2e=52.0, rel=0.001), "thunderfft")
        return rows + parse(native("copy-scalar", e2e=55.0), "copy-scalar")

    def test_divides_by_vector_copy_with_matching_bytes(self):
        summary = {row["library"]: row for row in MEASURE.summarize(self.rows())}
        self.assertEqual(summary["Comm. only"]["normalized"], 1.0)
        self.assertAlmostEqual(summary["ThunderFFT"]["normalized"], 50.0 / 52.0)
        self.assertAlmostEqual(summary["Comm. only (scalar)"]["normalized"], 50.0 / 55.0)

    def test_cufft_rows_pair_with_tf32_copy_bytes(self):
        text = native("cufft", e2e=62.5, rel=1e-6, precision="fp32")
        row = MEASURE.parse_native(text, library="cufft", n=64, precision="fp32", batch=1024, repeats=64,
                                   trials=1, gpu="A100", raw_csv="raw/c.csv")[0]
        summary = {r["library"]: r for r in MEASURE.summarize(self.rows() + [row])}
        self.assertAlmostEqual(summary["cuFFT"]["normalized"], 50.0 / 62.5)
        with self.assertRaises(ValueError):
            MEASURE.parse_native(native("cufft", rel=0.006, precision="fp32"), library="cufft", n=64,
                                 precision="fp32", batch=1024, repeats=64, trials=1, gpu="A100", raw_csv="x")

    def test_cufftdx_selects_fastest_median_e2e_candidate(self):
        fields = MEASURE.TABLE3.CUFFTDX_RAW_FIELDS
        output = io.StringIO()
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for candidate, e2e in (("block_default", 60.0), ("block_f2_e16", 55.0)):
            writer.writerow(dict(zip(fields, (64, "fp32", "register", 1024, 64, 0, 100, 164, 1, e2e, 0.001, 8192,
                                              "cufftdx", candidate, 2, 16, 64))))
        best, rows = MEASURE.parse_cufftdx(output.getvalue(), n=64, precision="fp32", batch=1024, repeats=64,
                                           trials=1, gpu="A100", raw_csv="raw/dx.csv")
        self.assertEqual((best, rows[0]["time_ms"], rows[0]["library"]), ("block_f2_e16", 0.055, "cuFFTDx"))

    def test_plot_leaves_missing_baselines_as_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            summary = MEASURE.summarize(self.rows())
            MEASURE.TABLE3.write_csv(output / "summary.csv", summary, MEASURE.SUMMARY_FIELDS)
            rows = PLOT.read_summary(output / "summary.csv")
            self.assertEqual({row["library"] for row in rows}, {"Comm. only", "ThunderFFT"})
            try:
                PLOT.make_plot(rows, output / "figure.pdf")
            except PLOT.Pending:
                self.skipTest("matplotlib unavailable")
            self.assertGreater((output / "figure.pdf").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
