#!/usr/bin/env python3
"""CPU checks: Table 6 native gates, candidate selection and speedups (synthetic input)."""

import csv
import importlib.util
import io
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


MEASURE = module("table6_measure", HERE / "measure.py")
REPEATS, BATCH = 64, 4096


def table6_native(library="tcfft", candidate="layer_256_0_A100", trials=(0,), compute=1.0,
                  rel=0.0005, **changes):
    output = io.StringIO()
    writer = csv.DictWriter(output, MEASURE.NATIVE_FIELDS)
    writer.writeheader()
    for trial in trials:
        row = {"N": 256, "precision": "fp16", "input": "complex",
               "placement": "register_fragments", "batch": BATCH, "inside_repeats": REPEATS, "trial": trial,
               "t_R_us": 100.0, "t_2R_us": 100.0 + compute * REPEATS, "compute_us": compute, "e2e_us": 10.0,
               "e2e_mode": "in_place" if library == "tcfft" else "out_of_place", "single_fft_relL2": rel,
               "shared_bytes": 2048, "library": library, "candidate": candidate,
               "ffts_per_block": 1 if candidate == "conv_kernel" else 32,
               "threads_per_block": 32}
        row.update(changes)
        writer.writerow(row)
    return output.getvalue()


def paper_fft(library="thunderfft", candidates=("",), trials=(0,), compute=1.0):
    table3 = MEASURE.TABLE3
    fields = table3.CUFFTDX_RAW_FIELDS if library == "cufftdx" else table3.RAW_FIELDS
    output = io.StringIO()
    writer = csv.DictWriter(output, fields)
    writer.writeheader()
    for index, candidate in enumerate(candidates):
        for trial in trials:
            value = compute * (1 + index)
            row = dict(zip(table3.RAW_FIELDS, (256, "fp16", "register", BATCH, REPEATS, trial, 100.0,
                                               100.0 + value * REPEATS, value, 10.0, 0.001, 17408)))
            row.update(library="cufftdx", candidate=candidate, ffts_per_block=16, elements_per_thread=16,
                       threads_per_block=256)
            writer.writerow({key: row[key] for key in fields})
    return output.getvalue()


def parse_native(text, library="tcfft", trials=1):
    return MEASURE.parse_native(text, library=library, batch=BATCH, repeats=REPEATS, trials=trials,
                                gpu="A100", revision="r")


def parse_paper(text, library="thunderfft", trials=1):
    return MEASURE.parse_paper_fft(text, library=library, batch=BATCH, repeats=REPEATS, trials=trials,
                                   gpu="A100", revision="r")


class NativeContract(unittest.TestCase):
    def test_valid_rows(self):
        row = parse_native(table6_native())[0]
        self.assertEqual((row["library"], row["variant"], row["time_ms"]), ("tcFFT", "layer_256_0_A100", 0.001))
        flash = parse_native(table6_native("flashfftconv", "conv_kernel"), "flashfftconv")[0]
        self.assertEqual((flash["library"], flash["variant"], flash["input"], flash["ffts_per_block"]),
                         ("FlashFFTConv", "conv_kernel", "complex", 1))

    def test_failure_gates(self):
        cases = ((table6_native(rel=0.006), "tcfft", 1),                      # accuracy gate
                 (table6_native(rel="nan"), "tcfft", 1),
                 (table6_native(compute=0), "tcfft", 1),                      # t_2R must exceed t_R
                 (table6_native(compute_us=5.0), "tcfft", 1),                 # inconsistent difference
                 (table6_native(e2e_us=0), "tcfft", 1),
                 (table6_native(N=512), "tcfft", 1), (table6_native(batch=8192), "tcfft", 1),
                 (table6_native(), "tcfft", 2), (table6_native(trials=(0, 0)), "tcfft", 2),
                 (table6_native(candidate="layer_256_0"), "tcfft", 1),                # V100 kernel
                 (table6_native(e2e_mode="out_of_place"), "tcfft", 1),
                 (table6_native("flashfftconv", "c2c_fpb64"), "flashfftconv", 1),        # removed path
                 (table6_native("flashfftconv", "conv_kernel_b8h8"), "flashfftconv", 1),
                 (table6_native("flashfftconv", "conv_kernel", input="real"), "flashfftconv", 1),
                 (table6_native("flashfftconv", "r2c_fpb1"), "flashfftconv", 1),
                 (table6_native(library="flashfftconv"), "tcfft", 1))
        for text, library, trials in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_native(text, library, trials)

    def test_header_contract(self):
        with self.assertRaises(ValueError):
            parse_native(table6_native().replace("e2e_mode,", "mode,", 1))


class Summary(unittest.TestCase):
    def test_selection_and_speedup(self):
        rows = parse_paper(paper_fft("cufftdx", ("block_f8_e16", "block_f16_e16"), compute=2.0), "cufftdx")
        rows += parse_paper(paper_fft(compute=1.6))
        rows += parse_native(table6_native(compute=1.8))
        flash = "".join(table6_native("flashfftconv", c, compute=v).split("\n", 1)[1] if i else
                        table6_native("flashfftconv", c, compute=v)
                        for i, (c, v) in enumerate((("conv_kernel", 1.8),)))
        rows += parse_native(flash, "flashfftconv")
        table = {(row["library"], row["variant"]): row for row in MEASURE.summarize(rows)}
        self.assertEqual(table[("cuFFTDx", "block_fft")]["selected_candidate"], "block_f8_e16")
        self.assertEqual(table[("cuFFTDx", "block_fft")]["speedup_vs_cufftdx"], 1.0)
        self.assertEqual(table[("ThunderFFT", "register")]["speedup_vs_cufftdx"], 1.25)
        self.assertEqual(table[("FlashFFTConv", "conv_kernel")]["selected_candidate"], "conv_kernel")
        # Paper values are attached only to the paper configuration (A100, batch 65536).
        self.assertEqual(table[("tcFFT", "layer_256_0_A100")]["paper_time_ms"], "")
        self.assertEqual([row["library"] for row in MEASURE.summarize(rows)][:2], ["cuFFTDx", "tcFFT"])

    def test_paper_values_on_paper_configuration(self):
        rows = [dict(row, batch=65536) for row in parse_native(table6_native(compute=1.8))]
        row = MEASURE.summarize(rows)[0]
        self.assertEqual((row["paper_time_ms"], row["paper_speedup"], row["speedup_vs_cufftdx"]), (0.021, 1.11, ""))


if __name__ == "__main__":
    unittest.main()
