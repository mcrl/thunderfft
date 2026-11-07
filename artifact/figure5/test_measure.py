#!/usr/bin/env python3
"""CPU checks: Figure 5 native gate parsing and paper workload (synthetic input)."""

import importlib.util
from pathlib import Path
import unittest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("figure5_measure", HERE / "measure.py")
MEASURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEASURE)
NATIVE = ("Filter size: 53\n"
          "Validation 2D (cufftdx) vs direct reference: L2_rel=1.2e-06 samples=5000\n"
          "Convolution validation: PASS (relative L2 limit=0.005)\n"
          "cufftdx 2D conv (N=16384, f=53, filter_fft_included=1, trial=0) time: 42.5 ms\n")


class NativeContract(unittest.TestCase):
    def test_parse_and_gate(self):
        self.assertEqual(MEASURE.parse(NATIVE), [(53, 0, 42.5, 1.2e-06, "5000 sampled outputs")])
        for text in (NATIVE.replace("PASS", "FAIL"), NATIVE.replace(" time: 42.5 ms", "")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                MEASURE.parse(text)

    def test_paper_command(self):
        argv, shape = MEASURE.command("cufft")(Path("/bin/thff_conv"), [3, 53], "paper", 3)
        self.assertEqual((argv[1:4], argv[-1], shape), (["16384", "3,53", "cufft"], "3", "N=16384"))


if __name__ == "__main__":
    unittest.main()
