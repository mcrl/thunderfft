#!/usr/bin/env python3
"""Contract checks: reject incomplete, corrupted and numerically invalid runs."""

import csv
import io
import unittest

from measure import RAW_FIELDS, gpu_name, parse_raw, summarize


class NativeOutputContract(unittest.TestCase):
    def raw(self, **changes):
        row = dict(zip(RAW_FIELDS, (64, "fp16", "register", 1024, 256, 0,
                                   100, 356, 1, 4, 0.0002, 1024)))
        row.update(changes)
        output = io.StringIO()
        writer = csv.DictWriter(output, RAW_FIELDS)
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue()

    def parse(self, text, **options):
        args = dict(n=64, precision="fp16", placement="register", batch=1024,
                    repeats=256, trials=1, gpu="A100", raw_csv="raw/example.csv")
        args.update(options)
        return parse_raw(text, **args)

    def test_valid_conversion_and_smoke_does_not_compare_paper(self):
        rows = self.parse(self.raw())
        self.assertEqual(rows[0]["time_us"], 1)
        self.assertEqual(rows[0]["passed"], "true")
        self.assertEqual(summarize(rows, "smoke")[0]["paper_thunderfft_target_us"], "")

    def test_failure_gates(self):
        for changes in ({"single_fft_relL2": "nan"}, {"single_fft_relL2": 0.006},
                        {"compute_us": 0}, {"compute_us": "inf"}, {"t_2R_us": 99},
                        {"compute_us": 2}, {"e2e_out_of_place_us": -1},
                        {"N": 128}, {"batch": 65536}, {"inside_repeats": 255},
                        {"trial": 1}, {"shared_bytes": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(self.raw(**changes))

    def test_incomplete_duplicate_and_malformed_csv(self):
        valid = self.raw()
        for text, options in ((valid, {"trials": 2}),
                              (valid + valid.splitlines()[1] + "\n", {}),
                              (valid + "64,fp16\n", {}), (valid.replace("compute_us", "time"), {})):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text, **options)

    def test_device_identity_never_defaults_unknown_gpu_to_a100(self):
        self.assertEqual(gpu_name("GPU=NVIDIA A100-PCIE-80GB compute=8.0 batch=1024 R=256\n")[0], "A100")
        self.assertEqual(gpu_name("GPU=Tesla V100 compute=7.0 batch=1024 R=256\n")[0], "Tesla V100")
        with self.assertRaises(ValueError):
            gpu_name("missing metadata")


if __name__ == "__main__":
    unittest.main()
