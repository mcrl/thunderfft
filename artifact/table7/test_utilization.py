#!/usr/bin/env python3
"""CPU checks of MMA counting, time units and hardware-peak normalization."""

import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("table7_utilization_test", Path(__file__).with_name("utilization.py"))
utilization = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(utilization)


class UtilizationChecks(unittest.TestCase):
    def test_a100_operations_time_and_percentage(self):
        row = utilization.calculate("register_uniform", 65536, 3.5, "NVIDIA A100 80GB PCIe")
        self.assertEqual(row["tensor_flops_per_fft"], 6144)
        self.assertEqual(row["tensor_flops_per_batch"], 402653184)
        self.assertAlmostEqual(row["achieved_tflops"], 115.04376685714286)
        self.assertAlmostEqual(row["utilization_pct"], 73.7460043956044)
        self.assertEqual(row["reference_peak_tflops"], 156)
        self.assertEqual(row["utilization_status"], "calculated")

    def test_same_operation_count_for_both_recovered_variants(self):
        a = utilization.calculate("register_branch", 65536, 7.9, "NVIDIA A100 80GB PCIe")
        b = utilization.calculate("register_uniform", 65536, 7.9, "NVIDIA A100 80GB PCIe")
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["utilization_pct"], 32.672280428432326)

    def test_time_and_batch_scaling_without_an_extra_repeat_multiplier(self):
        baseline = utilization.calculate("register_uniform", 64, 1, "A100")
        double_time = utilization.calculate("register_uniform", 64, 2, "A100")
        double_batch = utilization.calculate("register_uniform", 128, 1, "A100")
        self.assertAlmostEqual(baseline["achieved_tflops"], 0.393216)
        self.assertEqual(double_time["achieved_tflops"], baseline["achieved_tflops"] / 2)
        self.assertEqual(double_batch["achieved_tflops"], baseline["achieved_tflops"] * 2)

    def test_unconfigured_gpu_keeps_throughput_and_leaves_utilization_blank(self):
        row = utilization.calculate("register_uniform", 64, 1, "NVIDIA H100 PCIe")
        self.assertAlmostEqual(row["achieved_tflops"], 0.393216)
        self.assertEqual(row["utilization_pct"], "")
        self.assertEqual(row["reference_peak_tflops"], "")
        self.assertEqual(row["utilization_status"], "pending_reference")

    def test_unverified_variants_and_invalid_inputs_fail(self):
        for changes in ({"variant": "radix16"}, {"variant": "radix4_compute"},
                        {"batch": 0}, {"batch": 17}, {"batch": True},
                        {"time_us": 0}, {"time_us": float("nan")}, {"time_us": float("inf")},
                        {"precision": "fp16"}, {"gpu_name": ""}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                utilization.calculate(**{"variant": "register_uniform", "batch": 64, "time_us": 1,
                                         "gpu_name": "A100", **changes})

    def test_over_peak_value_is_not_clamped(self):
        row = utilization.calculate("register_uniform", 65536, 1, "A100")
        self.assertGreater(row["utilization_pct"], 100)


if __name__ == "__main__":
    unittest.main()
