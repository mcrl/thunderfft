"""CPU-only validation of profiler CSV parsing; never invokes Nsight Compute."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("ablation_profile", Path(__file__).with_name("collect_profile.py"))
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class ProfilerCSVChecks(unittest.TestCase):
    def fixture(self, value="75.5", unit="%", metric=None):
        return ('==PROF== Synthetic test input\n"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
                f'"0","test_kernel","{metric or module.DEFAULT_METRIC}","{unit}","{value}"\n')

    def test_explicit_percentage_metric(self):
        result = module.parse_metric(self.fixture(), module.DEFAULT_METRIC)
        self.assertEqual(result["value"], 75.5)
        self.assertEqual(result["metric"], module.DEFAULT_METRIC)

    def test_invalid_values_units_and_wrong_metric(self):
        for text in (self.fixture("nan"), self.fixture("inf"), self.fixture("-1"),
                     self.fixture("101"), self.fixture(unit="cycles"),
                     self.fixture(metric="different_metric"), "no metrics"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                module.parse_metric(text, module.DEFAULT_METRIC)

    def test_multiple_kernel_values_are_not_silently_averaged(self):
        text = self.fixture() + self.fixture().splitlines()[-1] + "\n"
        with self.assertRaises(ValueError):
            module.parse_metric(text, module.DEFAULT_METRIC)


if __name__ == "__main__":
    unittest.main()
