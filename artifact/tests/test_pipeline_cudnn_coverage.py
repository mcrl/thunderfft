"""Fast-mode omissions are explicit limits; required measurements still fail."""

import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from artifact import full_pipeline
from artifact.figure4 import policy


class CudnnCoverageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name)
        self.entries, self.states = [], []
        for name in ("figure4", "figure5"):
            (self.output / name).mkdir()
            (self.output / name / "figure.pdf").write_bytes(b"synthetic output fixture")
            self.entries.append({"id": name, "required_outputs": {"paper": ["figure.pdf"]}})
            self.states.append({"evaluation": name, "status": "complete", "missing": [],
                "cudnn_mode": "fast", "full_paper_comparison": False,
                "policy_skipped": [{"item": f"cuDNN:filter={item['filter']}", "reason": item["reason"]}
                    for item in policy.skipped_cells(name, range(3, 54, 2), ["cuDNN"], "fast")]})

    def result(self, mode="fast", states=None):
        return full_pipeline.assembly_result(self.states if states is None else states,
            self.output, self.entries, "paper", ROOT, cudnn_mode=mode)

    def test_fast_succeeds_with_declared_limits_but_full_requires_all_points(self):
        result = self.result()
        self.assertTrue(result["outputs_complete"])
        self.assertEqual(len(result["coverage_limits"]), 35)
        self.assertFalse(result["full_paper_comparison"])
        full = self.result("full")
        self.assertFalse(full["outputs_complete"])
        self.assertEqual(len(full["unresolved"]), 35)

    def test_missing_required_measurements_are_not_excused(self):
        for item in ("cuDNN:filter=3", "ThunderFFT:filter=53"):
            states = copy.deepcopy(self.states)
            states[0].update(status="partial", missing=[{"item": item, "reason": "not measured"}])
            result = self.result(states=states)
            self.assertFalse(result["outputs_complete"])
            self.assertEqual(result["unresolved"][0]["item"], item)

    def test_unrecognized_or_duplicated_skip_declarations_fail(self):
        for change in ("early", "reason", "mode", "duplicate"):
            states = copy.deepcopy(self.states)
            if change == "early":
                states[0]["policy_skipped"][0]["item"] = "cuDNN:filter=3"
            elif change == "reason":
                states[0]["policy_skipped"][0]["reason"] = "native command timed out"
            elif change == "mode":
                states[0]["cudnn_mode"] = "full"
            else:
                states[0]["policy_skipped"].append(states[0]["policy_skipped"][0])
            with self.subTest(change=change):
                self.assertFalse(self.result(states=states)["outputs_complete"])


if __name__ == "__main__":
    unittest.main()
