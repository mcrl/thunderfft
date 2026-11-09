#!/usr/bin/env python3
"""Figure 4 uses the same validated split-collection contract as Figure 5."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


ASSEMBLE = module("figure4_assemble_test", HERE / "assemble.py")
FIXTURE = module("figure4_fixture", HERE.parent / "figure5/test_assemble.py")


class Assembly(unittest.TestCase):
    def test_old_figure4_one_dimensional_bundle_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = FIXTURE.part(root, "old-figure4")
            for name in ("metadata.json", "status.json"):
                path = old / name
                metadata = json.loads(path.read_text())
                metadata["evaluation"] = "figure4"
                path.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "operation contract differs"):
                ASSEMBLE.assemble([old], root / "out", "smoke")

    def test_all_four_libraries_are_supported(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ASSEMBLE.SHARED.PLOT, "make_plot") as plot:
            root = Path(directory)
            spec = ASSEMBLE.MEASURE.FIGURE4
            parts = [FIXTURE.part(root, lib, library=lib, filters=(3,33,53), spec=spec) for lib in spec["libraries"]]
            self.assertEqual(ASSEMBLE.assemble(parts, root / "out", "smoke")["status"], "complete")
            with (root / "out/summary.csv").open() as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0], ["filter", "cudnn", "cufft", "cufftdx", "thunderfft"])
            self.assertEqual([row[0] for row in rows[1:]], ["3", "33", "53"])
            self.assertTrue(all(cell for row in rows[1:] for cell in row))
            self.assertEqual(plot.call_args.kwargs["measured_gpu"], "A100")
            metadata = json.loads((root / "out/metadata.json").read_text())
            self.assertEqual(metadata["identity"]["gpu"], "A100")
            self.assertEqual(metadata["paper_gpu"], "A100")

    def test_fast_mode_skips_only_cudnn_tail_and_keeps_other_libraries(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(ASSEMBLE.SHARED.PLOT, "make_plot"):
            root = Path(directory)
            spec = ASSEMBLE.MEASURE.FIGURE4
            parts = [FIXTURE.part(root, lib, library=lib, filters=(3, 33, 53), spec=spec)
                     for lib in spec["libraries"] if lib != "cuDNN"]
            parts.append(FIXTURE.with_policy(FIXTURE.part(root, "cuDNN", library="cuDNN", filters=(3,), spec=spec),
                                             spec, (3, 33, 53), ("cuDNN",)))
            result = ASSEMBLE.assemble(parts, root / "out", "smoke")
            self.assertEqual((result["status"], result["missing"]), ("complete", []))
            self.assertEqual([item["item"] for item in result["policy_skipped"]],
                             ["cuDNN:filter=33", "cuDNN:filter=53"])
            self.assertFalse(result["coverage_complete"])


if __name__ == "__main__":
    unittest.main()
