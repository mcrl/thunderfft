"""Synthetic CPU checks of partial Figure 2 assembly and its measurement evidence."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ASSEMBLE = module("figure2_assembly_tests", HERE / "assemble.py")
FIXTURE = module("figure2_table3_fixture", HERE.parent / "table3/test_assemble.py")


class PartialFigureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def table3(self):
        part = FIXTURE.make_part(self.root / "part", ((64, "fp16", "register"), (256, "fp16", "register")))
        source = self.root / "assembled/table3"
        FIXTURE.ASSEMBLE.assemble([part], source, "smoke")
        return source

    def test_partial_graph_has_gaps_and_blank_missing_cells(self):
        source = self.table3()
        output = self.root / "assembled/figure2"
        import matplotlib.axes
        original = matplotlib.axes.Axes.errorbar
        captured = []
        def capture(axis, *args, **kwargs):
            captured.append(args[1])
            return original(axis, *args, **kwargs)
        with patch.object(matplotlib.axes.Axes, "errorbar", capture):
            result = ASSEMBLE.assemble([source], output, "smoke")
        self.assertEqual((result["status"], len(result["missing"])), ("partial", 13))
        self.assertTrue(math.isnan(captured[0][1]))
        self.assertTrue((output / "figure.pdf").read_bytes().startswith(b"%PDF"))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["source"]["source_result_dir"], "../table3")
        with (output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 14)
        self.assertTrue(all(row["median_us"] == "" for row in rows if row["status"] == "pending"))

    def test_tampering_assembled_data_does_not_bypass_native_validation(self):
        source = self.table3()
        data = source / "data.csv"
        data.write_text(data.read_text().replace(",1.0,4.0,", ",9.0,4.0,"))
        metadata = json.loads((source / "metadata.json").read_text())
        metadata["outputs"]["data.csv"] = ASSEMBLE.PLOT.sha256(data)
        (source / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "disagree"):
            ASSEMBLE.assemble([source], self.root / "output", "smoke")

    def test_empty_assembly_keeps_all_points_pending_without_a_fake_plot(self):
        output = self.root / "output"
        result = ASSEMBLE.assemble([], output, "smoke")
        self.assertEqual((result["status"], len(result["missing"])), ("pending", 15))
        self.assertFalse((output / "figure.pdf").exists())

    def test_source_hash_or_profile_change_is_rejected(self):
        source = self.table3()
        with self.assertRaisesRegex(ValueError, "profile/status"):
            ASSEMBLE.assemble([source], self.root / "wrong-profile", "paper")
        original = self.root / "part/raw/n64-fp16-register.log"
        original.write_text(original.read_text() + "changed\n")
        with self.assertRaisesRegex(ValueError, "changed after assembly"):
            ASSEMBLE.assemble([source], self.root / "changed", "smoke")


if __name__ == "__main__":
    unittest.main()
