"""Synthetic CPU checks of partial Figure 2 assembly and its measurement evidence."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
import shutil
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
        self.assertEqual((result["status"], len(result["missing"])), ("partial", 26))
        self.assertTrue(math.isnan(captured[0][1]))
        self.assertTrue((output / "figure.pdf").read_bytes().startswith(b"%PDF"))
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["source"]["source_result_dir"], "../table3")
        with (output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 28)
        self.assertTrue(all(row["median_us"] == "" for row in rows if row["status"] == "pending"))

    def test_partial_cufftdx_series_use_the_selected_candidate(self):
        part = FIXTURE.make_part(self.root / "mixed", ((64, "fp16", "register"),),
                                 libraries=("ThunderFFT", "cuFFTDx"))
        source, output = self.root / "assembled/table3", self.root / "assembled/figure2"
        FIXTURE.ASSEMBLE.assemble([part], source, "smoke")
        result = ASSEMBLE.assemble([source], output, "smoke")
        self.assertEqual(result["collected_configurations"], 2)
        with (output / "data.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["candidate"] for row in rows if row["library"] == "cuFFTDx"}, {"block_f4_e4"})
        with (output / "summary.csv").open() as stream:
            summary = list(csv.DictReader(stream))
        matched = next(row for row in summary if row["library"] == "ThunderFFT" and row["N"] == "64" and row["precision"] == "FP16")
        self.assertEqual(float(matched["speedup_vs_cufftdx"]), 1)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["cufftdx_mode"], "all")

    def test_preset_assembly_keeps_source_contract_and_measured_values(self):
        part = FIXTURE.make_preset_part(self.root / "preset")
        source, output = self.root / "assembled/table3", self.root / "assembled/figure2"
        FIXTURE.ASSEMBLE.assemble([part], source, "smoke")
        result = ASSEMBLE.assemble([source], output, "smoke")
        self.assertEqual(result["collected_configurations"], 1)
        metadata = json.loads((output / "metadata.json").read_text())
        self.assertEqual(metadata["cufftdx_mode"], "preset")
        self.assertEqual(metadata["source"]["cufftdx_mode"], "preset")
        self.assertEqual(metadata["cufftdx_preset_sha256"],
                         FIXTURE.MEASURE.preset_descriptor("A100")["manifest_sha256"])
        self.assertIn("fixed preset candidates remeasured", metadata["cufftdx_selection_rule"])
        with (output / "summary.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        measured = [row for row in rows if row["status"] == "collected"]
        self.assertEqual(len(measured), 1)
        self.assertEqual(float(measured[0]["median_us"]), 3.0)
        self.assertTrue((output / "figure.pdf").read_bytes().startswith(b"%PDF"))

    def test_full_a100_register_matrix_completes_figure(self):
        configs = [(n, precision, "register") for n in FIXTURE.MEASURE.SIZES for precision in FIXTURE.MEASURE.PRECISIONS]
        part = FIXTURE.make_part(self.root / "mixed", configs, libraries=("ThunderFFT", "cuFFTDx"))
        source = self.root / "assembled/table3"
        FIXTURE.ASSEMBLE.assemble([part], source, "smoke")
        result = ASSEMBLE.assemble([source], self.root / "assembled/figure2", "smoke")
        self.assertEqual((result["status"], result["collected_configurations"], result["missing"]), ("complete", 28, []))
        self.assertFalse(result["full_paper_comparison"])

    def test_figure_regenerates_from_a_transferred_whole_collection(self):
        collection = self.root / "original"
        part = FIXTURE.make_part(collection / "parts/a100/table3", libraries=("ThunderFFT", "cuFFTDx"))
        FIXTURE.ASSEMBLE.assemble([part], collection / "assembled/table3", "smoke")
        transferred = self.root / "transferred"
        shutil.copytree(collection, transferred)
        shutil.rmtree(collection)
        output = transferred / "assembled/figure2"
        result = ASSEMBLE.assemble([transferred / "assembled/table3"], output, "smoke")
        self.assertEqual(result["collected_configurations"], 2)
        with (output / "data.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual({row["source_part"] for row in rows}, {"../../parts/a100/table3"})

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
        self.assertEqual((result["status"], len(result["missing"])), ("pending", 28))
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
