"""CPU checks for calculated/measured Table 2 collection joins."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
import importlib.util
import shutil
import subprocess
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ASSEMBLE = module("table2_assembly_tests", HERE / "assemble.py")
FIXTURE = module("table2_table3_fixture", HERE.parent / "table3/test_assemble.py")


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_model_can_be_generated_without_a_gpu_or_collected_part(self):
        result = ASSEMBLE.assemble([], self.root / "model", "smoke")
        self.assertEqual((result["status"], result["collected_configurations"]), ("partial", 4))
        self.assertEqual(len(result["missing"]), 1)

    def test_raw_cufftdx_part_fills_measured_row_without_false_missing(self):
        part = FIXTURE.make_part(self.root / "dx", libraries=("cuFFTDx",))
        output = self.root / "table2"
        result = ASSEMBLE.assemble([part], output, "smoke")
        self.assertEqual((result["status"], result["missing"]), ("complete", []))
        self.assertFalse(result["full_paper_comparison"])
        with (output / "data.csv").open() as stream:
            measured = next(row for row in csv.DictReader(stream) if row["origin"] == "measured")
        self.assertEqual(measured["candidate"], "block_f4_e4")
        self.assertEqual(float(measured["measured_time_us"]), 1.5)
        self.assertAlmostEqual(float(measured["utilization_ceiling"]),
                               1 / float(measured["relative_time"]))

    def test_calculated_then_measured_parts_join_and_reassemble(self):
        model, dx = self.root / "model", FIXTURE.make_part(self.root / "dx", libraries=("cuFFTDx",))
        ASSEMBLE.assemble([], model, "smoke")
        result = ASSEMBLE.assemble([model, dx], self.root / "measured", "smoke")
        self.assertEqual(result["status"], "complete")
        # A copied model source and the explicitly supplied raw source are the
        # same evidence, not two additional trials.
        result = ASSEMBLE.assemble([model, self.root / "measured", dx], self.root / "joined", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_validated_table3_assembly_is_an_accepted_source(self):
        part = FIXTURE.make_part(self.root / "dx", libraries=("ThunderFFT", "cuFFTDx"))
        table3 = self.root / "assembled/table3"
        FIXTURE.ASSEMBLE.assemble([part], table3, "smoke")
        result = ASSEMBLE.assemble([table3], self.root / "assembled/table2", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_transferred_derived_bundle_is_independent_of_original_paths(self):
        raw = FIXTURE.make_part(self.root / "original/table3", libraries=("cuFFTDx",))
        derived = self.root / "original/table2"
        ASSEMBLE.assemble([raw], derived, "smoke")
        imported = self.root / "imported/table2"
        shutil.copytree(derived, imported)
        shutil.rmtree(self.root / "original")
        result = ASSEMBLE.assemble([imported], self.root / "new", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_derivation_from_transferred_whole_collection_uses_relative_table3_sources(self):
        collection = self.root / "original"
        part = FIXTURE.make_part(collection / "parts/a100/table3", libraries=("ThunderFFT", "cuFFTDx"))
        FIXTURE.ASSEMBLE.assemble([part], collection / "assembled/table3", "smoke")
        transferred = self.root / "transferred"
        shutil.copytree(collection, transferred)
        shutil.rmtree(collection)
        result = ASSEMBLE.assemble([transferred / "assembled/table3"], transferred / "assembled/table2", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_model_producer_embeds_native_evidence_for_downloads(self):
        raw = FIXTURE.make_part(self.root / "original/table3", libraries=("cuFFTDx",))
        FIXTURE.MEASURE.atomic_json(raw / "status.json", {"evaluation": "table3", "profile": "smoke",
                                                        "status": "completed", "exit_code": 0})
        derived = self.root / "original/table2"
        process = subprocess.run([sys.executable, str(HERE / "model.py"), "--profile", "smoke",
                                  "--source-dir", str(raw), "--output", str(derived)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        imported = self.root / "imported/table2"
        shutil.copytree(derived, imported)
        shutil.rmtree(self.root / "original")
        result = ASSEMBLE.assemble([imported], self.root / "new", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_legacy_absolute_source_uses_explicit_imported_table3_hash_match(self):
        raw = FIXTURE.make_part(self.root / "original/table3", libraries=("cuFFTDx",))
        derived = self.root / "original/table2"
        ASSEMBLE.assemble([raw], derived, "smoke")
        metadata = json.loads((derived / "metadata.json").read_text())
        metadata["measured_source"]["source_result_dir"] = str(raw)
        metadata["measured_source"].pop("snapshot_sha256")
        FIXTURE.MEASURE.atomic_json(derived / "metadata.json", metadata)
        shutil.rmtree(derived / "source")
        imported_raw, imported_model = self.root / "imported/table3", self.root / "imported/table2"
        shutil.copytree(raw, imported_raw)
        shutil.copytree(derived, imported_model)
        shutil.rmtree(self.root / "original")
        result = ASSEMBLE.assemble([imported_model, imported_raw], self.root / "new", "smoke")
        self.assertEqual(result["status"], "complete")

    def test_missing_reference_configuration_stays_partial(self):
        part = FIXTURE.make_part(self.root / "dx", ((128, "fp16", "register"),), libraries=("cuFFTDx",))
        result = ASSEMBLE.assemble([part], self.root / "table2", "smoke")
        self.assertEqual((result["status"], result["collected_configurations"]), ("partial", 4))

    def test_tampered_calculated_or_native_rows_fail(self):
        model = self.root / "model"
        ASSEMBLE.assemble([], model, "smoke")
        for name in ("data.csv", "table.csv"):
            (model / name).write_text((model / name).read_text().replace("Radix-4", "Radix-5"))
        with self.assertRaisesRegex(ValueError, "declared model"):
            ASSEMBLE.assemble([model], self.root / "output", "smoke")
        dx = FIXTURE.make_part(self.root / "dx", libraries=("cuFFTDx",))
        raw = dx / "raw/n64-fp16-register-cufftdx.csv"
        raw.write_text(raw.read_text().replace("0.0002", "0.1"))
        with self.assertRaisesRegex(ValueError, "L2 gate"):
            ASSEMBLE.assemble([dx], self.root / "output", "smoke")

    def test_conflicting_reference_measurements_are_not_averaged(self):
        first = FIXTURE.make_part(self.root / "first", libraries=("cuFFTDx",))
        second = FIXTURE.make_part(self.root / "second", libraries=("cuFFTDx",))
        metadata = json.loads((second / "metadata.json").read_text())
        metadata["run_id"] = "another-execution"
        FIXTURE.MEASURE.atomic_json(second / "metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "overlapping"):
            ASSEMBLE.assemble([first, second], self.root / "output", "smoke")


if __name__ == "__main__":
    unittest.main()
