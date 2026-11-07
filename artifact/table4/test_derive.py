"""CPU checks for Table 3 to Table 4 conversion and portable collection parts."""

import csv
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("table4_assembly_tests", Path(__file__).with_name("assemble.py"))
ASSEMBLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLE)
DERIVE = ASSEMBLE.DERIVE


def make_part(path, gpu="A100", times=(2.0, 4.0), profile="smoke", batch=None, repeats=None, gpu_uuid=None):
    """Create synthetic completed Table 3 evidence without any GPU workload."""
    path.mkdir(parents=True)
    batch = batch if batch is not None else (65536 if profile == "paper" else 1024)
    repeats = repeats if repeats is not None else (1000 if profile == "paper" else 256)
    rows = [dict(gpu=gpu, N=64, batch=batch, placement="register", library="ThunderFFT",
                 precision="tf32", trial=index, time_us=time, e2e_out_of_place_us=17,
                 rel_l2=0.0002, passed="true", inside_repeats=repeats, t_R_us=100,
                 t_2R_us=100 + time * repeats, shared_bytes=8192,
                 raw_csv="raw/n64-tf32-register.csv", candidate="", ffts_per_block="",
                 elements_per_thread="", threads_per_block="") for index, time in enumerate(times)]
    DERIVE.write_csv(path / "data.csv", rows, DERIVE.SOURCE_FIELDS + DERIVE.SOURCE_EXTRA_FIELDS)
    metadata = {"evaluation": "table3", "profile": profile,
                      "status": "completed", "gpu": gpu,
                      "settings": {"batch": batch, "repeats": repeats, "trials": len(times)}}
    if gpu_uuid is not None:
        metadata["gpu_uuid"] = gpu_uuid
    DERIVE.write_json(path / "metadata.json", metadata)
    DERIVE.write_json(path / "status.json", {"evaluation": "table3", "profile": profile,
                      "status": "completed", "exit_code": 0})
    return path


UUIDS = {gpu: f"GPU-{index:08x}-1111-2222-3333-444444444444" for index, gpu in enumerate(DERIVE.GPUS, 1)}


def make_ideal(path, gpu="A100", gpu_uuid=None, profile="smoke", throughputs=(140, 150, 145, 130)):
    path.mkdir(parents=True)
    uuid = gpu_uuid or UUIDS[gpu]
    (path / "raw").mkdir()
    (path / "build").mkdir()
    (path / "raw/measurement.csv").write_text("Synthetic CPU fixture, no GPU measured\n")
    (path / "raw/measurement.log").write_text("Synthetic native log fixture\n")
    (path / "build/compile_commands.json").write_text("[]\n")
    (path / "disassembly.sass").write_text("Synthetic HMMA.1688.F32.TF32 fixture\n")
    rows = []
    for candidate, throughput in zip(DERIVE.IDEAL_CANDIDATES, throughputs):
        threads, chains = (int(value[1:]) for value in candidate.split("-"))
        instructions = 4 * (threads // 32) * chains * 64 * 32
        count = instructions * 2048
        for trial in range(2):
            rows.append(dict(gpu=gpu, gpu_uuid=uuid, candidate=candidate, trial=trial, blocks=4,
                             threads=threads, chains=chains, iterations=64, unroll=32,
                             mma_instructions=instructions, flop_count=count,
                             time_us=count / throughput / 1e6, tflops=throughput,
                             max_abs_error=0, passed="true", registers=32, shared_bytes=0, active_blocks_per_sm=2))
    DERIVE.write_csv(path / "ideal.csv", rows, DERIVE.IDEAL_FIELDS)
    metadata = {"schema_version": 1, "evaluation": "table4_ideal", "profile": profile, "status": "completed",
                "gpu": gpu, "gpu_name": f"NVIDIA {gpu}", "gpu_uuid": uuid, "compute_capability": "8.0",
                "settings": {"iterations": 64, "trials": 2, "warmups": 0, "waves": 2,
                             "candidates": list(DERIVE.IDEAL_CANDIDATES)}, "multiprocessor_count": 1,
                "source_sha256": {"ideal_mma.cu": "a" * 64}, "binary_sha256": "b" * 64,
                "outputs": {str(file.relative_to(path)): DERIVE.sha256(file) for file in path.rglob("*") if file.is_file()},
                "selected_candidate": min(zip(DERIVE.IDEAL_CANDIDATES, throughputs), key=lambda item: (-item[1], item[0]))[0]}
    DERIVE.write_json(path / "ideal_metadata.json", metadata)
    return path


def refresh_ideal_hash(path, filename="ideal.csv"):
    metadata = DERIVE.read_json(path / "ideal_metadata.json")
    metadata["outputs"][filename] = DERIVE.sha256(path / filename)
    DERIVE.write_json(path / "ideal_metadata.json", metadata)


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def mutate_row(path, key, value):
    values = rows(path / "data.csv")
    values[0][key] = value
    DERIVE.write_csv(path / "data.csv", values, DERIVE.SOURCE_FIELDS + DERIVE.SOURCE_EXTRA_FIELDS)


class DerivationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"

    def test_paper_round_trip_matches_all_three_reported_unit_values(self):
        parts = [make_part(self.root / gpu, gpu=gpu, times=(time,), profile="paper")
                 for gpu, time in (("A100", 3.55), ("H100", 2.57), ("B200", 1.99))]
        state = DERIVE.generate(parts, self.output, "paper")
        table = rows(self.output / "table.csv")
        self.assertEqual([round(float(row["unit_tflops"]), 2) for row in table], [113.42, 156.67, 202.34])
        self.assertEqual(state["collected_unit_gpus"], 3)
        self.assertEqual(len(state["missing"]), 3)
        self.assertTrue(state["complete_unit_subset"])
        self.assertFalse(state["complete_declared_subset"])
        self.assertFalse(state["full_paper_comparison"])
        self.assertTrue(all(row["ideal_mma_tflops"] == row["efficiency_pct"] == "" for row in table))

    def test_even_trial_median_is_taken_before_reciprocal(self):
        part = make_part(self.root / "part", times=(2.0, 4.0))
        DERIVE.generate([part], self.output, "smoke")
        row = rows(self.output / "table.csv")[0]
        self.assertEqual(float(row["median_time_us"]), 3.0)
        self.assertAlmostEqual(float(row["unit_tflops"]), 6144 * 1024 / 3 / 1e6)
        self.assertNotAlmostEqual(float(row["unit_tflops"]), (6144 * 1024 / 2 / 1e6 + 6144 * 1024 / 4 / 1e6) / 2)

    def test_missing_measurement_cells_are_blank_with_paper_references_labeled(self):
        state = DERIVE.generate([], self.output, "smoke")
        self.assertEqual((state["status"], len(state["missing"])), ("pending", 6))
        table = rows(self.output / "table.csv")
        self.assertEqual([float(row["hardware_peak_tflops"]) for row in table], [156, 378, 1100])
        self.assertTrue(all(row["unit_tflops"] == row["median_time_us"] == "" for row in table))
        self.assertTrue(all(row["hardware_peak_source"] == "paper_reference" for row in table))

    def test_only_exact_target_is_selected(self):
        part = make_part(self.root / "part")
        original = rows(part / "data.csv")
        extra = [{**original[0], **overrides} for overrides in (
            {"N": 128}, {"precision": "fp16"}, {"placement": "shared"},
            {"library": "cuFFTDx", "time_us": "nan", "passed": "false"})]
        DERIVE.write_csv(part / "data.csv", original + extra, DERIVE.SOURCE_FIELDS + DERIVE.SOURCE_EXTRA_FIELDS)
        DERIVE.generate([part], self.output, "smoke")
        self.assertEqual(len(rows(self.output / "data.csv")), 2)

    def test_unrelated_interrupted_source_is_skipped_before_status_check(self):
        part = make_part(self.root / "part")
        values = rows(part / "data.csv")
        for value in values:
            value["precision"] = "fp16"
        DERIVE.write_csv(part / "data.csv", values, DERIVE.SOURCE_FIELDS + DERIVE.SOURCE_EXTRA_FIELDS)
        (part / "status.json").unlink()
        state = ASSEMBLE.assemble([part], self.output, "smoke")
        self.assertEqual(state["status"], "pending")

    def test_selected_interrupted_source_rejected(self):
        part = make_part(self.root / "part")
        status = DERIVE.read_json(part / "status.json")
        status["status"] = "interrupted"
        DERIVE.write_json(part / "status.json", status)
        with self.assertRaisesRegex(ValueError, "completed"):
            DERIVE.generate([part], self.output, "smoke")

    def test_source_metadata_and_runner_profile_and_exit_must_match(self):
        for filename, field, value in (("metadata.json", "profile", "paper"),
                                      ("status.json", "evaluation", "table5"),
                                      ("metadata.json", "status", "running"),
                                      ("status.json", "exit_code", 1)):
            with self.subTest(filename=filename, field=field):
                part = make_part(self.root / (filename + field))
                record = DERIVE.read_json(part / filename)
                record[field] = value
                DERIVE.write_json(part / filename, record)
                with self.assertRaises(ValueError):
                    DERIVE.generate([part], self.output, "smoke")

    def test_invalid_finite_and_correctness_fields_rejected(self):
        for index, (key, value) in enumerate((("time_us", "nan"), ("time_us", "0"),
                    ("t_R_us", "inf"), ("t_2R_us", "-1"), ("rel_l2", "nan"),
                    ("rel_l2", "0.006"), ("passed", "false"), ("batch", "17"),
                    ("inside_repeats", "0"), ("gpu", "H100"))):
            with self.subTest(key=key, value=value):
                part = make_part(self.root / str(index))
                mutate_row(part, key, value)
                with self.assertRaises(ValueError):
                    DERIVE.generate([part], self.output, "smoke")

    def test_differential_time_must_match(self):
        part = make_part(self.root / "part")
        mutate_row(part, "time_us", "99")
        with self.assertRaisesRegex(ValueError, "differential"):
            DERIVE.generate([part], self.output, "smoke")

    def test_duplicate_and_noncontiguous_trials_rejected(self):
        for value in ("1", "-1", "2"):
            with self.subTest(trial=value):
                part = make_part(self.root / value)
                mutate_row(part, "trial", value)
                with self.assertRaisesRegex(ValueError, "trial"):
                    DERIVE.generate([part], self.output, "smoke")

    def test_incomplete_trials_rejected(self):
        part = make_part(self.root / "part")
        DERIVE.write_csv(part / "data.csv", rows(part / "data.csv")[:1], DERIVE.SOURCE_FIELDS + DERIVE.SOURCE_EXTRA_FIELDS)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            DERIVE.generate([part], self.output, "smoke")

    def test_paper_settings_are_enforced(self):
        for index, settings in enumerate(({"batch": 1024}, {"repeats": 256}, {"batch": 17})):
            with self.subTest(settings=settings):
                part = make_part(self.root / str(index), profile="paper", **settings)
                with self.assertRaises(ValueError):
                    DERIVE.generate([part], self.output, "paper")

    def test_legacy_table3_columns_are_accepted(self):
        part = make_part(self.root / "part")
        values = [{key: value for key, value in row.items() if key in DERIVE.SOURCE_FIELDS}
                  for row in rows(part / "data.csv")]
        DERIVE.write_csv(part / "data.csv", values, DERIVE.SOURCE_FIELDS)
        self.assertEqual(DERIVE.generate([part], self.output, "smoke")["collected_unit_gpus"], 1)

    def test_unused_table3_columns_are_optional(self):
        part = make_part(self.root / "part")
        fields = tuple(key for key in DERIVE.SOURCE_FIELDS if key in DERIVE.REQUIRED_SOURCE_FIELDS)
        values = [{key: value for key, value in row.items() if key in fields}
                  for row in rows(part / "data.csv")]
        DERIVE.write_csv(part / "data.csv", values, fields)
        self.assertEqual(DERIVE.generate([part], self.output, "smoke")["collected_unit_gpus"], 1)

    def test_malformed_header_and_rows_are_rejected(self):
        for index, transform in enumerate((lambda text: text.replace("time_us", "invalid", 1),
                                          lambda text: text + "broken,row\n",
                                          lambda text: text.replace("gpu,N,", "gpu,gpu,"))):
            with self.subTest(index=index):
                part = make_part(self.root / str(index))
                file = part / "data.csv"
                file.write_text(transform(file.read_text()))
                with self.assertRaises(ValueError):
                    DERIVE.generate([part], self.output, "smoke")

    def test_repeated_path_deduplicates_but_distinct_gpu_source_conflicts(self):
        part = make_part(self.root / "part")
        state = DERIVE.generate([part, part / "."], self.output, "smoke")
        self.assertEqual(state["collected_unit_gpus"], 1)
        copied = self.root / "copied"
        shutil.copytree(part, copied)
        state = DERIVE.generate([part, copied], self.output, "smoke")
        self.assertEqual(state["collected_unit_gpus"], 1)
        other = make_part(self.root / "other", times=(3.0, 5.0))
        with self.assertRaisesRegex(ValueError, "conflicting A100"):
            DERIVE.generate([part, other], self.output, "smoke")

    def test_snapshots_relocate_and_survive_original_deletion(self):
        part = make_part(self.root / "part")
        DERIVE.generate([part], self.output, "smoke")
        shutil.rmtree(part)
        moved = self.root / "relocated"
        self.output.rename(moved)
        state = ASSEMBLE.assemble([moved], self.root / "merged", "smoke")
        self.assertEqual(state["collected_unit_gpus"], 1)
        self.assertEqual(rows(moved / "table.csv"), rows(self.root / "merged/table.csv"))
        self.assertEqual(sorted(path.name for path in (moved / "inputs/table3-000").iterdir()), sorted(DERIVE.SOURCE_FILES))

    def test_mixed_direct_and_derived_parts_deduplicate_identical_snapshots(self):
        a = make_part(self.root / "a")
        h = make_part(self.root / "h", gpu="H100")
        DERIVE.generate([a], self.output, "smoke")
        state = ASSEMBLE.assemble([a, h, self.output], self.root / "merged", "smoke")
        self.assertEqual(state["collected_unit_gpus"], 2)

    def test_assembly_rejects_tampered_snapshot_and_derived_csv(self):
        part = make_part(self.root / "part")
        for index, relative in enumerate(("inputs/table3-000/data.csv", "table.csv")):
            with self.subTest(relative=relative):
                target = self.root / str(index)
                DERIVE.generate([part], target, "smoke")
                file = target / relative
                file.write_text(file.read_text() + "\n")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    ASSEMBLE.assemble([target], self.root / "merged", "smoke")

    def test_escaping_snapshot_paths_are_rejected(self):
        part = make_part(self.root / "part")
        DERIVE.generate([part], self.output, "smoke")
        metadata = DERIVE.read_json(self.output / "metadata.json")
        for relative in ("../part", str(part)):
            with self.subTest(relative=relative):
                metadata["sources"][0]["snapshot"] = relative
                DERIVE.write_json(self.output / "metadata.json", metadata)
                with self.assertRaisesRegex(ValueError, "escaping"):
                    ASSEMBLE.assemble([self.output], self.root / "merged", "smoke")

    def test_input_symlink_escape_is_rejected(self):
        part = make_part(self.root / "part")
        target = self.root / "data.csv"
        (part / "data.csv").rename(target)
        (part / "data.csv").symlink_to(target)
        with self.assertRaisesRegex(ValueError, "escaping"):
            DERIVE.generate([part], self.output, "smoke")

    def test_source_overwrite_and_output_symlink_escape_are_rejected(self):
        part = make_part(self.root / "part")
        original_hash = DERIVE.sha256(part / "data.csv")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            DERIVE.generate([part], part, "smoke")
        self.assertEqual(DERIVE.sha256(part / "data.csv"), original_hash)
        self.output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        (self.output / "inputs").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "escaping"):
            DERIVE.generate([part], self.output, "smoke")
        self.assertEqual(list(outside.iterdir()), [])

    def test_metadata_explains_scope_and_references(self):
        part = make_part(self.root / "part")
        DERIVE.generate([part], self.output, "smoke")
        metadata = DERIVE.read_json(self.output / "metadata.json")
        self.assertFalse(metadata["gpu_measurements_performed"])
        self.assertIn("nonfinite", metadata["limitations"][0])
        self.assertEqual(metadata["formula"]["tensor_flops_per_fft"], 6144)
        self.assertEqual(metadata["sources"][0]["sha256"]["data.csv"], DERIVE.sha256(part / "data.csv"))

    def test_environment_selection_is_explicit_then_dependency(self):
        with patch.dict("os.environ", {"THUNDERFFT_TABLE4_SOURCE_DIR": "/explicit",
                       "THUNDERFFT_DEPENDENCY_RESULTS": '{"table3":"/dependency"}'}, clear=True):
            self.assertEqual(DERIVE.default_sources(), [Path("/explicit")])
        with patch.dict("os.environ", {"THUNDERFFT_DEPENDENCY_RESULTS": '{"table3":"/dependency"}'}, clear=True):
            self.assertEqual(DERIVE.default_sources(), [Path("/dependency")])
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(DERIVE.default_sources(), [])
        with patch.dict("os.environ", {"THUNDERFFT_DEPENDENCY_RESULTS": "[]"}, clear=True):
            with self.assertRaises(ValueError):
                DERIVE.default_sources()

    def test_cli_pending_exits_78_but_preserves_blank_outputs(self):
        with patch.dict("os.environ", {}, clear=True):
            result = subprocess.run([sys.executable, str(Path(__file__).with_name("derive.py")),
                                     "--output-dir", str(self.output)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 78, result.stderr)
        self.assertTrue((self.output / "table.md").is_file())
        self.assertTrue(all(row["unit_tflops"] == "" for row in rows(self.output / "table.csv")))


class IdealAssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"

    def test_all_three_verified_paper_pairs_complete(self):
        units = [make_part(self.root / f"unit-{gpu}", gpu=gpu, profile="paper", gpu_uuid=UUIDS[gpu])
                 for gpu in DERIVE.GPUS]
        ideals = [make_ideal(self.root / f"ideal-{gpu}", gpu=gpu, profile="paper") for gpu in DERIVE.GPUS]
        state = DERIVE.generate(units, self.output, "paper", ideal_dirs=ideals)
        self.assertEqual((state["status"], state["matched_gpu_pairs"], state["missing"]), ("complete", 3, []))
        self.assertTrue(state["full_paper_comparison"])
        self.assertTrue(state["complete_declared_subset"])
        table = rows(self.output / "table.csv")
        for row in table:
            self.assertEqual(row["efficiency_identity"], "same_gpu_uuid")
            self.assertEqual(row["ideal_selected_candidate"], "t128-c8")
            self.assertAlmostEqual(float(row["efficiency_pct"]), float(row["unit_tflops"]) / 150 * 100)

    def test_ideal_only_retains_raw_trials_and_leaves_unit_blank(self):
        ideal = make_ideal(self.root / "ideal")
        state = DERIVE.generate([], self.output, "smoke", ideal_dirs=[ideal])
        self.assertEqual((state["collected_unit_gpus"], state["collected_ideal_gpus"], state["status"]), (0, 1, "partial"))
        self.assertEqual(len(rows(self.output / "data.csv")), 0)
        self.assertEqual(len(rows(self.output / "ideal.csv")), 8)
        row = rows(self.output / "table.csv")[0]
        self.assertEqual((row["unit_tflops"], row["efficiency_pct"], row["ideal_mma_tflops"]), ("", "", "150.0"))
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("derive.py")),
                                 "--ideal-dir", str(ideal), "--output-dir", str(self.root / "cli")], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_unverified_unit_identity_does_not_create_efficiency(self):
        unit = make_part(self.root / "unit")
        ideal = make_ideal(self.root / "ideal")
        state = DERIVE.generate([unit], self.output, "smoke", ideal_dirs=[ideal])
        row = rows(self.output / "table.csv")[0]
        self.assertEqual((row["efficiency_identity"], row["efficiency_pct"]), ("unverified", ""))
        self.assertEqual(state["matched_gpu_pairs"], 0)
        self.assertTrue(any(item["item"] == "A100/efficiency_identity" for item in state["missing"]))

    def test_wrong_physical_uuid_rejected(self):
        unit = make_part(self.root / "unit", gpu_uuid=UUIDS["H100"])
        ideal = make_ideal(self.root / "ideal")
        with self.assertRaisesRegex(ValueError, "GPU UUID mismatch"):
            DERIVE.generate([unit], self.output, "smoke", ideal_dirs=[ideal])

    def test_historical_inventory_can_verify_unique_selected_gpu(self):
        unit = make_part(self.root / "unit")
        metadata = DERIVE.read_json(unit / "metadata.json")
        metadata.update(gpu_name="NVIDIA A100", gpu_inventory_before={"exit_code": 0,
            "stdout": f"0, {UUIDS['A100']}, NVIDIA A100, 575.1\n1, {UUIDS['H100']}, NVIDIA V100, 575.1\n"})
        DERIVE.write_json(unit / "metadata.json", metadata)
        ideal = make_ideal(self.root / "ideal")
        self.assertEqual(DERIVE.generate([unit], self.output, "smoke", ideal_dirs=[ideal])["matched_gpu_pairs"], 1)

    def test_ambiguous_historical_inventory_remains_unverified(self):
        metadata = {"gpu_name": "NVIDIA A100", "gpu_inventory_before": {"exit_code": 0,
            "stdout": f"0, {UUIDS['A100']}, NVIDIA A100\n1, {UUIDS['H100']}, NVIDIA A100\n"}}
        self.assertIsNone(DERIVE.unit_gpu_uuid(metadata))
        metadata["effective_CUDA_VISIBLE_DEVICES"] = UUIDS["A100"]
        self.assertEqual(DERIVE.unit_gpu_uuid(metadata), UUIDS["A100"])

    def test_candidate_selection_uses_median_throughput_and_lexical_tie(self):
        ideal = make_ideal(self.root / "ideal", throughputs=(150, 150, 145, 130))
        self.assertEqual(DERIVE.load_ideal(ideal, "smoke")["selected_candidate"], "t128-c4")
        values = rows(ideal / "ideal.csv")
        for value in values:
            if value["candidate"] == "t128-c8":
                throughput = 100 if value["trial"] == "0" else 240
                value["tflops"] = throughput
                value["time_us"] = int(value["flop_count"]) / throughput / 1e6
        DERIVE.write_csv(ideal / "ideal.csv", values, DERIVE.IDEAL_FIELDS)
        refresh_ideal_hash(ideal)
        metadata = DERIVE.read_json(ideal / "ideal_metadata.json")
        metadata["selected_candidate"] = "t128-c8"
        DERIVE.write_json(ideal / "ideal_metadata.json", metadata)
        self.assertEqual(DERIVE.load_ideal(ideal, "smoke")["tflops"], 170)

    def test_formula_correctness_and_settings_are_validated(self):
        cases = (("flop_count", "1"), ("mma_instructions", "1"), ("tflops", "999"),
                 ("time_us", "nan"), ("max_abs_error", "0.0001"), ("passed", "false"),
                 ("threads", "64"), ("iterations", "65"), ("unroll", "16"),
                 ("shared_bytes", "-1"), ("registers", "0"), ("blocks", "8"))
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key):
                ideal = make_ideal(self.root / str(index))
                values = rows(ideal / "ideal.csv")
                values[0][key] = value
                DERIVE.write_csv(ideal / "ideal.csv", values, DERIVE.IDEAL_FIELDS)
                refresh_ideal_hash(ideal)
                with self.assertRaises(ValueError):
                    DERIVE.load_ideal(ideal, "smoke")

    def test_full_candidate_trial_coverage_is_required(self):
        ideal = make_ideal(self.root / "ideal")
        values = rows(ideal / "ideal.csv")
        for malformed in (values[:-1], [*values, values[0]]):
            DERIVE.write_csv(ideal / "ideal.csv", malformed, DERIVE.IDEAL_FIELDS)
            refresh_ideal_hash(ideal)
            with self.assertRaisesRegex(ValueError, "coverage|trial"):
                DERIVE.load_ideal(ideal, "smoke")

    def test_bad_hash_status_source_and_selection_are_rejected(self):
        cases = (("status", "running"), ("binary_sha256", "invalid"), ("source_sha256", {}),
                 ("selected_candidate", "t256-c8"), ("gpu_uuid", "invalid"), ("profile", "paper"))
        for index, (key, value) in enumerate(cases):
            with self.subTest(key=key):
                ideal = make_ideal(self.root / str(index))
                metadata = DERIVE.read_json(ideal / "ideal_metadata.json")
                metadata[key] = value
                DERIVE.write_json(ideal / "ideal_metadata.json", metadata)
                with self.assertRaises(ValueError):
                    DERIVE.load_ideal(ideal, "smoke")
        ideal = make_ideal(self.root / "hash")
        (ideal / "disassembly.sass").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            DERIVE.load_ideal(ideal, "smoke")

    def test_mixed_bundles_relocate_with_all_ideal_evidence(self):
        unit = make_part(self.root / "unit", gpu_uuid=UUIDS["A100"])
        ideal = make_ideal(self.root / "ideal")
        DERIVE.generate([unit], self.output, "smoke", ideal_dirs=[ideal])
        result = ASSEMBLE.assemble([unit, ideal, self.output], self.root / "dedup", "smoke")
        self.assertEqual((result["collected_unit_gpus"], result["collected_ideal_gpus"]), (1, 1))
        shutil.rmtree(unit)
        shutil.rmtree(ideal)
        moved = self.root / "moved"
        self.output.rename(moved)
        result = ASSEMBLE.assemble([moved], self.root / "relocated", "smoke")
        self.assertEqual(result["matched_gpu_pairs"], 1)
        self.assertTrue((self.root / "relocated/inputs/ideal-000/build/compile_commands.json").is_file())
        file = moved / "inputs/ideal-000/raw/measurement.log"
        file.write_text(file.read_text() + "changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            ASSEMBLE.assemble([moved], self.root / "bad", "smoke")

    def test_conflicting_ideal_sources_rejected(self):
        first = make_ideal(self.root / "first")
        second = make_ideal(self.root / "second", throughputs=(140, 155, 145, 130))
        with self.assertRaisesRegex(ValueError, "conflicting A100 ideal"):
            DERIVE.generate([], self.output, "smoke", ideal_dirs=[first, second])

    def test_escaping_ideal_evidence_path_is_rejected(self):
        ideal = make_ideal(self.root / "ideal")
        metadata = DERIVE.read_json(ideal / "ideal_metadata.json")
        metadata["outputs"]["../external"] = "c" * 64
        DERIVE.write_json(ideal / "ideal_metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "evidence hashes"):
            DERIVE.load_ideal(ideal, "smoke")

    def test_native_sm_count_checks_grid_and_legacy_bundle_still_imports(self):
        ideal = make_ideal(self.root / "ideal")
        metadata = DERIVE.read_json(ideal / "ideal_metadata.json")
        metadata.pop("multiprocessor_count")
        metadata["sm_count"] = 2
        DERIVE.write_json(ideal / "ideal_metadata.json", metadata)
        with self.assertRaisesRegex(ValueError, "block count"):
            DERIVE.load_ideal(ideal, "smoke")
        unit = make_part(self.root / "unit")
        DERIVE.generate([unit], self.output, "smoke")
        old = DERIVE.read_json(self.output / "metadata.json")
        old.pop("ideal_sources")
        old["outputs"].pop("ideal.csv")
        (self.output / "ideal.csv").unlink()
        DERIVE.write_json(self.output / "metadata.json", old)
        self.assertEqual(ASSEMBLE.assemble([self.output], self.root / "legacy", "smoke")["collected_unit_gpus"], 1)


if __name__ == "__main__":
    unittest.main()
