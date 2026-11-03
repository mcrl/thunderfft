"""Synthetic CPU checks for split collection, missing cells and evidence validation."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("table3_assembler_tests", Path(__file__).with_name("assemble.py"))
ASSEMBLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSEMBLE)
MEASURE = ASSEMBLE.MEASURE


def make_part(path, configs=((64, "fp16", "register"),), gpu="A100", status="completed", **overrides):
    path.mkdir(parents=True)
    (path / "raw").mkdir()
    metadata = {"evaluation": "table3", "profile": "smoke", "status": status,
                "settings": {"batch": 1024, "repeats": 256, "trials": 2}, "gpu": gpu,
                "gpu_name": f"NVIDIA {gpu}", "compute_capability": "8.0", "effective_CUDA_VISIBLE_DEVICES": "0",
                "source_sha256": {"benchmark.cu": "a" * 64}, "binary_sha256": "b" * 64,
                "build_configuration": {"flags": "-O3"},
                "gpu_inventory_before": {"exit_code": 0, "stdout": f"0, GPU-{gpu}, NVIDIA {gpu}, 575.1\n"},
                "commands": []}
    rows = []
    for n, precision, placement in configs:
        stem = f"n{n}-{precision}-{placement}"
        native = [{"N": n, "precision": precision, "placement": placement, "batch": 1024,
                   "inside_repeats": 256, "trial": trial, "t_R_us": 100, "t_2R_us": 356 + 256 * trial,
                   "compute_us": 1 + trial, "e2e_out_of_place_us": 4, "single_fft_relL2": 0.0002,
                   "shared_bytes": 1024} for trial in range(2)]
        csv_relative, log_relative = f"raw/{stem}.csv", f"raw/{stem}.log"
        MEASURE.write_csv(path / csv_relative, native, MEASURE.RAW_FIELDS)
        (path / log_relative).write_text(f"compiled_CUDART=12090 runtime=12080 driver=12090 cuFFT=11303 extra_R_warmups=3\n"
                                       f"GPU=NVIDIA {gpu} compute=8.0 batch=1024 R=256 trials=2\n")
        metadata["commands"].append({"stdout": csv_relative, "stderr": log_relative, "exit_code": 0,
                                     "argv": ["/synthetic/benchmark", str(n), precision, placement, "256", "2", "1024"]})
        rows += MEASURE.parse_raw((path / csv_relative).read_text(), n=n, precision=precision, placement=placement,
                                  batch=1024, repeats=256, trials=2, gpu=gpu, raw_csv=csv_relative)
    metadata.update(overrides)
    MEASURE.atomic_json(path / "metadata.json", metadata)
    MEASURE.write_csv(path / "data.csv", rows, MEASURE.FIELDS)
    return path


class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_split_parts_fill_cells_and_preserve_blank_missing(self):
        a = make_part(self.root / "first", status="failed")
        b = make_part(self.root / "second", ((256, "tf32", "shared"),))
        result = ASSEMBLE.assemble([a, b], self.root / "assembled", "smoke")
        self.assertEqual((result["status"], result["collected_configurations"], len(result["missing"])), ("partial", 2, 83))
        with (self.root / "assembled/table.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 84)
        self.assertEqual(sum(row["status"] == "collected" for row in rows), 2)
        self.assertTrue(all(row["median_time_us"] == "" for row in rows if row["status"] == "pending"))
        self.assertFalse(result["full_paper_comparison"])

    def test_distinct_gpu_identities_are_kept_in_separate_rows(self):
        a = make_part(self.root / "a100")
        h = make_part(self.root / "h100", gpu="H100", binary_sha256="c" * 64)
        result = ASSEMBLE.assemble([a, h], self.root / "output", "smoke")
        self.assertEqual(result["collected_configurations"], 2)

    def test_different_gpu_families_must_use_the_same_source(self):
        a = make_part(self.root / "a100")
        h = make_part(self.root / "h100", gpu="H100", source_sha256={"benchmark.cu": "d" * 64})
        with self.assertRaisesRegex(ValueError, "source_sha256 across GPU"):
            ASSEMBLE.assemble([a, h], self.root / "output", "smoke")

    def test_native_uuid_is_authoritative_when_cuda_and_inventory_ordinals_differ(self):
        first = "GPU-11111111-1111-1111-1111-111111111111"
        selected = "GPU-22222222-2222-2222-2222-222222222222"
        a = make_part(self.root / "first", gpu_inventory_before={"exit_code": 0, "stdout":
            f"0, {first}, NVIDIA A100, 575.1\n1, {selected}, NVIDIA A100, 575.1\n"})
        log = a / "raw/n64-fp16-register.log"
        log.write_text(log.read_text() + f"GPU_UUID={selected}\n")
        _, identity, _ = ASSEMBLE.load_part(a, "smoke")
        self.assertEqual(identity["gpu_uuid"], selected)

    def test_legacy_same_model_devices_require_a_uuid_mask(self):
        first = "GPU-11111111-1111-1111-1111-111111111111"
        second = "GPU-22222222-2222-2222-2222-222222222222"
        a = make_part(self.root / "first", gpu_inventory_before={"exit_code": 0, "stdout":
            f"0, {first}, NVIDIA A100, 575.1\n1, {second}, NVIDIA A100, 575.1\n"})
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            ASSEMBLE.load_part(a, "smoke")
        metadata = json.loads((a / "metadata.json").read_text())
        metadata["effective_CUDA_VISIBLE_DEVICES"] = second
        MEASURE.atomic_json(a / "metadata.json", metadata)
        _, identity, _ = ASSEMBLE.load_part(a, "smoke")
        self.assertEqual(identity["gpu_uuid"], second)

    def test_incompatible_same_gpu_parts_and_overlaps_fail(self):
        a = make_part(self.root / "first")
        b = make_part(self.root / "second", ((128, "tf32", "register"),), binary_sha256="c" * 64)
        with self.assertRaisesRegex(ValueError, "incompatible"):
            ASSEMBLE.assemble([a, b], self.root / "output", "smoke")
        duplicate = make_part(self.root / "duplicate")
        with self.assertRaisesRegex(ValueError, "overlapping"):
            ASSEMBLE.assemble([a, duplicate], self.root / "output", "smoke")

    def test_corrupted_checkpoint_and_native_correctness_fail(self):
        a = make_part(self.root / "first")
        text = (a / "data.csv").read_text()
        (a / "data.csv").write_text(text.replace(",1.0,4.0,", ",9.0,4.0,"))
        with self.assertRaisesRegex(ValueError, "disagrees"):
            ASSEMBLE.assemble([a], self.root / "output", "smoke")
        (a / "data.csv").write_text(text)
        native = a / "raw/n64-fp16-register.csv"
        native.write_text(native.read_text().replace("0.0002", "0.28"))
        with self.assertRaisesRegex(ValueError, "L2 gate"):
            ASSEMBLE.assemble([a], self.root / "output", "smoke")

    def test_incomplete_checkpoint_trials_fail(self):
        a = make_part(self.root / "first")
        data = (a / "data.csv").read_text().splitlines()
        (a / "data.csv").write_text("\n".join(data[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "checkpoint trials"):
            ASSEMBLE.assemble([a], self.root / "output", "smoke")

    def test_pending_assembly_and_profile_mismatch(self):
        result = ASSEMBLE.assemble([], self.root / "pending", "smoke")
        self.assertEqual((result["status"], len(result["missing"])), ("pending", 85))
        a = make_part(self.root / "first")
        with self.assertRaisesRegex(ValueError, "profile mismatch"):
            ASSEMBLE.assemble([a], self.root / "output", "paper")

    def test_explicit_selection_is_validated(self):
        with patch.dict("os.environ", {"THUNDERFFT_TABLE3_N_LIST": "64,256"}):
            self.assertEqual(MEASURE.selection_env("THUNDERFFT_TABLE3_N_LIST", MEASURE.SIZES, int), (64, 256))
        for value in ("", "64,", "64,64", "32"):
            with self.subTest(value=value), patch.dict("os.environ", {"THUNDERFFT_TABLE3_N_LIST": value}):
                with self.assertRaises(ValueError):
                    MEASURE.selection_env("THUNDERFFT_TABLE3_N_LIST", MEASURE.SIZES, int)


if __name__ == "__main__":
    unittest.main()
