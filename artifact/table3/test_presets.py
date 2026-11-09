#!/usr/bin/env python3
"""Preset selection must preserve native identity, geometry and trial evidence."""

import copy
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import measure


UUID = "GPU-11111111-2222-3333-4444-555555555555"


def metadata(gpu="A100"):
    descriptor = measure.preset_descriptor(gpu)
    manifest, _ = measure.PRESETS.load_manifest()
    record = manifest["gpus"][gpu]
    return {"gpu": gpu, "gpu_name": record["gpu_name"], "gpu_uuid": UUID,
            "compute_capability": record["compute_capability"],
            "cufftdx": {"mode": "preset", "preset": descriptor}, "cufftdx_build": descriptor["build"],
            "source_sha256": {**descriptor["provenance"]["candidate_source_sha256"],
                              descriptor["manifest_path"]: descriptor["manifest_sha256"]},
            "gpu_inventory_before": {"exit_code": 0, "stdout":
                f"0, {UUID}, {record['gpu_name']}, 575.51.03, {record['measurement_environment']['memory_MiB']} MiB"}}


def parsed_rows(gpu="A100", n=64, precision="fp16", placement="register"):
    cell = measure.PRESETS.cell(gpu, n, precision, placement)
    return [{"gpu": gpu, "N": str(n), "precision": precision, "placement": placement,
             "library": "cuFFTDx", "candidate": cell["candidate"],
             **{k: str(v) for k, v in cell["geometry"].items()}, "trial": str(trial)} for trial in range(3)]


class PresetContractTests(unittest.TestCase):
    def test_complete_manifest_and_fixed_gpu_specific_candidates(self):
        manifest, digest = measure.PRESETS.load_manifest()
        self.assertEqual(sum(len(gpu["cells"]) for gpu in manifest["gpus"].values()), 84)
        self.assertEqual(measure.PRESETS.cell("A100", 64, "fp16", "register")["candidate"], "thread_t32")
        self.assertEqual(measure.PRESETS.cell("B200", 64, "fp16", "register")["candidate"], "thread_t256")
        for gpu in manifest["gpus"]:
            meta = metadata(gpu)
            self.assertEqual(measure.cufftdx_contract(meta), {"mode": "preset", "preset_sha256": digest})
            for cell in manifest["gpus"][gpu]["cells"]:
                n, precision, placement = cell["N"], cell["precision"], cell["placement"]
                self.assertEqual(measure.validate_preset_rows(parsed_rows(gpu, n, precision, placement),
                                 meta, n, precision, placement), cell["candidate"])

    def test_historical_all_mode_and_invalid_modes(self):
        self.assertEqual(measure.cufftdx_contract({}), {"mode": "all"})
        self.assertEqual(measure.cufftdx_contract({"cufftdx": {"selection": "historical sweep"}}), {"mode": "all"})
        self.assertIsNone(measure.validate_preset_rows([], {"cufftdx": {"mode": "all"}}, 64, "fp16", "register"))
        for info in ({"mode": "typo"}, {"mode": "all", "preset": {}}, {"mode": "preset"}):
            with self.subTest(info=info), self.assertRaises(ValueError):
                measure.cufftdx_contract({"cufftdx": info})

    def test_reject_wrong_label_extra_candidate_and_geometry(self):
        good = parsed_rows()
        mutations = [[], [dict(row, candidate="thread_t64") for row in good],
                     good + [dict(good[0], candidate="block_default")],
                     [dict(row, ffts_per_block="128") for row in good],
                     [dict(row, shared_bytes="16") for row in good],
                     [dict(row, N="128") for row in good]]
        for rows in mutations:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                measure.validate_preset_rows(rows, metadata(), 64, "fp16", "register")

    def test_reject_changed_provenance_build_source_and_gpu(self):
        changes = [lambda m: m["cufftdx"]["preset"].update(manifest_sha256="0" * 64),
                   lambda m: m["cufftdx"]["preset"]["provenance"].update(source_commit="0" * 40),
                   lambda m: m.update(cufftdx_build={"cufftdx_version": 10502, "cufftdx_target_sm": 800}),
                   lambda m: m.update(cufftdx_build={"cufftdx_version": 10501, "cufftdx_target_sm": 900}),
                   lambda m: m["source_sha256"].update({"sample/bench/baseline/cufftdx_bench.h": "0" * 64}),
                   lambda m: m["source_sha256"].pop("artifact/table3/cufftdx_presets.json"),
                   lambda m: m.update(gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"),
                   lambda m: m.update(compute_capability="9.0"),
                   lambda m: m["gpu_inventory_before"].update(stdout=f"0, {UUID}, NVIDIA A100-SXM4-80GB, 575.51.03, 81920 MiB")]
        for change in changes:
            meta = metadata(); change(meta)
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                measure.cufftdx_contract(meta)

    def test_visibility_mask_resolves_selected_gpu_and_rejects_unsupported_skus(self):
        inventory = metadata()["gpu_inventory_before"]
        self.assertEqual(measure.PRESETS.select_device(inventory, UUID)["gpu"], "A100")
        self.assertEqual(measure.PRESETS.select_device(inventory, "0")["gpu_uuid"], UUID)
        for visibility in ("", "-1", "1", "GPU-missing"):
            with self.subTest(visibility=visibility), self.assertRaises(ValueError):
                measure.PRESETS.select_device(inventory, visibility)
        multiple = dict(inventory, stdout=inventory["stdout"] + "\n1, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, NVIDIA B200, 575.51.03, 183359 MiB")
        with self.assertRaisesRegex(ValueError, "CUDA ordinals"):
            measure.PRESETS.select_device(multiple, "0")
        self.assertEqual(measure.PRESETS.select_device(multiple, UUID)["gpu"], "A100")
        for name, memory in (("NVIDIA H100 NVL", 95830), ("NVIDIA A100 40GB PCIe", 40960), ("NVIDIA H200", 143771)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                measure.PRESETS.family_for_device(name, memory)

    def test_preceding_native_identity_overrides_ambiguous_cuda_ordinal(self):
        meta = metadata("B200")
        meta["gpu_inventory_before"]["stdout"] = (
            "0, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, NVIDIA A100 80GB PCIe, 575.51.03, 81920 MiB\n"
            + meta["gpu_inventory_before"]["stdout"].replace("0, ", "1, ", 1))
        meta["effective_CUDA_VISIBLE_DEVICES"] = "0"
        meta["build_configuration"] = {"compile_commands": [{"command": "nvcc -DTHFF_CUFFTDX_SM=1000"}]}
        self.assertEqual(measure.preset_preflight(meta, native_device=True)[0], "B200")
        self.assertEqual(meta["cufftdx"]["preset"]["gpu"], "B200")


class PresetExecutionTests(unittest.TestCase):
    def run_measure(self, *, mode=None, env_mode=None, gpu="A100", native_change=None, build_sm=None):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); binary = root / "dx"; binary.write_bytes(b"mock native executable")
            output = root / "result"
            argv = ["measure.py", "--binary", str(root / "unused"), "--cufftdx-binary", str(binary),
                    "--build-dir", str(root), "--output", str(output), "--profile", "smoke"]
            if mode is not None:
                argv += ["--cufftdx-mode", mode]
            meta = metadata(gpu); calls = []
            env = {"THUNDERFFT_TABLE3_LIBRARIES": "cuFFTDx", "THUNDERFFT_TABLE3_N_LIST": "64",
                   "THUNDERFFT_TABLE3_PRECISIONS": "fp16", "THUNDERFFT_TABLE3_PLACEMENTS": "register",
                   "CUDA_VISIBLE_DEVICES": UUID}
            if env_mode is not None:
                env["THUNDERFFT_TABLE3_CUFFTDX_MODE"] = env_mode
            def execute(command, **kwargs):
                calls.append(command)
                cell = copy.deepcopy(measure.PRESETS.cell(gpu, 64, "fp16", "register"))
                if native_change:
                    native_change(cell)
                stream = io.StringIO(); writer = csv.DictWriter(stream, measure.CUFFTDX_RAW_FIELDS); writer.writeheader()
                for trial in (0, 1):
                    writer.writerow({"N": 64, "precision": "fp16", "placement": "register", "batch": 1024,
                                     "inside_repeats": 256, "trial": trial, "t_R_us": 100, "t_2R_us": 356,
                                     "compute_us": 1, "e2e_out_of_place_us": 4, "single_fft_relL2": .002,
                                     "library": "cufftdx", "candidate": cell["candidate"], **cell["geometry"]})
                kwargs["stdout"].write(stream.getvalue())
                kwargs["stderr"].write(f"GPU={meta['gpu_name']} compute={meta['compute_capability']} batch=1024 R=256 trials=2\n"
                                       f"GPU_UUID={UUID}\nlibrary=cufftdx cuFFTDx=10501 target_SM={meta['cufftdx_build']['cufftdx_target_sm']}\n")
                return SimpleNamespace(returncode=0)
            def capture(command):
                return meta["gpu_inventory_before"] if command[0] == "nvidia-smi" else {}
            build = {"compile_commands": [{"command": f"nvcc -DTHFF_CUFFTDX_SM={build_sm or meta['cufftdx_build']['cufftdx_target_sm']}"}]}
            with patch.dict("os.environ", env, clear=True), patch("sys.argv", argv), \
                    patch.object(measure, "capture", side_effect=capture), \
                    patch.object(measure, "build_evidence", return_value=build), \
                    patch.object(measure, "source_hashes", return_value=meta["source_sha256"]), \
                    patch.object(measure.subprocess, "run", side_effect=execute):
                result = measure.main()
            return result, calls, json.loads((output / "metadata.json").read_text())

    def test_default_selects_exact_gpu_specific_candidate_and_records_evidence(self):
        for gpu, expected in (("A100", "thread_t32"), ("H100", "thread_t32"), ("B200", "thread_t256")):
            result, calls, meta = self.run_measure(gpu=gpu)
            self.assertEqual(result, 0)
            self.assertEqual(calls[0][1:5], ["--library", "cufftdx", "--candidate", expected])
            self.assertEqual(meta["cufftdx"]["mode"], "preset")
            self.assertEqual(meta["coverage"]["cufftdx_candidates"], 1)
            self.assertEqual(meta["commands"][0]["cufftdx_preset"]["candidate"], expected)

    def test_cli_all_overrides_environment_and_preserves_sweep_command(self):
        result, calls, meta = self.run_measure(mode="all", env_mode="preset")
        self.assertEqual(result, 0)
        self.assertNotIn("--candidate", calls[0])
        self.assertEqual(meta["cufftdx"]["mode"], "all")
        self.assertNotIn("preset", meta["cufftdx"])

    def test_environment_selects_all_and_cli_preset_overrides_it(self):
        self.assertEqual(self.run_measure(env_mode="all")[2]["cufftdx"]["mode"], "all")
        self.assertEqual(self.run_measure(mode="preset", env_mode="all")[2]["cufftdx"]["mode"], "preset")

    def test_wrong_build_fails_before_native_launch(self):
        result, calls, meta = self.run_measure(build_sm=900)
        self.assertEqual((result, calls, meta["status"]), (1, [], "failed"))

    def test_native_candidate_and_geometry_mismatch_fail_without_fallback(self):
        for change in (lambda c: c.update(candidate="thread_t64"), lambda c: c["geometry"].update(ffts_per_block=128)):
            result, calls, meta = self.run_measure(native_change=change)
            self.assertEqual((result, len(calls), meta["status"]), (1, 1, "failed"))


if __name__ == "__main__":
    unittest.main()
