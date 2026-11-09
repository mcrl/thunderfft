#!/usr/bin/env python3
"""Contract checks: reject incomplete, corrupted and numerically invalid runs."""

import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import measure
from measure import CUFFTDX_RAW_FIELDS, RAW_FIELDS, gpu_name, native_identity, parse_raw, summarize


class NativeOutputContract(unittest.TestCase):
    def raw(self, **changes):
        row = dict(zip(RAW_FIELDS, (64, "fp16", "register", 1024, 256, 0,
                                   100, 356, 1, 4, 0.0002, 1024)))
        row.update(changes)
        output = io.StringIO()
        writer = csv.DictWriter(output, RAW_FIELDS)
        writer.writeheader()
        writer.writerow(row)
        return output.getvalue()

    def parse(self, text, **options):
        args = dict(n=64, precision="fp16", placement="register", batch=1024,
                    repeats=256, trials=1, gpu="A100", raw_csv="raw/example.csv")
        args.update(options)
        return parse_raw(text, **args)

    def test_valid_conversion_and_smoke_does_not_compare_paper(self):
        rows = self.parse(self.raw())
        self.assertEqual(rows[0]["time_us"], 1)
        self.assertEqual(rows[0]["passed"], "true")
        table, candidates, speedups = summarize(rows, "smoke")
        self.assertEqual(table[0]["paper_target_us"], "")
        self.assertEqual((candidates, speedups), ([], []))

    def test_failure_gates(self):
        for changes in ({"single_fft_relL2": "nan"}, {"single_fft_relL2": 0.006},
                        {"compute_us": 0}, {"compute_us": "inf"}, {"t_2R_us": 99},
                        {"compute_us": 2}, {"e2e_out_of_place_us": -1},
                        {"N": 128}, {"batch": 65536}, {"inside_repeats": 255},
                        {"trial": 1}, {"shared_bytes": 0}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.parse(self.raw(**changes))

    def test_incomplete_duplicate_and_malformed_csv(self):
        valid = self.raw()
        for text, options in ((valid, {"trials": 2}),
                              (valid + valid.splitlines()[1] + "\n", {}),
                              (valid + "64,fp16\n", {}), (valid.replace("compute_us", "time"), {})):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.parse(text, **options)

    def test_device_identity_never_defaults_unknown_gpu_to_a100(self):
        self.assertEqual(gpu_name("GPU=NVIDIA A100-PCIE-80GB compute=8.0 batch=1024 R=256\n")[0], "A100")
        self.assertEqual(gpu_name("GPU=Tesla V100 compute=7.0 batch=1024 R=256\n")[0], "Tesla V100")
        with self.assertRaises(ValueError):
            gpu_name("missing metadata")


class CufftdxContract(unittest.TestCase):
    """cuFFTDx output: every candidate gated, fastest median selected, matched speedups."""

    def raw(self, rows, fields=CUFFTDX_RAW_FIELDS):
        output = io.StringIO()
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        return output.getvalue()

    def row(self, candidate="block_f2_e16", trial=0, compute=2.0, precision="fp16", shared=8192, n=64):
        t_r = 100.0
        return dict(zip(CUFFTDX_RAW_FIELDS, (n, precision, "register", 1024, 256, trial, t_r,
                                             t_r + 256 * compute, compute, 4, 0.002, shared,
                                             "cufftdx", candidate, 2, 16, 64)))

    def parse(self, text, **options):
        args = dict(n=64, precision="fp16", placement="register", batch=1024, repeats=256,
                    trials=2, gpu="A100", raw_csv="raw/dx.csv", library="cuFFTDx")
        args.update(options)
        return parse_raw(text, **args)

    def thunder(self, compute, precision="fp16", n=64):
        row = dict(zip(RAW_FIELDS, (n, precision, "register", 1024, 256, 0, 100, 100 + 256 * compute,
                                    compute, 4, 0.0002, 1024)))
        return parse_raw(NativeOutputContract.raw(NativeOutputContract(), **row), n=n, precision=precision,
                         placement="register", batch=1024, repeats=256, trials=1, gpu="A100",
                         raw_csv="raw/t.csv")

    def test_candidates_selection_and_speedup(self):
        rows = self.parse(self.raw([self.row("block_default", 0, 3.0), self.row("block_default", 1, 3.0),
                                    self.row("thread_t64", 0, 1.0, shared=0), self.row("thread_t64", 1, 1.5, shared=0)]))
        self.assertEqual({row["library"] for row in rows}, {"cuFFTDx"})
        table, candidates, speedups = summarize(self.thunder(0.5) + rows, "smoke")
        self.assertEqual(len(candidates), 2)
        winner = [row for row in table if row["library"] == "cuFFTDx"]
        self.assertEqual((len(winner), winner[0]["candidate"], winner[0]["cufftdx_candidates"]),
                         (1, "thread_t64", 2))
        self.assertEqual([c["selected"] for c in candidates if c["candidate"] == "thread_t64"], ["true"])
        self.assertAlmostEqual(speedups[0]["speedup"], 1.25 / 0.5)
        self.assertEqual(speedups[-1]["N"], "geomean(1 sizes)")
        self.assertEqual(speedups[0]["paper_speedup"], "")

    def test_tf32_pairs_with_cufftdx_fp32(self):
        dx = self.parse(self.raw([self.row(precision="fp32", trial=t, compute=4.0) for t in (0, 1)]),
                        precision="fp32")
        _, _, speedups = summarize(self.thunder(1.0, "tf32") + self.thunder(1.0, "fp16") + dx, "smoke")
        self.assertEqual([(s["thunderfft_precision"], s["cufftdx_precision"], s["speedup"])
                          for s in speedups if not str(s["N"]).startswith("geomean")], [("tf32", "fp32", 4.0)])

    def test_cufftdx_failure_gates(self):
        good = [self.row(trial=0), self.row(trial=1)]
        cases = ([self.row(trial=0)],                                   # incomplete candidate
                 [self.row(trial=0), self.row(trial=0)],                # duplicate trial
                 [self.row(trial=0, shared=0), self.row(trial=1, shared=0)],  # block without shared
                 [dict(r, candidate="best") for r in good],             # unknown label
                 [dict(r, library="thunderfft") for r in good],
                 [dict(r, single_fft_relL2=0.01) for r in good],
                 [good[0], dict(good[1], threads_per_block=128)],
                 [dict(r, threads_per_block=0) for r in good], [])
        for rows in cases:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.parse(self.raw(rows))
        with self.assertRaises(ValueError):  # ThunderFFT header is not a cuFFTDx contract
            self.parse(NativeOutputContract.raw(NativeOutputContract()))

    def test_native_cufftdx_identity(self):
        self.assertEqual(native_identity("GPU=x\nlibrary=cufftdx cuFFTDx=10501 target_SM=800\n"),
                         {"cufftdx_version": 10501, "cufftdx_target_sm": 800})
        self.assertEqual(native_identity("GPU=x\n"), {})


class LibrarySelectionTests(unittest.TestCase):
    def test_defaults_explicit_selection_and_missing_dependency(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(measure.selected_libraries(False), ("ThunderFFT",))
            self.assertEqual(measure.selected_libraries(True), ("ThunderFFT", "cuFFTDx"))
        with patch.dict("os.environ", {"THUNDERFFT_TABLE3_LIBRARIES": "cuFFTDx"}):
            self.assertEqual(measure.selected_libraries(True), ("cuFFTDx",))
            with self.assertRaises(ValueError):
                measure.selected_libraries(False)
        for value in ("", "ThunderFFT,ThunderFFT", "cuFFT", "ThunderFFT,"):
            with patch.dict("os.environ", {"THUNDERFFT_TABLE3_LIBRARIES": value}), self.assertRaises(ValueError):
                measure.selected_libraries(True)

    def test_cufftdx_only_run_has_no_thunder_binary_or_speedup_requirement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binary = root / "dx"
            binary.write_bytes(b"synthetic executable; subprocess mocked")
            output = root / "output"
            argv = ["measure.py", "--binary", str(root / "not-built"), "--cufftdx-binary", str(binary),
                    "--output", str(output), "--build-dir", str(root), "--profile", "smoke"]
            calls = []
            contract = CufftdxContract()
            def execute(command, **kwargs):
                calls.append(command)
                kwargs["stdout"].write(contract.raw([contract.row(trial=trial) for trial in (0, 1)]))
                kwargs["stderr"].write("GPU=NVIDIA A100 compute=8.0 batch=1024 R=256 trials=2\n"
                                       "library=cufftdx cuFFTDx=10501 target_SM=800\n")
                return SimpleNamespace(returncode=0)
            env = {"THUNDERFFT_TABLE3_LIBRARIES": "cuFFTDx", "THUNDERFFT_TABLE3_N_LIST": "64",
                   "THUNDERFFT_TABLE3_PRECISIONS": "fp16", "THUNDERFFT_TABLE3_PLACEMENTS": "register",
                   "THUNDERFFT_TABLE3_CUFFTDX_MODE": "all"}
            with patch.dict("os.environ", env, clear=True), patch("sys.argv", argv), \
                    patch.object(measure, "capture", return_value={}), \
                    patch.object(measure, "source_hashes", return_value={"source.cu": "a" * 64}), \
                    patch.object(measure.subprocess, "run", side_effect=execute):
                self.assertEqual(measure.main(), 0)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][1:3], ["--library", "cufftdx"])
            metadata = json.loads((output / "metadata.json").read_text())
            self.assertIsNone(metadata["binary_sha256"])
            self.assertEqual(metadata["coverage"]["selected_thunderfft_configurations"], 0)
            self.assertEqual((output / "speedup.csv").read_text().strip(), ",".join(measure.SPEEDUP_FIELDS))


if __name__ == "__main__":
    unittest.main()
