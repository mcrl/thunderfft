#!/usr/bin/env python3
"""CPU-only synthetic tests for independent libraries and evidence validation."""
from contextlib import redirect_stderr
import csv
import hashlib
import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_measure import module, table6_native, paper_fft, BATCH, REPEATS

HERE = Path(__file__).resolve().parent
A = module("table6_test_assemble", HERE / "assemble.py")
S = module("table6_test_selection", HERE / "selection.py")
UUID = "GPU-11111111-2222-3333-4444-555555555555"
NAME = "NVIDIA A100-SXM4-80GB"


def bundle(root, library, *, uuid=UUID, runtime=12090, driver=13000, binary_digest=None):
    root.mkdir()
    (root / "raw").mkdir()
    is_external = library in ("tcfft", "flashfftconv")
    binary_digest = binary_digest or hashlib.sha256(library.encode()).hexdigest()
    raw = (table6_native(library, "conv_kernel" if library == "flashfftconv" else "layer_256_0_A100")
           if is_external else paper_fft(library, (A.M.CUFFTDX_CANDIDATE,) if library == "cufftdx" else ("",)))
    (root / "raw/data.csv").write_text(raw)
    (root / "raw/data.log").write_text(f"compiled_CUDART=12090 runtime={runtime} driver={driver}\nGPU={NAME} compute=8.0 batch={BATCH}\nGPU_UUID={uuid}\n")
    args = [str(REPEATS), "1", str(BATCH)]
    if library == "cufftdx":
        args = ["--library", "cufftdx", "--candidate", A.M.CUFFTDX_CANDIDATE, "256", "fp16", "register"] + args
    elif library == "thunderfft":
        args = ["256", "fp16", "register"] + args
    meta = {"evaluation": "table6", "profile": "smoke", "status": "completed", "numerical_and_timing_gates_passed": True,
            "settings": {"batch": BATCH, "repeats": REPEATS, "trials": 1}, "selection": {"libraries": [library]},
            "gpu": "A100", "gpu_name": NAME, "gpu_uuid": uuid, "compute_capability": "8.0",
            "source_sha256": {"src.cu": "a" * 64}, "binary_sha256": {library: binary_digest},
            "gpu_inventory_before": {"exit_code": 0, "stdout": f"0, {uuid}, {NAME}, 580\n"},
            "commands": [{"argv": ["/binary"] + args, "exit_code": 0, "library": library, "revision": "fixture",
                          "stdout": "raw/data.csv", "stderr": "raw/data.log"}]}
    if is_external:
        meta["table6_build"] = {"binaries": {library: {"sha256": binary_digest}}, library: {
            "pinned_commit": "1" * 40, "patch": {"sha256": "c" * 64}}}
    (root / "metadata.json").write_text(json.dumps(meta))
    parser = A.M.parse_native if is_external else A.M.parse_paper_fft
    parsed = parser(raw, library=library, batch=BATCH, repeats=REPEATS, trials=1, gpu="A100", revision="fixture")
    A.M.TABLE3.write_csv(root / "data.csv", parsed, A.M.FIELDS)
    return root


class Assembly(unittest.TestCase):
    def test_distinct_executables_retain_their_own_cuda_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / name, name, runtime=12090 if name in ("tcfft", "flashfftconv") else 12080)
                     for name in S.LIBRARIES]
            # Exercise both separately collected libraries and a single native
            # invocation bundle, as produced by the default full paper run.
            combined = root / "combined"
            (combined / "raw").mkdir(parents=True)
            meta = json.loads((parts[0] / "metadata.json").read_text())
            meta.update(commands=[], binary_sha256={}, table6_build={"binaries": {}},
                        selection={"libraries": list(S.LIBRARIES)})
            data = []
            for part in parts:
                current = json.loads((part / "metadata.json").read_text())
                command = current["commands"][0]
                library = command["library"]
                for key in ("stdout", "stderr"):
                    destination = f"raw/{library}{Path(command[key]).suffix}"
                    (combined / destination).write_bytes((part / command[key]).read_bytes())
                    command[key] = destination
                meta["commands"].append(command)
                meta["binary_sha256"].update(current["binary_sha256"])
                if "table6_build" in current:
                    meta["table6_build"][library] = current["table6_build"][library]
                    meta["table6_build"]["binaries"].update(current["table6_build"]["binaries"])
                data.extend(A.E.strict_csv(part / "data.csv", A.M.FIELDS))
            (combined / "metadata.json").write_text(json.dumps(meta))
            A.M.TABLE3.write_csv(combined / "data.csv", data, A.M.FIELDS)
            for name, inputs in (("separate", parts), ("combined", [combined])):
                with self.subTest(collection=name):
                    output = root / (name + "-output")
                    self.assertEqual(A.assemble(inputs, output, "smoke")["status"], "complete")
                    identity = json.loads((output / "metadata.json").read_text())["compatibility_by_gpu"]["A100"]
                    self.assertEqual(identity["native_driver_api"], "13000")
                    self.assertEqual(identity["native_cuda_by_binary"], {
                        library: ["12090", "12090" if library in ("tcfft", "flashfftconv") else "12080", "13000"]
                        for library in S.LIBRARIES})
                    self.assertEqual(identity["native_cuda_by_digest"], {
                        hashlib.sha256(library.encode()).hexdigest(): versions
                        for library, versions in identity["native_cuda_by_binary"].items()})

    def test_reject_runtime_change_for_same_executable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / "first", "tcfft", runtime=12080),
                     bundle(root / "second", "tcfft", runtime=12090, binary_digest="d" * 64)]
            with self.assertRaisesRegex(ValueError, "native_cuda_by_binary.tcfft"):
                A.assemble(parts, root / "out", "smoke")

    def test_reject_runtime_change_for_shared_binary_across_library_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / "thunder", "thunderfft", runtime=12080, binary_digest="d" * 64),
                     bundle(root / "dx", "cufftdx", runtime=12090, binary_digest="d" * 64)]
            with self.assertRaisesRegex(ValueError, "native_cuda_by_digest"):
                A.assemble(parts, root / "out", "smoke")

    def test_shared_binary_with_matching_cuda_can_supply_different_library_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / "thunder", "thunderfft", binary_digest="d" * 64),
                     bundle(root / "dx", "cufftdx", binary_digest="d" * 64)]
            self.assertEqual(A.assemble(parts, root / "out", "smoke")["status"], "partial")
            identity = json.loads((root / "out/metadata.json").read_text())["compatibility_by_gpu"]["A100"]
            self.assertEqual(identity["native_cuda_by_digest"], {"d" * 64: ["12090", "12090", "13000"]})

    def test_reject_native_driver_change_between_executables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / "tc", "tcfft"), bundle(root / "flash", "flashfftconv", driver=12090)]
            with self.assertRaisesRegex(ValueError, "native_driver_api"):
                A.assemble(parts, root / "out", "smoke")

    def test_combine_four_libraries_and_calculate_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parts = [bundle(root / name, name) for name in S.LIBRARIES]
            result = A.assemble(parts, root / "out", "smoke")
            self.assertEqual(result["status"], "complete")
            rows = list(csv.DictReader(io.StringIO((root / "out/table.csv").read_text())))
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(row["speedup_vs_cufftdx"] and row["status"] == "collected" for row in rows))
            self.assertFalse(result["full_paper_comparison"])

    def test_missing_cells_stay_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = A.assemble([bundle(root / "tc", "tcfft")], root / "out", "smoke")
            rows = list(csv.DictReader(io.StringIO((root / "out/table.csv").read_text())))
            self.assertEqual((result["status"], len(result["missing"])), ("partial", 3))
            self.assertTrue(all(row["speedup_vs_cufftdx"] == "" for row in rows))
            self.assertTrue(all(row["median_time_ms"] == "" for row in rows if row["status"] == "pending"))

    def test_output_cannot_overlap_source_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = bundle(root / "part", "tcfft")
            original = {path.relative_to(part).as_posix(): path.read_bytes() for path in part.rglob("*") if path.is_file()}
            alias = root / "alias"
            alias.symlink_to(part, target_is_directory=True)
            for output in (part, root, part / "nested-output", alias):
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "output must be separate"):
                    A.assemble([part], output, "smoke")
            self.assertFalse((part / "nested-output").exists())
            self.assertEqual(original, {path.relative_to(part).as_posix(): path.read_bytes()
                                        for path in part.rglob("*") if path.is_file()})

    def test_reject_modified_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            part = bundle(Path(directory) / "tc", "tcfft")
            (part / "data.csv").write_text((part / "data.csv").read_text().replace("0.001,", "0.002,", 1))
            with self.assertRaisesRegex(ValueError, "disagrees"):
                A.load_part(part, "smoke")

    def test_reject_duplicate_or_incompatible_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = bundle(root / "tc1", "tcfft")
            duplicate = bundle(root / "tc2", "tcfft")
            other = bundle(root / "flash", "flashfftconv", uuid=UUID.replace("11111111", "99999999"))
            for parts, message in (([first, duplicate], "overlapping"), ([first, other], "incompatible")):
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    A.assemble(parts, root / message, "smoke")

    def test_reject_missing_raw_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            part = bundle(Path(directory) / "tc", "tcfft")
            (part / "raw/data.csv").unlink()
            with self.assertRaisesRegex(ValueError, "evidence"):
                A.load_part(part, "smoke")

    def test_paper_repeats_rejected_before_native_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["measure.py", "--libraries", "thunderfft", "--thunderfft-binary", str(root / "not-built"),
                    "--build-dir", str(root), "--output", str(root / "out"), "--profile", "paper"]
            errors = io.StringIO()
            with patch.dict("os.environ", {"THUNDERFFT_TABLE6_REPEATS": "64"}, clear=True), \
                    patch("sys.argv", argv), patch.object(A.M.subprocess, "run") as native, \
                    redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                A.M.main()
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("R=1000", errors.getvalue())
            native.assert_not_called()

    def test_selection(self):
        self.assertEqual(S.libraries("tcfft"), ("tcfft",))
        self.assertEqual(S.libraries("all"), S.LIBRARIES)
        for value in ("", "tcfft,tcfft", "unknown", "all,tcfft"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                S.libraries(value)


if __name__ == "__main__":
    unittest.main()
