"""CPU checks for ideal MMA evidence parsing, selection, and run limits."""

import contextlib
import csv
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("table4_measure_tests", Path(__file__).with_name("measure.py"))
MEASURE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEASURE)

UUID = "GPU-f4bb2c48-b43c-0f5a-8f59-75d6373ea4e6"
DEVICE = {"gpu": "A100", "gpu_uuid": UUID, "sm_count": 108}
SETTINGS = {"iterations": 64, "trials": 3, "warmups": 2, "waves": 2}


def native_rows(candidate="t128-c4", settings=None):
    settings = settings or SETTINGS
    threads, chains = MEASURE.CANDIDATES[candidate]
    blocks = DEVICE["sm_count"] * 4 * settings["waves"]
    mma = blocks * (threads // 32) * chains * settings["iterations"] * 32
    return [dict(trial=trial, blocks=blocks, threads=threads, chains=chains,
                 iterations=settings["iterations"], unroll=32, mma_instructions=mma,
                 flop_count=mma * 2048, time_us=400 + 25 * trial,
                 tflops=mma * 2048 / (400 + 25 * trial) / 1e6,
                 max_abs_error=0, passed="true", registers=32, shared_bytes=0,
                 active_blocks_per_sm=4)
            for trial in range(settings["trials"])]


def csv_text(rows, fields=None):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields or MEASURE.RAW_FIELDS)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def identity_log(name="NVIDIA A100 80GB PCIe", uuid=UUID):
    return (f"GPU={name} compute=8.0 sm_count=108\nGPU_UUID={uuid}\n"
            "compiled_CUDART=12090 runtime=12090 driver=13000\n")


class NativeCsvTests(unittest.TestCase):
    def parse(self, rows=None, candidate="t128-c4", settings=None):
        return MEASURE.parse_raw(csv_text(rows if rows is not None else native_rows(candidate)),
                                 candidate, settings or SETTINGS, DEVICE)

    def test_all_launch_configurations_have_consistent_counts(self):
        for candidate in MEASURE.CANDIDATES:
            with self.subTest(candidate=candidate):
                rows = self.parse(candidate=candidate)
                self.assertEqual(len(rows), SETTINGS["trials"])
                row = rows[0]
                expected = row["blocks"] * row["threads"] // 32 * row["chains"] * 64 * 32
                self.assertEqual(row["mma_instructions"], expected)
                self.assertEqual(row["flop_count"], expected * 2048)
                self.assertAlmostEqual(row["tflops"], expected * 2048 / row["time_us"] / 1e6)
                self.assertEqual((row["gpu"], row["gpu_uuid"], row["candidate"]),
                                 ("A100", UUID, candidate))
                self.assertEqual(row["passed"], "true")

    def test_thread_count_is_divided_by_warp_size(self):
        rows = native_rows()
        rows[0]["mma_instructions"] *= 32
        rows[0]["flop_count"] *= 32
        rows[0]["tflops"] *= 32
        with self.assertRaisesRegex(ValueError, "operation count"):
            self.parse(rows)

    def test_mma_and_flop_counts_are_both_checked(self):
        for field in ("mma_instructions", "flop_count"):
            with self.subTest(field=field):
                rows = native_rows()
                rows[0][field] += 1
                with self.assertRaisesRegex(ValueError, "operation count"):
                    self.parse(rows)

    def test_grid_must_match_sm_count_occupancy_and_waves(self):
        for field in ("blocks", "active_blocks_per_sm"):
            with self.subTest(field=field):
                rows = native_rows()
                rows[0][field] += 1
                with self.assertRaisesRegex(ValueError, "grid"):
                    self.parse(rows)

    def test_candidate_and_loop_settings_must_match_request(self):
        for field in ("threads", "chains", "iterations", "unroll"):
            with self.subTest(field=field):
                rows = native_rows()
                rows[0][field] *= 2
                with self.assertRaisesRegex(ValueError, "settings mismatch"):
                    self.parse(rows)

    def test_rate_is_recomputed_from_counts_and_time(self):
        rows = native_rows()
        rows[0]["tflops"] *= 1.01
        with self.assertRaisesRegex(ValueError, "throughput"):
            self.parse(rows)

    def test_bounded_decimal_output_rounding_is_accepted(self):
        rows = native_rows()
        for row in rows:
            row["tflops"] = format(row["tflops"], ".12g")
        self.assertEqual(len(self.parse(rows)), 3)

    def test_nonfinite_nonpositive_timings_and_rates_fail(self):
        for field in ("time_us", "tflops"):
            for value in ("nan", "inf", "-inf", "0", "-1"):
                with self.subTest(field=field, value=value):
                    rows = native_rows()
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        self.parse(rows)

    def test_exact_arithmetic_gate_rejects_any_nonzero_or_nonfinite_error(self):
        for error in ("1e-30", "-1", "nan", "inf"):
            with self.subTest(error=error):
                rows = native_rows()
                rows[0]["max_abs_error"] = error
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_validation_pass_flag_is_required(self):
        for passed in ("false", "0", "", "TRUE"):
            with self.subTest(passed=passed):
                rows = native_rows()
                rows[0]["passed"] = passed
                with self.assertRaisesRegex(ValueError, "exact-output"):
                    self.parse(rows)

    def test_missing_duplicate_and_out_of_range_trials_fail(self):
        complete = native_rows()
        invalid = ([], complete[:-1], complete + [complete[0]],
                   [{**complete[0], "trial": 3}, *complete[1:]])
        for rows in invalid:
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_trials_may_arrive_out_of_order_if_coverage_is_complete(self):
        self.assertEqual([row["trial"] for row in self.parse(list(reversed(native_rows())))], [2, 1, 0])

    def test_noncanonical_and_negative_integer_fields_fail(self):
        for field in ("trial", "registers", "shared_bytes"):
            for value in ("-1", "01", "+1", "1.0", " 1", "nan"):
                with self.subTest(field=field, value=value):
                    rows = native_rows()
                    rows[0][field] = value
                    with self.assertRaises(ValueError):
                        self.parse(rows)

    def test_resource_and_execution_counts_must_be_positive(self):
        for field in ("blocks", "threads", "chains", "iterations", "unroll", "mma_instructions",
                      "flop_count", "registers", "active_blocks_per_sm"):
            with self.subTest(field=field):
                rows = native_rows()
                rows[0][field] = 0
                with self.assertRaises(ValueError):
                    self.parse(rows)

    def test_zero_trial_index_and_static_shared_bytes_are_valid(self):
        row = self.parse()[0]
        self.assertEqual((row["trial"], row["shared_bytes"]), (0, 0))

    def test_native_header_order_and_fields_are_exact(self):
        valid = csv_text(native_rows())
        for text in ("", "diagnostic\n" + valid,
                     valid.replace("trial,blocks,", "blocks,trial,", 1),
                     valid.replace(",active_blocks_per_sm", "", 1)):
            with self.subTest(text=text[:80]):
                with self.assertRaisesRegex(ValueError, "header"):
                    MEASURE.parse_raw(text, "t128-c4", SETTINGS, DEVICE)

    def test_csv_rows_with_missing_or_extra_columns_fail(self):
        lines = csv_text(native_rows()).splitlines()
        for bad in (lines[1] + ",extra", lines[1].rsplit(",", 1)[0]):
            with self.subTest(bad=bad):
                text = "\n".join([lines[0], bad, *lines[2:]]) + "\n"
                with self.assertRaisesRegex(ValueError, "malformed"):
                    MEASURE.parse_raw(text, "t128-c4", SETTINGS, DEVICE)


class NativeIdentityTests(unittest.TestCase):
    def test_gpu_identity_and_cuda_versions_are_retained(self):
        found = MEASURE.identity(identity_log())
        self.assertEqual((found["gpu"], found["gpu_name"], found["gpu_uuid"]),
                         ("A100", "NVIDIA A100 80GB PCIe", UUID))
        self.assertEqual((found["compute_capability"], found["sm_count"]), ("8.0", 108))
        self.assertEqual(found["cuda_versions"], {"compiled": 12090, "runtime": 12090, "driver": 13000})

    def test_uuid_hexadecimal_case_is_normalized(self):
        self.assertEqual(MEASURE.identity(identity_log(uuid=UUID.upper()))["gpu_uuid"], UUID)

    def test_supported_gpu_families_are_recognized_from_native_name(self):
        for name, family in (("NVIDIA A100-SXM4-80GB", "A100"),
                             ("NVIDIA H100 80GB HBM3", "H100"), ("NVIDIA B200", "B200")):
            with self.subTest(name=name):
                self.assertEqual(MEASURE.identity(identity_log(name=name))["gpu"], family)

    def test_unrelated_gpu_family_fails(self):
        for name in ("Tesla V100-SXM2-32GB", "NVIDIA H200", "NVIDIA A1000"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "requires A100"):
                    MEASURE.identity(identity_log(name=name))

    def test_missing_or_duplicate_device_and_uuid_fail(self):
        text = identity_log()
        for invalid in (text.replace("GPU=", "NOT_GPU=", 1),
                        text.replace("GPU_UUID=", "NOT_UUID=", 1),
                        text + text.splitlines()[0] + "\n", text + f"GPU_UUID={UUID}\n"):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "one device and GPU UUID"):
                    MEASURE.identity(invalid)

    def test_malformed_or_mig_uuid_is_rejected(self):
        for uuid in ("GPU-1234", "GPU-xxxxxxxx-b43c-0f5a-8f59-75d6373ea4e6", "MIG-" + UUID):
            with self.subTest(uuid=uuid):
                with self.assertRaisesRegex(ValueError, "GPU UUID"):
                    MEASURE.identity(identity_log(uuid=uuid))

    def test_versions_must_be_reported_exactly_once(self):
        text = identity_log()
        version = text.splitlines()[-1] + "\n"
        for invalid in (text.replace(version, ""), text + version,
                        text.replace("runtime=12090", "runtime=unknown")):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "CUDA versions"):
                    MEASURE.identity(invalid)


class CandidateSelectionTests(unittest.TestCase):
    def test_highest_median_wins_not_largest_individual_outlier(self):
        rows = []
        for candidate, rates in zip(MEASURE.CANDIDATES, ((100, 100, 10000), (101, 102, 103),
                                                      (99, 100, 101), (90, 90, 90))):
            rows.extend({"candidate": candidate, "tflops": rate} for rate in rates)
        self.assertEqual(MEASURE.selected_candidate(rows), "t128-c8")

    def test_even_trial_median_is_used(self):
        rows = [{"candidate": candidate, "tflops": rate}
                for candidate in MEASURE.CANDIDATES
                for rate in ((100, 104) if candidate == "t256-c4" else (99, 101))]
        self.assertEqual(MEASURE.selected_candidate(rows), "t256-c4")

    def test_ties_have_a_deterministic_lexical_winner(self):
        rows = [{"candidate": candidate, "tflops": 100} for candidate in reversed(MEASURE.CANDIDATES)]
        self.assertEqual(MEASURE.selected_candidate(rows), "t128-c4")

    def test_missing_candidate_is_not_silently_ignored(self):
        rows = [{"candidate": "t128-c4", "tflops": 100}]
        with self.assertRaises(MEASURE.statistics.StatisticsError):
            MEASURE.selected_candidate(rows)


class BuildEvidenceTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build"
        self.build.mkdir()
        self.output = self.root / "measurement"
        self.output.mkdir()
        self.compiler = self.root / "cuda" / "bin" / "nvcc"
        self.compiler.parent.mkdir(parents=True)
        self.compiler.write_text("synthetic compiler path; never executed\n")
        self.dump = self.compiler.with_name("cuobjdump")
        self.dump.write_text("synthetic disassembler path; never executed\n")
        self.binary = self.build / "thff_ideal_mma"
        self.binary.write_text("synthetic binary; never executed\n")
        (self.build / "compile_commands.json").write_text("[]\n")

    def make_cache(self, kind="STRING"):
        (self.build / "CMakeCache.txt").write_text(f"CMAKE_CUDA_COMPILER:{kind}={self.compiler}\n")

    def evidence(self, sass="Function: ideal_mma\nHMMA.1688.F32.TF32 R0, R4, R6, R0;\n"):
        process = SimpleNamespace(returncode=0, stdout=sass, stderr="")
        with patch.object(MEASURE.subprocess, "run", return_value=process) as run, \
                patch.object(MEASURE, "capture", return_value={"test_compiler": True}) as capture:
            result = MEASURE.build_evidence(self.build, self.output, self.binary)
        return result, run, capture

    def test_string_compiler_cache_retains_disassembly_and_build_inputs(self):
        self.make_cache("STRING")
        result, run, capture = self.evidence()
        self.assertEqual(result, {"test_compiler": True})
        self.assertEqual(run.call_args.args[0], [str(self.dump), "--dump-sass", str(self.binary)])
        capture.assert_called_once_with([str(self.compiler), "--version"])
        self.assertIn("HMMA.1688.F32.TF32", (self.output / "disassembly.sass").read_text())
        self.assertEqual((self.output / "build" / "CMakeCache.txt").read_text(),
                         (self.build / "CMakeCache.txt").read_text())
        self.assertTrue((self.output / "build" / "compile_commands.json").is_file())

    def test_filepath_compiler_cache_is_also_supported(self):
        self.make_cache("FILEPATH")
        _, run, capture = self.evidence()
        self.assertEqual(run.call_args.args[0][0], str(self.dump))
        capture.assert_called_once_with([str(self.compiler), "--version"])

    def test_uninitialized_compiler_cache_after_cmake_override_is_supported(self):
        self.make_cache("UNINITIALIZED")
        _, run, capture = self.evidence()
        self.assertEqual(run.call_args.args[0][0], str(self.dump))
        capture.assert_called_once_with([str(self.compiler), "--version"])

    def test_missing_cmake_cache_fails_without_running_disassembler(self):
        with patch.object(MEASURE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "missing CMake build evidence"):
                MEASURE.build_evidence(self.build, self.output, self.binary)
            run.assert_not_called()

    def test_missing_compile_commands_fails_without_running_disassembler(self):
        self.make_cache()
        (self.build / "compile_commands.json").unlink()
        with patch.object(MEASURE.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "missing CMake build evidence"):
                MEASURE.build_evidence(self.build, self.output, self.binary)
            run.assert_not_called()

    def test_missing_disassembler_is_a_concrete_failure(self):
        self.make_cache()
        self.dump.unlink()
        with patch.object(MEASURE.shutil, "which", return_value=None):
            with self.assertRaisesRegex(ValueError, "cuobjdump is required"):
                MEASURE.build_evidence(self.build, self.output, self.binary)

    def test_disassembly_without_tf32_mma_is_rejected(self):
        self.make_cache()
        with self.assertRaisesRegex(ValueError, "disassembly lacks"):
            self.evidence(sass="Function: ideal_mma\nFFMA R0, R4, R6, R0;\n")

    def test_different_tf32_mma_shape_is_rejected(self):
        self.make_cache()
        with self.assertRaisesRegex(ValueError, "disassembly lacks"):
            self.evidence(sass="Function: ideal_mma\nHMMA.16816.F32.TF32 R0, R4, R6, R0;\n")

    def test_disassembler_failure_is_rejected_even_if_output_mentions_mma(self):
        self.make_cache()
        process = SimpleNamespace(returncode=1, stdout="HMMA.1688.F32.TF32 R0, R4, R6, R0;", stderr="failed")
        with patch.object(MEASURE.subprocess, "run", return_value=process):
            with self.assertRaisesRegex(ValueError, "disassembly lacks"):
                MEASURE.build_evidence(self.build, self.output, self.binary)


class CliBoundsTests(unittest.TestCase):
    def invoke(self, environment=None, extra=()):
        argv = ["measure.py", "--output-dir", "/unused/table4-output", "--build-dir", "/unused/table4-build", *extra]
        state = {"collected_unit_gpus": 0, "collected_ideal_gpus": 1, "status": "partial"}
        with patch.object(sys, "argv", argv), patch.dict(os.environ, environment or {}, clear=True), \
                patch.object(MEASURE.DERIVE, "default_sources", return_value=[]), \
                patch.object(MEASURE, "measure", return_value=Path("/unused/table4-ideal")) as measure, \
                patch.object(MEASURE.DERIVE, "generate", return_value=state), \
                patch.object(MEASURE.DERIVE, "read_json", return_value={}), \
                patch.object(MEASURE.DERIVE, "write_json") as write_json, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = MEASURE.main()
        if result == 0:
            metadata = write_json.call_args.args[1]
            self.assertTrue(metadata["gpu_measurements_performed"])
            self.assertEqual(metadata["measurement_directory"], "measurement")
        return result, measure

    def test_invalid_runtime_limits_fail_before_any_measurement(self):
        bounds = {"ITERATIONS": (0, 8193), "TRIALS": (0, 51), "WARMUPS": (-1, 101),
                  "WAVES": (0, 17), "TIMEOUT": (0, 1201)}
        for setting, invalid in bounds.items():
            for value in (*invalid, "not-an-integer"):
                with self.subTest(setting=setting, value=value):
                    result, measure = self.invoke({"THUNDERFFT_TABLE4_" + setting: str(value)})
                    self.assertEqual(result, 1)
                    measure.assert_not_called()

    def test_each_documented_boundary_is_accepted(self):
        for setting, endpoints in {"ITERATIONS": (1, 8192), "TRIALS": (1, 50), "WARMUPS": (0, 100),
                                   "WAVES": (1, 16), "TIMEOUT": (1, 1200)}.items():
            for value in endpoints:
                with self.subTest(setting=setting, value=value):
                    result, measure = self.invoke({"THUNDERFFT_TABLE4_" + setting: str(value)})
                    self.assertEqual(result, 0)
                    self.assertEqual(getattr(measure.call_args.args[0], setting.lower()), value)

    def test_negative_device_fails_before_measurement(self):
        result, measure = self.invoke(extra=("--device", "-1"))
        self.assertEqual(result, 1)
        measure.assert_not_called()

    def test_smoke_and_paper_defaults_are_distinct_and_bounded(self):
        for profile in ("smoke", "paper"):
            with self.subTest(profile=profile):
                result, measure = self.invoke(extra=("--profile", profile))
                self.assertEqual(result, 0)
                args = measure.call_args.args[0]
                self.assertEqual((args.iterations, args.trials, args.warmups),
                                 (64, 3, 2) if profile == "smoke" else (1024, 5, 5))
                self.assertEqual((args.waves, args.timeout), (2, 120))


if __name__ == "__main__":
    unittest.main()
