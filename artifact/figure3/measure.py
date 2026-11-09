#!/usr/bin/env python3
"""Collect Figure 3: communication-only copies and end-to-end ThunderFFT, cuFFT
and cuFFTDx FFTs, then normalize each by the communication-only time."""

import argparse
import csv
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import shlex
import statistics
import subprocess
import sys


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLE3 = module("figure3_table3_measure", HERE.parent / "table3/measure.py")
SIZES = (64, 256, 1024, 4096)
PRECISIONS = ("tf32", "fp16")
# Native --library name -> (Figure 3 library label, placement argument).
NATIVE = {"copy": ("Comm. only", "global"), "copy-scalar": ("Comm. only (scalar)", "global"),
          "thunderfft": ("ThunderFFT", "register"), "cufft": ("cuFFT", "global"),
          "cufftdx": ("cuFFTDx", "register")}
COPIES = ("copy", "copy-scalar")
# cuFFT runs FP32 only; cuFFTDx pairs FP16 with FP16 and TF32 with FP32 (Table 3).
COMM = "Comm. only"
# Complex storage bytes per element; cuFFT/cuFFTDx FP32 pairs with TF32 storage.
ELEMENT_BYTES = {"fp16": 4, "tf32": 8, "fp32": 8}
FIELDS = ("gpu", "N", "batch", "library", "precision", "repeat", "time_ms", "bytes",
          "rel_l2", "passed", "source")
SUMMARY_FIELDS = ("gpu", "N", "batch", "library", "precision", "bytes", "repeats",
                  "median_time_ms", "comm_median_time_ms", "normalized")


def parse_native(text, *, library, n, precision, batch, repeats, trials, gpu, raw_csv):
    """Check the native CSV; keep the out-of-place single-pass time per trial."""
    labeled = library != "thunderfft"
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != TABLE3.RAW_FIELDS + (("library",) if labeled else ()):
        raise ValueError(f"native CSV header does not match the {library} contract")
    rows, seen = [], set()
    for raw in reader:
        if None in raw or any(value is None for value in raw.values()):
            raise ValueError("malformed native CSV row")
        actual = (int(raw["N"]), raw["precision"], raw["placement"], int(raw["batch"]), int(raw["inside_repeats"]))
        if actual != (n, precision, NATIVE[library][1], batch, repeats):
            raise ValueError(f"native CSV configuration mismatch: {actual}")
        if labeled and raw["library"] != library:
            raise ValueError("native CSV library label mismatch")
        trial = int(raw["trial"])
        if trial in seen or not 0 <= trial < trials:
            raise ValueError("duplicate or out-of-range native trial")
        seen.add(trial)
        e2e = float(raw["e2e_out_of_place_us"])
        rel_l2 = float(raw["single_fft_relL2"])
        if not (math.isfinite(e2e) and e2e > 0):
            raise ValueError("end-to-end time must be positive and finite")
        # Copies are checked byte-exact natively and report 0.
        if not math.isfinite(rel_l2) or not 0 <= rel_l2 <= (0 if library in COPIES else 0.005):
            raise ValueError("native correctness gate failed")
        rows.append({"gpu": gpu, "N": n, "batch": batch, "library": NATIVE[library][0],
                     "precision": precision, "repeat": trial, "time_ms": e2e / 1000.0,
                     "bytes": 2 * n * batch * ELEMENT_BYTES[precision], "rel_l2": rel_l2,
                     "passed": "true", "source": raw_csv})
    if seen != set(range(trials)):
        raise ValueError(f"expected {trials} complete trials for {library}, got {len(seen)}")
    return rows


def parse_cufftdx(text, *, n, precision, batch, repeats, trials, gpu, raw_csv):
    """Table 3's gated cuFFTDx output; keep the candidate with the fastest median e2e."""
    rows = TABLE3.parse_raw(text, n=n, precision=precision, placement="register", batch=batch,
                            repeats=repeats, trials=trials, gpu=gpu, raw_csv=raw_csv, library="cuFFTDx")
    by_candidate = {}
    for row in rows:
        by_candidate.setdefault(row["candidate"], []).append(row)
    best = min(by_candidate, key=lambda name: statistics.median(r["e2e_out_of_place_us"] for r in by_candidate[name]))
    return best, [{"gpu": gpu, "N": n, "batch": batch, "library": "cuFFTDx", "precision": precision,
                   "repeat": row["trial"], "time_ms": row["e2e_out_of_place_us"] / 1000.0,
                   "bytes": 2 * n * batch * ELEMENT_BYTES[precision], "rel_l2": row["rel_l2"],
                   "passed": "true", "source": f"{raw_csv}#{best}"} for row in by_candidate[best]]


def summarize(rows):
    """normalized = median communication-only time / median implementation time,
    paired by GPU, N, batch and bytes moved."""
    groups = {}
    for row in rows:
        key = row["gpu"], int(row["N"]), int(row["batch"]), row["library"], row["precision"], int(row["bytes"])
        groups.setdefault(key, []).append(float(row["time_ms"]))
    comm = {(gpu, n, batch, size): statistics.median(times)
            for (gpu, n, batch, library, _, size), times in groups.items() if library == COMM}
    summary = []
    for (gpu, n, batch, library, precision, size), times in sorted(groups.items()):
        median = statistics.median(times)
        reference = comm.get((gpu, n, batch, size))
        summary.append({"gpu": gpu, "N": n, "batch": batch, "library": library, "precision": precision,
                        "bytes": size, "repeats": len(times), "median_time_ms": median,
                        "comm_median_time_ms": reference if reference is not None else "",
                        "normalized": reference / median if reference is not None else ""})
    return summary


def selected_libraries(cufftdx_available=False):
    requested = os.environ.get("THUNDERFFT_FIGURE3_LIBRARIES")
    if requested is None:
        return tuple(name for name in NATIVE if name != "cufftdx" or cufftdx_available)
    result = tuple(value.strip() for value in requested.split(","))
    if result == ("all",):
        result = tuple(NATIVE)
    if not result or any(value not in NATIVE for value in result) or len(result) != len(set(result)):
        raise ValueError("THUNDERFFT_FIGURE3_LIBRARIES must be all or unique comma-separated names: " + ",".join(NATIVE))
    if "cufftdx" in result and not cufftdx_available:
        raise ValueError("Selected cuFFTDx requires MATHDX_ROOT and the cuFFTDx executable")
    return result


def expected_configurations(precisions=("tf32",), sizes=SIZES):
    result = []
    for n in sizes:
        for precision in precisions:
            result.extend((n, library, precision) for library in ("Comm. only", "ThunderFFT"))
            result.append((n, "cuFFTDx", TABLE3.MATCHED_PRECISION[precision]))
            if precision == "tf32":
                result.append((n, "cuFFT", "fp32"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--cufftdx-binary", type=Path, help="thff_paper_fft_cufftdx; omit to leave cuFFTDx pending")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    raw = args.output / "raw"
    raw.mkdir(exist_ok=True)
    try:
        libraries = selected_libraries(bool(args.cufftdx_binary))
        if any(name != "cufftdx" for name in libraries) and not args.binary:
            raise ValueError("Selected libraries require --binary")
        settings = {"batch": TABLE3.positive_env("THUNDERFFT_FIGURE3_BATCH", 1024 if args.profile == "smoke" else 65536),
                    # Figure 3 uses only the separate single-pass e2e timing; the in-kernel R/2R loop
                    # (Table 3's compute time) just has to satisfy the native contract.
                    "repeats": TABLE3.positive_env("THUNDERFFT_FIGURE3_REPEATS", 16),
                    "trials": TABLE3.positive_env("THUNDERFFT_FIGURE3_TRIALS", 2 if args.profile == "smoke" else 3),
                    "timeout_seconds_per_configuration": TABLE3.positive_env("THUNDERFFT_FIGURE3_TIMEOUT", 300)}
        selection = {"libraries": libraries, "N": TABLE3.selection_env("THUNDERFFT_FIGURE3_N_LIST", SIZES, int),
                     # Default TF32: the earlier A100 evaluation measured FP32 storage.
                     "precisions": TABLE3.selection_env("THUNDERFFT_FIGURE3_PRECISIONS", PRECISIONS)
                                   if "THUNDERFFT_FIGURE3_PRECISIONS" in os.environ else ("tf32",)}
    except ValueError as error:
        parser.error(str(error))
    if settings["batch"] % 16:
        parser.error("THUNDERFFT_FIGURE3_BATCH must be a positive multiple of 16")
    if args.profile == "paper" and settings["batch"] != 65536:
        parser.error("paper profile requires batch=65536; use smoke for custom checks")
    metadata = {"schema_version": 1, "evaluation": "figure3", "profile": args.profile,
                "started_at": TABLE3.now(), "status": "running", "settings": settings, "selection": selection,
                "scope": "communication-only, ThunderFFT, cuFFT (FP32) and cuFFTDx (with MATHDX_ROOT) on one GPU",
                "full_paper_comparison_complete": False,
                "timing": {"metric": "e2e_out_of_place_us: mean of 10 back-to-back single-pass launches per trial",
                           "boundary": "kernel only; twiddle initialization, allocation and input generation excluded",
                           "central_statistic": "median over trials",
                           "normalized": "Comm. only (int4 vector copy) median / implementation median, same N and bytes"},
                "correctness": {"ThunderFFT/cuFFT/cuFFTDx": "single-pass and repeated launches, relL2 <= 0.005 vs cuFFT FP64",
                                "Comm. only": "byte-exact copy of the validation input before timing"},
                "communication_only": {"Comm. only": "int4 (16-byte) grid-stride global load/store",
                                       "Comm. only (scalar)": "one complex element per load/store",
                                       "bytes": "2 * N * batch * complex element bytes (read + write)",
                                       "layout": "out of place, same input/output types as the FFT"},
                "cufftdx": "Table 3 candidates; selected by smallest median e2e per N/precision",
                "cufft": "cufftExecC2C out of place; plan creation excluded from timing",
                "run_id": os.environ.get("THUNDERFFT_RUN_ID"),
                "effective_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "git": TABLE3.capture(["git", "rev-parse", "HEAD"]),
                "worktree": TABLE3.capture(["git", "status", "--porcelain"]),
                "source_sha256": dict(TABLE3.source_hashes(), **{
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in sorted(HERE.glob("*.py")) + [HERE / "run.sh"]}),
                "build_configuration": TABLE3.build_evidence(args.build_dir),
                "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest() if args.binary else None,
                "cufftdx_binary_sha256": hashlib.sha256(args.cufftdx_binary.read_bytes()).hexdigest()
                  if args.cufftdx_binary else None, "cufftdx_selection": {},
                "gpu_inventory_before": TABLE3.capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": []}
    jobs = []
    for n in selection["N"]:
        for precision in selection["precisions"]:
            jobs += [(n, library, precision) for library in libraries if library in ("copy", "copy-scalar", "thunderfft")]
            if precision == "tf32" and "cufft" in libraries:
                jobs.append((n, "cufft", "fp32"))
            if "cufftdx" in libraries:
                jobs.append((n, "cufftdx", TABLE3.MATCHED_PRECISION[precision]))
    if not jobs:
        parser.error("selected libraries have no jobs for the requested precision (cuFFT requires tf32 storage)")
    rows, identity = [], None
    TABLE3.atomic_json(args.output / "metadata.json", metadata)
    try:
        for completed, (n, library, precision) in enumerate(jobs):
            label = f"n{n}-{precision}-{library}"
            binary = args.cufftdx_binary if library == "cufftdx" else args.binary
            command = [str(binary.resolve()), "--library", library, str(n), precision,
                       NATIVE[library][1], str(settings["repeats"]), str(settings["trials"]),
                       str(settings["batch"])]
            csv_path, log_path = raw / f"{label}.csv", raw / f"{label}.log"
            record = {"argv": command, "stdout": str(csv_path.relative_to(args.output)),
                      "stderr": str(log_path.relative_to(args.output)), "started_at": TABLE3.now()}
            metadata["commands"].append(record)
            with (args.output / "commands.txt").open("a") as stream:
                stream.write(shlex.join(command) + "\n")
            print(f"Figure 3 [{label}] batch={settings['batch']}", flush=True)
            detail = f"{NATIVE[library][0]} N={n} {precision}"
            TABLE3.report_progress(completed, len(jobs), unit="configs", detail=detail)
            with csv_path.open("w") as stdout, log_path.open("w") as stderr:
                try:
                    process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                             timeout=settings["timeout_seconds_per_configuration"], check=False)
                except subprocess.TimeoutExpired:
                    record["exit_code"] = 124
                    raise RuntimeError(f"{label}: timed out; see {log_path}")
            record.update(exit_code=process.returncode, finished_at=TABLE3.now())
            if process.returncode:
                raise RuntimeError(f"{label}: executable exited {process.returncode}; see {log_path}")
            log = log_path.read_text()
            this_identity = TABLE3.gpu_name(log) + (TABLE3.gpu_uuid(log),)
            if identity is not None and this_identity != identity:
                raise ValueError("CUDA device identity changed during the run")
            identity = this_identity
            metadata.update(gpu=identity[0], gpu_name=identity[1], compute_capability=identity[2],
                            gpu_uuid=identity[3])
            options = dict(n=n, precision=precision, batch=settings["batch"], repeats=settings["repeats"],
                           trials=settings["trials"], gpu=identity[0], raw_csv=str(csv_path.relative_to(args.output)))
            if library == "cufftdx":
                best, measured = parse_cufftdx(csv_path.read_text(), **options)
                metadata["cufftdx_selection"][f"N{n}-{precision}"] = best
                rows += measured
            else:
                rows += parse_native(csv_path.read_text(), library=library, **options)
            TABLE3.write_csv(args.output / "data.csv", rows, FIELDS)
            TABLE3.atomic_json(args.output / "metadata.json", metadata)
            TABLE3.report_progress(completed + 1, len(jobs), unit="configs", detail=detail)
        # Baselines are also kept as their own CSV (Figure 3's separate comparison file).
        TABLE3.write_csv(args.output / "baselines.csv",
                         [row for row in rows if row["library"] in ("cuFFT", "cuFFTDx")], FIELDS)
        TABLE3.write_csv(args.output / "summary.csv", summarize(rows), SUMMARY_FIELDS)
        present = {(int(row["N"]), row["library"], row["precision"]) for row in rows}
        metadata["missing"] = [f"N{n}/{library}/{precision}" for n, library, precision
                               in expected_configurations(tuple(sorted(set(selection["precisions"]) | {"tf32"})))
                               if (n, library, precision) not in present]
        metadata["full_paper_comparison_complete"] = not metadata["missing"]
        metadata.update(status="completed", numerical_and_timing_gates_passed=True)
    except (OSError, ValueError, RuntimeError) as error:
        metadata.update(status="failed", error=str(error), numerical_and_timing_gates_passed=False)
        print(f"Figure 3 failed: {error}", file=sys.stderr)
        return 1
    finally:
        metadata["finished_at"] = TABLE3.now()
        TABLE3.atomic_json(args.output / "metadata.json", metadata)
    missing = ", ".join(metadata["missing"])
    print(f"Completed Figure 3 measurements for N={list(selection['N'])}"
          + (f"; pending: {missing} (cuFFTDx needs MATHDX_ROOT)" if missing else "") + f". Results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
