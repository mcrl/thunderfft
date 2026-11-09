#!/usr/bin/env python3
"""Measure ideal TF32 m16n8k8 throughput, then combine it with Table 3 CSVs."""

import argparse
from datetime import datetime, timezone
import importlib.util
import io
import csv
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact.progress import report as report_progress


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


DERIVE = module("table4_derive_measure", HERE / "derive.py")
DEVICE = module("table4_device", HERE.parent / "table5/select_device.py")
RAW_FIELDS = ("trial", "blocks", "threads", "chains", "iterations", "unroll",
              "mma_instructions", "flop_count", "time_us", "tflops", "max_abs_error",
              "passed", "registers", "shared_bytes", "active_blocks_per_sm")
FIELDS = ("gpu", "gpu_uuid", "candidate", *RAW_FIELDS)
CANDIDATES = {f"t{threads}-c{chains}": (threads, chains)
              for threads in (128, 256) for chains in (4, 8)}


def now():
    return datetime.now(timezone.utc).isoformat()


def capture(argv):
    try:
        run = subprocess.run(argv, text=True, capture_output=True, timeout=15, cwd=ROOT)
        return {"argv": argv, "exit_code": run.returncode,
                "stdout": run.stdout.strip(), "stderr": run.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": str(error)}


def identity(log):
    devices = re.findall(r"^GPU=(.+?) compute=(\d+)\.(\d+) sm_count=(\d+)$", log, re.M)
    uuids = re.findall(r"^GPU_UUID=(GPU-[0-9a-fA-F-]+)$", log, re.M)
    if len(devices) != 1 or len(uuids) != 1 or not re.fullmatch(
            r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", uuids[0]):
        raise ValueError("native benchmark must report one device and GPU UUID")
    name, major, minor, sms = devices[0]
    family = re.search(r"\b(A100|H100|B200)\b", name)
    if not family:
        raise ValueError(f"Table 4 requires A100, H100 or B200; got {name}")
    versions = re.findall(r"^compiled_CUDART=(\d+) runtime=(\d+) driver=(\d+)$", log, re.M)
    if len(versions) != 1:
        raise ValueError("native benchmark did not report CUDA versions")
    return {"gpu": family.group(1), "gpu_name": name, "gpu_uuid": uuids[0].lower().replace("gpu-", "GPU-"),
            "compute_capability": f"{major}.{minor}", "sm_count": int(sms),
            "cuda_versions": dict(zip(("compiled", "runtime", "driver"), map(int, versions[0])))}


def parse_raw(text, candidate, settings, device):
    reader = csv.DictReader(io.StringIO(text))
    if tuple(reader.fieldnames or ()) != RAW_FIELDS:
        raise ValueError("unexpected ideal MMA native CSV header")
    rows, seen = [], set()
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed ideal MMA CSV row")
        integers = {}
        for key in RAW_FIELDS:
            if key in ("time_us", "tflops", "max_abs_error", "passed"):
                continue
            number = int(row[key])
            if str(number) != row[key] or number < 0 or (number == 0 and key not in ("trial", "shared_bytes")):
                raise ValueError(f"invalid native {key}")
            integers[key] = number
        trial = integers["trial"]
        if trial in seen or trial >= settings["trials"]:
            raise ValueError("duplicate or unexpected ideal MMA trial")
        seen.add(trial)
        threads, chains = CANDIDATES[candidate]
        if (integers["threads"], integers["chains"], integers["iterations"], integers["unroll"]) != (
                threads, chains, settings["iterations"], 32):
            raise ValueError("native MMA settings mismatch")
        if integers["blocks"] != device["sm_count"] * integers["active_blocks_per_sm"] * settings["waves"]:
            raise ValueError("native grid disagrees with occupancy and wave count")
        mma = integers["blocks"] * (threads // 32) * chains * settings["iterations"] * 32
        if (integers["mma_instructions"], integers["flop_count"]) != (mma, 2048 * mma):
            raise ValueError("native MMA operation count is inconsistent")
        time = DERIVE.finite_number(row["time_us"], "time_us")
        rate = DERIVE.finite_number(row["tflops"], "tflops")
        error = DERIVE.finite_number(row["max_abs_error"], "max_abs_error", allow_zero=True)
        if not math.isclose(rate, 2048 * mma / time / 1e6, rel_tol=1e-7):
            raise ValueError("native MMA throughput is inconsistent")
        if row["passed"] != "true" or error != 0:
            raise ValueError("native MMA exact-output check failed")
        rows.append({"gpu": device["gpu"], "gpu_uuid": device["gpu_uuid"], "candidate": candidate,
                     **integers, "time_us": time, "tflops": rate, "max_abs_error": error, "passed": "true"})
    if seen != set(range(settings["trials"])):
        raise ValueError("incomplete ideal MMA timing trials")
    return rows


def build_evidence(build, output, binary):
    target = output / "build"
    target.mkdir()
    for filename in ("CMakeCache.txt", "compile_commands.json",
                     "CMakeFiles/thff_ideal_mma.dir/flags.make", "CMakeFiles/thff_ideal_mma.dir/link.txt"):
        source = build / filename
        if source.is_file():
            shutil.copyfile(source, target / source.name)
    if not (target / "compile_commands.json").is_file() or not (target / "CMakeCache.txt").is_file():
        raise ValueError("missing CMake build evidence; use artifact/table4/build.sh")
    cache = (target / "CMakeCache.txt").read_text()
    match = re.search(r"^CMAKE_CUDA_COMPILER:(?:FILEPATH|STRING|UNINITIALIZED)=(.+)$", cache, re.M)
    compiler = Path(match.group(1)) if match else None
    dump = compiler.with_name("cuobjdump") if compiler else None
    if not dump or not dump.is_file():
        found = shutil.which("cuobjdump")
        if not found:
            raise ValueError("cuobjdump is required to retain MMA instruction evidence")
        dump = Path(found)
    run = subprocess.run([str(dump), "--dump-sass", str(binary)], text=True, capture_output=True, timeout=60)
    if run.returncode or not re.search(r"HMMA\.1688\.F32\.TF32\b", run.stdout):
        raise ValueError("binary disassembly lacks the expected TF32 HMMA instruction")
    (output / "disassembly.sass").write_text(run.stdout)
    return capture([str(compiler), "--version"]) if compiler else {}


def selected_candidate(rows):
    return min(CANDIDATES, key=lambda candidate: (
        -statistics.median(row["tflops"] for row in rows if row["candidate"] == candidate), candidate))


def measure(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    measured = output / "measurement"
    measured.mkdir(exist_ok=False)
    (measured / "raw").mkdir()
    settings = {name: getattr(args, name) for name in ("iterations", "trials", "warmups", "waves")}
    settings["candidates"] = list(CANDIDATES)
    binary = (args.build_dir / "thff_ideal_mma").resolve()
    source_files = [HERE / "ideal_mma.cu", HERE / "CMakeLists.txt", HERE / "measure.py", HERE / "build.sh"]
    source_files += list((ROOT / "include").rglob("*.cuh")) + list((ROOT / "include").rglob("*.h"))
    metadata = {"schema_version": 1, "evaluation": "table4_ideal", "profile": args.profile,
                "status": "running", "started_at": now(), "settings": settings,
                "precision": "tf32", "accumulator_precision": "fp32", "mma_shape": "m16n8k8",
                "source_sha256": {str(path.relative_to(ROOT)): DERIVE.sha256(path) for path in sorted(source_files)},
                "binary_sha256": DERIVE.sha256(binary), "commands": [],
                "correctness": "All FP32 accumulator outputs must exactly match iterations*32/128, "
                               "with nonzero exactly representable TF32 A=B=1/32; each timed launch checked outside timing.",
                "method": "Four launch configurations; select highest median TFLOPS, tie by candidate label. "
                          "CUDA-event full launch timing includes final stores; no load/store subtraction.",
                "paper_protocol": "Paper gives MMA shape but not grid, chain count, unroll or selection rule; "
                                  "these reproducible parameters are artifact assumptions, not recovered original settings.",
                "clock_policy": "Observe clocks before/after; no application clock or power changes."}
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = DEVICE.resolve(str(args.device), environment)
    metadata["effective_CUDA_VISIBLE_DEVICES"] = environment["CUDA_VISIBLE_DEVICES"]
    inventory = ["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit",
                 "--format=csv,noheader"]
    try:
        metadata["compiler"] = build_evidence(args.build_dir, measured, binary)
        metadata["gpu_inventory_before"] = capture(inventory)
        DERIVE.write_json(measured / "ideal_metadata.json", metadata)
        all_rows, device = [], None
        for completed, (candidate, (threads, chains)) in enumerate(CANDIDATES.items()):
            argv = [str(binary), "--iterations", str(args.iterations), "--trials", str(args.trials),
                    "--warmups", str(args.warmups), "--waves", str(args.waves),
                    "--chains", str(chains), "--threads", str(threads)]
            record = {"argv": argv, "stdout": f"raw/{candidate}.csv", "stderr": f"raw/{candidate}.log"}
            metadata["commands"].append(record)
            report_progress(completed, len(CANDIDATES), unit="configs", detail=candidate)
            with (measured / record["stdout"]).open("w") as out, (measured / record["stderr"]).open("w") as err:
                run = subprocess.run(argv, cwd=ROOT, env=environment, stdout=out, stderr=err, timeout=args.timeout)
            record["exit_code"] = run.returncode
            if run.returncode:
                raise ValueError(f"ideal MMA {candidate} failed (exit {run.returncode}); see {record['stderr']}")
            found = identity((measured / record["stderr"]).read_text())
            if device is not None and device != found:
                raise ValueError("native device or CUDA versions changed between candidates")
            device = found
            if environment["CUDA_VISIBLE_DEVICES"].startswith("GPU-") and environment["CUDA_VISIBLE_DEVICES"] != device["gpu_uuid"]:
                raise ValueError("native UUID disagrees with the requested GPU")
            all_rows.extend(parse_raw((measured / record["stdout"]).read_text(), candidate, settings, device))
            metadata.update(device)
            metadata["multiprocessor_count"] = device["sm_count"]
            DERIVE.write_json(measured / "ideal_metadata.json", metadata)
            report_progress(completed + 1, len(CANDIDATES), unit="configs", detail=candidate)
        DERIVE.write_csv(measured / "ideal.csv", all_rows, FIELDS)
        metadata.update(status="completed", finished_at=now(), selected_candidate=selected_candidate(all_rows),
                        gpu_inventory_after=capture(inventory))
        metadata["outputs"] = {str(path.relative_to(measured)): DERIVE.sha256(path)
                               for path in sorted(measured.rglob("*"))
                               if path.is_file() and path.name != "ideal_metadata.json"}
        DERIVE.write_json(measured / "ideal_metadata.json", metadata)
        return measured
    except BaseException as error:
        metadata.update(status="failed", finished_at=now(), error=str(error))
        DERIVE.write_json(measured / "ideal_metadata.json", metadata)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--source-dir", action="append", type=Path)
    args = parser.parse_args()
    defaults = {"iterations": 1024 if args.profile == "paper" else 64,
                "trials": 5 if args.profile == "paper" else 3, "warmups": 5 if args.profile == "paper" else 2,
                "waves": 2, "timeout": 120}
    bounds = {"iterations": (1, 8192), "trials": (1, 50), "warmups": (0, 100),
              "waves": (1, 16), "timeout": (1, 1200)}
    try:
        for name, default in defaults.items():
            value = int(os.environ.get("THUNDERFFT_TABLE4_" + name.upper(), default))
            low, high = bounds[name]
            if not low <= value <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
            setattr(args, name, value)
        if args.device < 0:
            raise ValueError("device must be nonnegative")
        sources = args.source_dir if args.source_dir is not None else DERIVE.default_sources()
        # Validate supplied Table 3 inputs before running GPU work.
        for source in sources:
            DERIVE.load_source(source, args.profile)
        measured = measure(args)
        state = DERIVE.generate(sources, args.output_dir, args.profile, ideal_dirs=[measured])
        metadata_path = args.output_dir / "metadata.json"
        metadata = DERIVE.read_json(metadata_path)
        metadata.update(gpu_measurements_performed=True, measurement_directory="measurement",
                        operation="Ideal MMA GPU measurement followed by Table 3 CSV derivation and table assembly")
        DERIVE.write_json(metadata_path, metadata)
    except (ValueError, OSError, subprocess.SubprocessError) as error:
        print(f"Table 4 measurement failed: {error}", file=sys.stderr)
        return 1
    print(f"Table 4: {state['collected_unit_gpus']} unit and {state['collected_ideal_gpus']} ideal MMA GPU rows; "
          f"{state['status']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
