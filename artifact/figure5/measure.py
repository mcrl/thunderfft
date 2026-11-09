#!/usr/bin/env python3
"""Collect Figure 5: correctness-gated ThunderFFT FFT convolution and cuDNN over a
filter-length sweep, joined with separately produced backend CSVs (FlashFFTConv).

The sweep/collection helpers are shared with Figure 4 (artifact/figure4/measure.py).
"""

import argparse
import csv
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import subprocess
import sys
import time


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TABLE3 = module("conv_table3_measure", HERE.parent / "table3/measure.py")
SELECTION = module("conv_selection", HERE / "selection.py")
POLICY = module("conv_cudnn_policy", HERE / "policy.py")
# Backend adapters (cuDNN, FlashFFTConv, cuFFT, cuFFTDx) write these columns.
BACKEND_FIELDS = ("gpu", "filter", "library", "repeat", "time_ms", "rel_l2", "passed", "operation")
FIELDS = ("gpu", "shape", "filter", "library", "precision", "repeat", "time_ms", "filter_fft_included",
          "rel_l2", "validated", "passed", "operation", "source")
BUSY_RETRIES = 20
SUMMARY_FIELDS = ("gpu", "shape", "filter", "library", "repeats", "median_time_ms", "min_time_ms", "max_time_ms")

FIGURE5 = {
    "evaluation": "figure5", "title": "Figure 5", "env": "THUNDERFFT_FIGURE5",
    "shape": "L=4096,B=64,D=768",
    # Paper axis 3..51; the earlier A100 evaluation measured odd lengths.
    "filters": {"smoke": (3, 9, 53), "paper": tuple(range(3, 54, 2))},
    "allowed_filters": tuple(range(1, 4097)),
    "operation": "depthwise circular convolution y[n]=sum_k x[(n-k) mod L] h[k], filter not flipped, "
                 "full L outputs, real fp16 input/output; fp32 spatial filter for FFT paths, fp16-rounded filter for cuDNN; x~U[0,1] (minstd seed 42), h~U[0,1] (seed 42+K)",
    "precision": "fp16",
    "truncate": ("cuDNN",),
    "libraries": ("ThunderFFT", "cuDNN", "FlashFFTConv"),
    "notes": {"ThunderFFT": "FP16 input/output and FFT arithmetic; spatial filter FP32, cuFFT filter FFT then FP16 conversion",
              "FlashFFTConv": "unmodified pinned upstream 4096 FP16 kernel; native cuFFT filter FFT and frequency permutation/conversion timed; spatial filter FP32",
              "cuDNN": "FP16 input, spatial filter and output; the same generated filter is rounded to FP16"},
}


# Each native invocation sweeps a comma-separated filter list with in-process trials,
# so the input is generated once per process and each filter is gated once.
def figure5_command(binary, sizes, profile, trials):
    batch, channels = (64, 768) if profile == "paper" else (4, 64)
    sample = os.environ.get("THUNDERFFT_FIGURE5_VALIDATE_CHANNELS", "1024")
    # N K-list device B D warmup runs include_filter_fft validate real_input real_filter validate_channels trials
    return [str(binary), "4096", ",".join(map(str, sizes)), "0", str(batch), str(channels), "5", "30",
            "1", "1", "1", "1", sample, str(trials)], f"L=4096,B={batch},D={channels}"


def figure5_cudnn_command(binary, sizes, profile, trials):
    batch, channels = (64, 768) if profile == "paper" else (4, 64)
    sample = os.environ.get("THUNDERFFT_FIGURE5_VALIDATE_CHANNELS", "1024")
    return [str(binary), "--figure5", "4096", ",".join(map(str, sizes)), str(batch), str(channels), "5", "30",
            sample, str(trials)], f"L=4096,B={batch},D={channels}"


def selected_libraries(spec, available):
    requested = SELECTION.normalize(os.environ.get(spec["env"] + "_LIBRARIES", "auto"), spec["libraries"])
    if requested == "auto":
        return tuple(library for library in spec["libraries"] if library in available)
    values = tuple(requested.split(","))
    missing = set(values) - set(available)
    if missing:
        raise ValueError(f"requested libraries are unavailable: {', '.join(sorted(missing))}")
    return values


def figure5_flash_command(binary, sizes, profile, trials):
    batch, channels = (64, 768) if profile == "paper" else (4, 64)
    sample = os.environ.get("THUNDERFFT_FIGURE5_VALIDATE_CHANNELS", "1024")
    return [str(binary), ",".join(map(str, sizes)), str(batch), str(channels), sample, str(trials), "5", "30"], \
        f"L=4096,B={batch},D={channels}"


def parse_blocks(stdout, gate):
    """Split native output at "Filter size: f" lines; gate(block) returns
    (rel_l2, validated) or raises. Yields (size, trial, time_ms, rel_l2, validated)."""
    blocks = re.split(r"^Filter size: (\d+)$", stdout, flags=re.MULTILINE)
    results = []
    for size, block in zip(blocks[1::2], blocks[2::2]):
        rel_l2, validated = gate(block)
        times = re.findall(r"trial=(\d+)\) time: (\S+) ms$", block, re.MULTILINE)
        if not times:
            raise ValueError(f"f={size}: native output has no timing lines")
        results += [(int(size), int(trial), float(ms), rel_l2, validated) for trial, ms in times]
    return results


def figure5_gate(block):
    if "Circular convolution validation: PASS" not in block:
        raise ValueError("native circular-convolution correctness gate did not pass")
    rel = re.search(r"direct circular conv \((?:raw/N|cuDNN|FlashFFTConv)\): L2_rel=(\S+)", block)
    checked = re.search(r"^Validation sequences: (\d+) of (\d+)$", block, re.MULTILINE)
    if not (rel and checked):
        raise ValueError("native output lacks validation lines")
    return float(rel.group(1)), f"{checked.group(1)}/{checked.group(2)} sequences"


def figure5_parse(stdout):
    return parse_blocks(stdout, figure5_gate)


def check_row(row):
    time_ms, rel_l2 = float(row["time_ms"]), float(row["rel_l2"])
    if not (math.isfinite(time_ms) and time_ms > 0):
        raise ValueError(f"{row['library']} f={row['filter']}: time must be positive and finite")
    if not (math.isfinite(rel_l2) and 0 <= rel_l2 <= 5e-3):
        raise ValueError(f"{row['library']} f={row['filter']}: relative L2 gate failed")
    return time_ms


def read_backends(paths, spec, gpu, shape, filters, measured=("ThunderFFT",)):
    """Join backend CSVs; reject rows whose operation or workload differs from ours."""
    rows, hashes, seen = [], {}, set()
    for path in paths:
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            if not set(BACKEND_FIELDS + ("shape",)) <= set(reader.fieldnames or ()):
                raise ValueError(f"{path}: backend CSV needs columns {BACKEND_FIELDS + ('shape',)}")
            for row in reader:
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"{path}: malformed backend CSV row")
                if row["library"] not in spec["libraries"] or row.get("precision", spec["precision"]) != spec["precision"]:
                    raise ValueError(f"{path}: unsupported backend library/precision")
                key = (row["library"], int(row["filter"]), int(row["repeat"]))
                if key in seen or key[-1] < 0:
                    raise ValueError(f"{path}: duplicate or negative backend trial")
                seen.add(key)
                if row["library"] in measured:
                    raise ValueError(f"{path}: backend rows cannot replace measured {row['library']}")
                if row["passed"].strip().lower() not in ("true", "1"):
                    raise ValueError(f"{path}: unvalidated {row['library']} f={row['filter']}")
                if (row["gpu"], row["shape"]) != (gpu, shape) or int(row["filter"]) not in filters:
                    raise ValueError(f"{path}: backend GPU/shape/filter does not match this run")
                if row["operation"] != spec["operation"]:
                    raise ValueError(f"{path}: {row['library']} operation differs: {row['operation']!r}")
                check_row(row)
                normalized = dict({key: row.get(key, "") for key in FIELDS[:-1]}, source=str(path))
                normalized["precision"] = row.get("precision") or spec["precision"]
                normalized["filter_fft_included"] = row.get("filter_fft_included") or "unknown"
                rows.append(normalized)
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return rows, hashes


def snapshot_backends(paths, output, spec, gpu, shape, filters, measured):
    """Retain byte-identical backend CSVs inside the immutable result bundle."""
    rows, hashes, origins, seen = [], {}, [], set()
    for index, path in enumerate(paths):
        payload = path.read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        relative = f"raw/backends/{index:03d}-{sha[:16]}.csv"
        saved = output / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(payload)
        imported, _ = read_backends([saved], spec, gpu, shape, filters, measured)
        for row in imported:
            key = (row["library"], int(row["filter"]), int(row["repeat"]))
            if key in seen:
                raise ValueError("duplicate backend measurement across imported CSVs")
            seen.add(key)
            row["source"] = relative
        rows.extend(imported)
        hashes[relative] = sha
        origins.append({"original_path": str(path), "path": relative, "sha256": sha})
    return rows, hashes, origins


def measured_identity(metadata, log):
    """Resolve actual CUDA identity from native/probe evidence, never ordinal guessing."""
    gpu, name, capability = TABLE3.gpu_name(log)
    native_uuid = TABLE3.gpu_uuid(log)
    probe = metadata.get("device_uuid_probe", {})
    probed_uuid = TABLE3.gpu_uuid(probe.get("stdout", "")) if probe.get("exit_code") == 0 else None
    mask = metadata.get("effective_CUDA_VISIBLE_DEVICES", "") or ""
    masked_uuid = mask if re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", mask) else None
    known = {value for value in (native_uuid, probed_uuid, masked_uuid, metadata.get("gpu_uuid")) if value}
    if len(known) > 1:
        raise ValueError("native/probed/recorded CUDA UUIDs disagree")
    inventory = metadata.get("gpu_inventory_before", {})
    devices = list(csv.reader(io.StringIO(inventory.get("stdout", "")), skipinitialspace=True))
    candidates = [fields for fields in devices if len(fields) >= 4 and fields[2].strip() == name
                  and (not known or fields[1].strip() in known)]
    if inventory.get("exit_code") != 0 or len(candidates) != 1:
        raise ValueError("cannot identify the measured CUDA device unambiguously from UUID and GPU inventory")
    fields = candidates[0]
    actual_uuid = fields[1].strip()
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", actual_uuid):
        raise ValueError("invalid measured GPU UUID")
    return gpu, name, capability, actual_uuid, fields[3].strip()


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["gpu"], row["shape"], int(row["filter"]), row["library"]), []).append(float(row["time_ms"]))
    summary = [{"gpu": gpu, "shape": shape, "filter": size, "library": library, "repeats": len(times),
                "median_time_ms": statistics.median(times), "min_time_ms": min(times), "max_time_ms": max(times)}
               for (gpu, shape, size, library), times in sorted(groups.items())]
    # Per-library mean over the swept filters (the paper reports average runtimes).
    for library in sorted({row["library"] for row in summary}):
        medians = [row["median_time_ms"] for row in summary if row["library"] == library]
        summary.append({"gpu": summary[0]["gpu"], "shape": summary[0]["shape"], "filter": f"mean({len(medians)})",
                        "library": library, "repeats": "", "median_time_ms": statistics.mean(medians),
                        "min_time_ms": "", "max_time_ms": ""})
    return summary


def summarize_wide(rows, libraries):
    """One row per filter with the mean time over trials of each library in its own
    lower-case column (empty when that cell was not measured)."""
    groups = {}
    for row in rows:
        groups.setdefault((int(row["filter"]), row["library"]), []).append(float(row["time_ms"]))
    return [{"filter": size, **{library.lower(): f"{statistics.mean(groups[size, library]):.4f}"
                                if (size, library) in groups else "" for library in libraries}}
            for size in sorted({size for size, _ in groups})]


def write_summary(path, rows, spec):
    """Long median/min/max summary, or the wide per-filter mean table when the spec
    sets summary="wide_mean" (Figure 4)."""
    if spec.get("summary") == "wide_mean":
        columns = spec["summary_columns"]
        TABLE3.write_csv(path, summarize_wide(rows, columns), ("filter", *(c.lower() for c in columns)))
    else:
        TABLE3.write_csv(path, summarize(rows), SUMMARY_FIELDS)


def diverged(rows, library, size, ratio):
    """True when `library` is more than `ratio` times slower than the fastest other
    library measured at this filter size (median over trials)."""
    medians = {}
    for row in rows:
        if int(row["filter"]) == size:
            medians.setdefault(row["library"], []).append(float(row["time_ms"]))
    others = [statistics.median(v) for name, v in medians.items() if name != library]
    return ratio > 0 and library in medians and bool(others) and \
        statistics.median(medians[library]) > ratio * min(others)


def collect(spec, runners, build_dir, output, profile, sources):
    """Batch selected filters per library; write validated data and explicit omissions.

    runners: (library, binary, command, parse, filter_fft_included) tuples.
    """
    env = spec["env"]
    chosen = selected_libraries(spec, tuple(r[0] for r in runners))
    runners = [r for r in runners if r[0] in chosen]
    if not runners:
        raise ValueError("no runnable convolution backends selected")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    trials = TABLE3.positive_env(f"{env}_TRIALS", 1 if profile == "smoke" else 3)
    timeout = TABLE3.positive_env(f"{env}_TIMEOUT", 1800)
    filters = (TABLE3.selection_env(f"{env}_FILTERS", spec["allowed_filters"], int)
               if f"{env}_FILTERS" in os.environ else spec["filters"][profile])
    cudnn_mode = POLICY.validate_mode(os.environ.get(POLICY.ENVIRONMENT, "fast"))
    policy_skipped = POLICY.skipped_cells(spec["evaluation"], filters, chosen, cudnn_mode)
    filters_by_library = {library: POLICY.selected_filters(spec["evaluation"], filters, library, cudnn_mode)
                          for library in chosen}
    runners = [r for r in runners if filters_by_library[r[0]]]
    if not runners:
        raise ValueError("fast mode skips every selected cuDNN filter; use --cudnn-mode full")
    backends = [Path(path) for path in os.environ.get(f"{env}_BACKENDS", "").split(",") if path]
    # Stop timing a direct baseline once it is far slower than the FFT paths and
    # still growing with the filter; the plot leaves the remaining sizes empty.
    requested_slow_ratio = float(os.environ.get(f"{env}_SLOW_RATIO", "0"))
    if not math.isfinite(requested_slow_ratio) or requested_slow_ratio < 0:
        raise ValueError(f"{env}_SLOW_RATIO must be finite and nonnegative")
    slow_ratio = 0 if cudnn_mode == "full" else requested_slow_ratio
    truncated = {}
    metadata = {"schema_version": 1, "evaluation": spec["evaluation"], "profile": profile,
                "started_at": TABLE3.now(), "status": "running", "operation": spec["operation"],
                "notes": spec.get("notes"),
                "settings": {"trials": trials, "filters": list(filters), "timeout_seconds": timeout,
                             "cudnn_mode": cudnn_mode, "selected_libraries": list(chosen),
                             "requested_slow_ratio": requested_slow_ratio,
                             "slow_ratio": slow_ratio, "truncatable": list(spec.get("truncate", ()))},
                "truncated": truncated,
                "policy_skipped": policy_skipped,
                "scope": f"{', '.join(r[0] for r in runners)} measured here; other libraries from {env}_BACKENDS",
                "full_paper_comparison_complete": False,
                "timing": {"metric": "native mean over its timed launches per trial",
                           "central_statistic": "median over trials",
                           "filter_fft_included": {r[0]: r[4] for r in runners}},
                "correctness": "native gate before the time is accepted: finite relL2 <= 5e-3 vs direct reference",
                "run_id": os.environ.get("THUNDERFFT_RUN_ID"),
                "effective_CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "device_uuid_probe": TABLE3.capture([sys.executable, str(HERE / "device_identity.py")]),
                "git": TABLE3.capture(["git", "rev-parse", "HEAD"]),
                "worktree": TABLE3.capture(["git", "status", "--porcelain"]),
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(set(sources)) if path.is_file()},
                "build_cache": TABLE3.build_evidence(build_dir).get("CMakeCache.txt"),
                "binary_sha256": {r[0]: hashlib.sha256(r[1].read_bytes()).hexdigest() for r in runners},
                "gpu_inventory_before": TABLE3.capture(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,memory.total,clocks.sm,clocks.mem,power.limit", "--format=csv,noheader"]),
                "commands": []}
    external = build_dir / "artifact_figure5/build.json"
    if "FlashFFTConv" in chosen and external.is_file():
        import json
        metadata["external_build"] = json.loads(external.read_text())
        if metadata["external_build"].get("binary_sha256") != metadata["binary_sha256"]["FlashFFTConv"]:
            raise ValueError("FlashFFTConv build manifest does not match the selected binary")
    rows, identity, shape = [], None, None
    completed = 0
    total = sum(len(filters_by_library[r[0]]) for r in runners)
    TABLE3.atomic_json(output / "metadata.json", metadata)
    try:
        for library, binary, command, parse, filter_fft in runners:
            # One process per library for the whole sweep; a truncatable library runs
            # one size per process so it can stop once it diverges.
            cut = library in spec.get("truncate", ()) and slow_ratio > 0
            selected = filters_by_library[library]
            for group in ([[size] for size in selected] if cut else [list(selected)]):
                if library in truncated:
                    break
                argv, shape = command(binary.resolve(), group, profile, trials)
                label = f"{library.lower()}-f{group[0]}" + (f"-f{group[-1]}" if len(group) > 1 else "")
                out_path, log_path = raw / f"{label}.out", raw / f"{label}.log"
                record = {"library": library, "argv": argv, "stdout": str(out_path.relative_to(output)),
                          "stderr": str(log_path.relative_to(output)), "started_at": TABLE3.now()}
                metadata["commands"].append(record)
                with (output / "commands.txt").open("a") as stream:
                    stream.write(shlex.join(argv) + "\n")
                print(f"{spec['title']} [{label}] {shape} trials={trials}", flush=True)
                detail = f"{library} filter={group[0]}" + (f"..{group[-1]}" if len(group) > 1 else "")
                TABLE3.report_progress(completed, total, unit="cases", detail=detail)
                # Shared exclusive-process GPUs: another user may hold the device between
                # runs. Only that start-up failure is retried; nothing is timed concurrently.
                for attempt in range(BUSY_RETRIES + 1):
                    with out_path.open("w") as stdout, log_path.open("w") as stderr:
                        try:
                            process = subprocess.run(argv, cwd=ROOT, stdout=stdout, stderr=stderr,
                                                     timeout=timeout, check=False)
                        except subprocess.TimeoutExpired:
                            record["exit_code"] = 124
                            raise RuntimeError(f"{label}: timed out; see {log_path}")
                    busy = "busy or unavailable" in out_path.read_text() + log_path.read_text()
                    if not (process.returncode and busy) or attempt == BUSY_RETRIES:
                        break
                    record["busy_retries"] = attempt + 1
                    time.sleep(30)
                record.update(exit_code=process.returncode, finished_at=TABLE3.now())
                if process.returncode:
                    raise RuntimeError(f"{label}: executable exited {process.returncode}; see {out_path}")
                this_identity = measured_identity(metadata, log_path.read_text())
                if identity is not None and this_identity != identity:
                    raise ValueError("CUDA device identity changed during the sweep")
                identity = this_identity
                metadata.update(gpu=identity[0], gpu_name=identity[1], compute_capability=identity[2],
                                gpu_uuid=identity[3], driver_version=identity[4])
                results = parse(out_path.read_text())
                if sorted((size, trial) for size, trial, *_ in results) != \
                        sorted((size, trial) for size in group for trial in range(trials)):
                    raise ValueError(f"{label}: expected {trials} trials for f={group}")
                for size, trial, time_ms, rel_l2, validated in results:
                    row = {"gpu": identity[0], "shape": shape, "filter": size, "library": library,
                           "precision": spec["precision"], "repeat": trial, "time_ms": time_ms,
                           "filter_fft_included": filter_fft, "rel_l2": rel_l2, "validated": validated,
                           "passed": "true", "operation": spec["operation"], "source": record["stdout"]}
                    check_row(row)
                    rows.append(row)
                TABLE3.write_csv(output / "data.csv", rows, FIELDS)
                if cut and diverged(rows, library, group[0], slow_ratio):
                    truncated[library] = {"last_measured_filter": group[0],
                                          "rule": f"median > {slow_ratio:g}x the fastest other library"}
                    print(f"{spec['title']}: {library} is over {slow_ratio:g}x slower at f={group[0]}; "
                          "larger filters are not measured", flush=True)
                    # Exclude a newly skipped tail; skipped work is never counted done.
                    omitted = len(selected) - selected.index(group[0]) - 1
                    total -= omitted
                    detail += f"; {omitted} larger filters skipped"
                TABLE3.atomic_json(output / "metadata.json", metadata)
                completed += len(group)
                TABLE3.report_progress(completed, total, unit="cases", detail=detail)
        backend_rows, metadata["backend_sha256"], metadata["backend_imports"] = snapshot_backends(
            backends, output, spec, identity[0], shape, filters, tuple(r[0] for r in runners))
        rows += backend_rows
        TABLE3.write_csv(output / "data.csv", rows, FIELDS)
        write_summary(output / "summary.csv", rows, spec)
        metadata["libraries"] = sorted({row["library"] for row in rows})
        measured_cells = {(row["library"], int(row["filter"])) for row in rows}
        metadata["coverage_complete"] = all((library, size) in measured_cells
            for library in spec["libraries"] for size in spec["filters"][profile])
        metadata.update(status="completed", numerical_and_timing_gates_passed=True)
    except (OSError, ValueError, RuntimeError) as error:
        metadata.update(status="failed", error=str(error), numerical_and_timing_gates_passed=False)
        print(f"{spec['title']} failed: {error}", file=sys.stderr)
        return 1
    finally:
        metadata["finished_at"] = TABLE3.now()
        TABLE3.atomic_json(output / "metadata.json", metadata)
    print(f"Completed {spec['title']}: {len(filters)} filters, libraries={metadata['libraries']}. Results: {output}")
    if policy_skipped:
        print(f"Fast mode skipped {len(policy_skipped)} cuDNN filters; use --full to measure every filter.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--cudnn-binary", type=Path, help="thff_conv1d_compare; omit to leave cuDNN pending")
    parser.add_argument("--flash-binary", type=Path, help="native pinned FlashFFTConv 4096 driver")
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"), required=True)
    args = parser.parse_args()
    runners = [("ThunderFFT", args.binary, figure5_command, figure5_parse, "true")]
    if args.cudnn_binary:
        # Direct convolution: there is no filter transform to include.
        runners.append(("cuDNN", args.cudnn_binary, figure5_cudnn_command, figure5_parse, "n/a"))
    if args.flash_binary:
        runners.append(("FlashFFTConv", args.flash_binary, figure5_flash_command, figure5_parse, "true"))
    sources = [*ROOT.glob("include/**/*.cuh"), ROOT / "sample/utils.h", ROOT / "sample/conv1d/conv1d_fullfft_l4096_thunder.cu",
               ROOT / "sample/conv1d/conv1d_compare_cudnn_thunder.cu", *HERE.glob("*.py"), HERE / "flashfftconv.cu", HERE / "run.sh"]
    try:
        return collect(FIGURE5, runners, args.build_dir, args.output, args.profile, sources)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
