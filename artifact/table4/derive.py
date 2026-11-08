#!/usr/bin/env python3
"""Derive Table 4 unit throughput from completed Table 3 timing bundles.

This is a CPU-only conversion that can also join independently measured ideal
MMA bundles. Paper reference numbers never populate measurement columns.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
import sys


GPUS = ("A100", "H100", "B200")
PAPER = {"A100": (156.0, 147.22, 113.42),
         "H100": (378.0, 243.82, 156.67),
         "B200": (1100.0, 275.80, 202.34)}
PAPER_SHA256 = "b1e34fdfc407292093756bbabf57b0dae51b12c8fb88789123c2f3fccc100d50"
FLOPS_PER_FFT = 3 * 16 * (2 * 16 * 8 * 8) // 16
SOURCE_FILES = ("data.csv", "metadata.json", "status.json")
SOURCE_FIELDS = ("gpu", "N", "batch", "placement", "library", "precision", "trial",
                 "time_us", "e2e_out_of_place_us", "rel_l2", "passed", "inside_repeats",
                 "t_R_us", "t_2R_us", "shared_bytes", "raw_csv")
SOURCE_EXTRA_FIELDS = ("candidate", "ffts_per_block", "elements_per_thread", "threads_per_block")
REQUIRED_SOURCE_FIELDS = set(SOURCE_FIELDS) - {"e2e_out_of_place_us", "shared_bytes", "raw_csv"}
DATA_FIELDS = ("gpu", "N", "batch", "precision", "placement", "library", "trial",
               "inside_repeats", "time_us", "t_R_us", "t_2R_us", "rel_l2", "passed",
               "tensor_flops_per_fft", "tensor_flops_per_batch", "unit_tflops",
               "source_snapshot", "source_data_sha256")
TABLE_FIELDS = ("gpu", "status", "N", "precision", "placement", "library", "batch",
                "inside_repeats", "trials", "median_time_us", "max_rel_l2",
                "tensor_flops_per_fft", "tensor_flops_per_batch", "unit_tflops",
                "hardware_peak_tflops", "hardware_peak_source", "ideal_mma_tflops",
                "efficiency_pct", "ideal_mma_status", "paper_ideal_mma_tflops",
                "paper_unit_tflops", "source_snapshot", "unit_gpu_uuid", "ideal_gpu_uuid",
                "ideal_selected_candidate", "ideal_trials", "ideal_source_snapshot", "efficiency_identity")
IDEAL_FIELDS = ("gpu", "gpu_uuid", "candidate", "trial", "blocks", "threads", "chains", "iterations",
                "unroll", "mma_instructions", "flop_count", "time_us", "tflops", "max_abs_error",
                "passed", "registers", "shared_bytes", "active_blocks_per_sm")
IDEAL_CANDIDATES = ("t128-c4", "t128-c8", "t256-c4", "t256-c8")
UUID_PATTERN = r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
SCHEMA = "table4-from-table3-v1"
LIMITATION = ("Reuses Table 3 differential register timing. Correctness was checked separately "
              "on a single forward FFT against cuFFT FP64; repeated timed FFT values may "
              "become nonfinite. This conversion performs no additional correctness check "
              "or GPU measurement.")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows, fields):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def evidence_file(directory, relative):
    """Accept only contained relative evidence paths, including after symlinks."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("invalid or escaping evidence path")
    root = Path(directory).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"{root}: missing or escaping evidence path {relative!r}")
    return path


def source_rows(path):
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = tuple(reader.fieldnames or ())
        if len(set(fields)) != len(fields) or not REQUIRED_SOURCE_FIELDS.issubset(fields):
            raise ValueError(f"{path}: unexpected Table 3 CSV columns")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path}: malformed Table 3 CSV row")
    return rows


def positive_int(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    try:
        number = int(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if number <= 0 or (not isinstance(value, (str, int))) or str(number) != str(value):
        raise ValueError(f"{name} must be a positive integer")
    return number


def finite_number(value, name, allow_zero=False):
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def nonnegative_int(value, name):
    if type(value) in (int, str) and value in (0, "0"):
        return 0
    return positive_int(value, name)


def valid_hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def unit_gpu_uuid(metadata):
    """Use native UUID when present, or an unambiguous historical inventory."""
    explicit = metadata.get("gpu_uuid")
    if explicit is not None:
        if not isinstance(explicit, str) or not re.fullmatch(UUID_PATTERN, explicit):
            raise ValueError("invalid Table 3 GPU UUID")
        return "GPU-" + explicit[4:].lower()
    inventory = metadata.get("gpu_inventory_before", {})
    if not isinstance(inventory, dict) or inventory.get("exit_code") != 0:
        return None
    name = metadata.get("gpu_name")
    candidates = []
    for row in csv.reader(inventory.get("stdout", "").splitlines(), skipinitialspace=True):
        if len(row) >= 3 and row[2].strip() == name and re.fullmatch(UUID_PATTERN, row[1].strip()):
            candidates.append("GPU-" + row[1].strip()[4:].lower())
    mask = metadata.get("effective_CUDA_VISIBLE_DEVICES", "")
    if isinstance(mask, str) and re.fullmatch(UUID_PATTERN, mask):
        candidates = [candidate for candidate in candidates if candidate.lower() == mask.lower()]
    return candidates[0] if len(candidates) == 1 else None


def load_ideal(directory, profile):
    """Validate a completed MMA measurement, including retained evidence hashes."""
    directory = Path(directory).resolve()
    metadata_path = evidence_file(directory, "ideal_metadata.json")
    metadata = read_json(metadata_path)
    if (metadata.get("schema_version"), metadata.get("evaluation"), metadata.get("profile"),
            metadata.get("status")) != (1, "table4_ideal", profile, "completed"):
        raise ValueError("ideal MMA metadata must be completed and match the requested profile")
    gpu, uuid = metadata.get("gpu"), metadata.get("gpu_uuid")
    if gpu not in GPUS or not isinstance(metadata.get("gpu_name"), str) or not metadata["gpu_name"].strip():
        raise ValueError("invalid ideal MMA GPU identity")
    if not isinstance(uuid, str) or not re.fullmatch(UUID_PATTERN, uuid):
        raise ValueError("invalid ideal MMA GPU UUID")
    if not isinstance(metadata.get("compute_capability"), str) or not re.fullmatch(r"\d+\.\d+", metadata["compute_capability"]):
        raise ValueError("invalid ideal MMA compute capability")
    source_hashes = metadata.get("source_sha256")
    if (not isinstance(source_hashes, dict) or not source_hashes or not all(isinstance(name, str) and name
            and valid_hash(value) for name, value in source_hashes.items()) or not valid_hash(metadata.get("binary_sha256"))):
        raise ValueError("ideal MMA source and binary hashes are required")
    outputs = metadata.get("outputs")
    if not isinstance(outputs, dict) or not outputs or not all(isinstance(name, str) and name and
            not Path(name).is_absolute() and ".." not in Path(name).parts and str(Path(name)) == name
            and valid_hash(value) for name, value in outputs.items()):
        raise ValueError("invalid ideal MMA evidence hashes")
    if "ideal.csv" not in outputs or "disassembly.sass" not in outputs or "ideal_metadata.json" in outputs:
        raise ValueError("ideal MMA evidence must include ideal.csv and disassembly.sass")
    if not all(any(name.startswith("raw/") and name.endswith(extension) for name in outputs)
               for extension in (".csv", ".log")) or not any(name.startswith("build") for name in outputs):
        raise ValueError("ideal MMA raw CSV/log and build evidence are required")
    paths = {name: evidence_file(directory, name) for name in outputs}
    for name, path in paths.items():
        if sha256(path) != outputs[name]:
            raise ValueError(f"ideal MMA evidence {name} hash mismatch")
    settings = metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("invalid ideal MMA settings")
    iterations, trials, waves = (positive_int(settings.get(key), f"ideal settings.{key}")
                                  for key in ("iterations", "trials", "waves"))
    nonnegative_int(settings.get("warmups"), "ideal settings.warmups")
    candidates = settings.get("candidates")
    if (not isinstance(candidates, list) or len(candidates) != 4 or not all(isinstance(candidate, str)
            for candidate in candidates) or set(candidates) != set(IDEAL_CANDIDATES)):
        raise ValueError("ideal MMA must collect all four declared candidates")
    with paths["ideal.csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != IDEAL_FIELDS:
            raise ValueError("unexpected ideal MMA CSV columns")
        raw_rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in raw_rows):
        raise ValueError("malformed ideal MMA CSV row")
    groups, configurations = {}, {}
    for raw in raw_rows:
        candidate = raw["candidate"]
        if raw["gpu"] != gpu or raw["gpu_uuid"].lower() != uuid.lower() or candidate not in IDEAL_CANDIDATES:
            raise ValueError("ideal MMA CSV GPU/candidate mismatch")
        if raw["passed"] != "true" or finite_number(raw["max_abs_error"], "max_abs_error", allow_zero=True) != 0:
            raise ValueError("ideal MMA exact correctness gate failed")
        trial = nonnegative_int(raw["trial"], "ideal trial")
        values = {key: positive_int(raw[key], f"ideal {key}") for key in (
            "blocks", "threads", "chains", "iterations", "unroll", "mma_instructions", "flop_count",
            "registers", "active_blocks_per_sm")}
        values["shared_bytes"] = nonnegative_int(raw["shared_bytes"], "ideal shared_bytes")
        threads, chains = (int(number) for number in re.fullmatch(r"t(\d+)-c(\d+)", candidate).groups())
        if (values["threads"], values["chains"], values["iterations"]) != (threads, chains, iterations):
            raise ValueError("ideal MMA CSV/settings mismatch")
        if values["unroll"] != 32:
            raise ValueError("ideal MMA version 1 requires unroll=32")
        if "unroll" in settings and values["unroll"] != positive_int(settings["unroll"], "ideal settings.unroll"):
            raise ValueError("ideal MMA unroll setting mismatch")
        sm_count = metadata.get("sm_count", metadata.get("multiprocessor_count"))
        if sm_count is not None:
            sms = positive_int(sm_count, "sm_count")
            if values["blocks"] != sms * waves * values["active_blocks_per_sm"]:
                raise ValueError("ideal MMA block count disagrees with occupancy/waves")
        instructions = values["blocks"] * (threads // 32) * chains * iterations * values["unroll"]
        if values["mma_instructions"] != instructions or values["flop_count"] != instructions * 2048:
            raise ValueError("ideal MMA instruction/FLOP count mismatch")
        time_us = finite_number(raw["time_us"], "ideal time_us")
        throughput = finite_number(raw["tflops"], "ideal tflops")
        if not math.isclose(throughput, values["flop_count"] / time_us / 1e6, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError("ideal MMA throughput formula mismatch")
        configuration = tuple(values[key] for key in sorted(values))
        if candidate in configurations and configurations[candidate] != configuration:
            raise ValueError("ideal MMA configuration changes between trials")
        configurations[candidate] = configuration
        rows = groups.setdefault(candidate, {})
        if trial >= trials or trial in rows:
            raise ValueError("duplicate or out-of-range ideal MMA trial")
        rows[trial] = {**raw, **values, "trial": trial, "time_us": time_us, "tflops": throughput,
                       "max_abs_error": 0.0}
    if set(groups) != set(IDEAL_CANDIDATES) or any(len(rows) != trials for rows in groups.values()):
        raise ValueError("incomplete ideal MMA candidate/trial coverage")
    medians = {candidate: statistics.median(row["tflops"] for row in rows.values()) for candidate, rows in groups.items()}
    selected = min(medians, key=lambda candidate: (-medians[candidate], candidate))
    if metadata.get("selected_candidate") != selected:
        raise ValueError("ideal MMA selected candidate disagrees with highest median throughput")
    return {"directory": directory, "gpu": gpu, "gpu_uuid": "GPU-" + uuid[4:].lower(), "selected_candidate": selected,
            "tflops": medians[selected], "trials": trials,
            "rows": [groups[candidate][trial] for candidate in sorted(groups) for trial in sorted(groups[candidate])],
            "sha256": {**outputs, "ideal_metadata.json": sha256(metadata_path)}}


def load_source(directory, profile):
    directory = Path(directory).resolve()
    data_path = evidence_file(directory, "data.csv")
    selected = []
    for row in source_rows(data_path):
        if row["library"] != "ThunderFFT":
            continue
        n = positive_int(row["N"], "N")
        if (n, row["precision"], row["placement"]) == (64, "tf32", "register"):
            selected.append(row)
    # Other Table 3 configurations, including incomplete unrelated runs, are
    # not dependencies of this conversion.
    if not selected:
        return None
    paths = {name: evidence_file(directory, name) for name in SOURCE_FILES}
    metadata, status = read_json(paths["metadata.json"]), read_json(paths["status.json"])
    for name, record in (("metadata", metadata), ("status", status)):
        if record.get("evaluation") != "table3" or record.get("profile") != profile:
            raise ValueError(f"{directory}: Table 3 {name} evaluation/profile mismatch")
        if record.get("status") != "completed":
            raise ValueError(f"{directory}: requires a completed Table 3 {name}")
    if type(status.get("exit_code")) is not int or status["exit_code"] != 0:
        raise ValueError(f"{directory}: Table 3 runner exit_code must be zero")
    settings = metadata.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError("invalid Table 3 settings")
    batch, repeats, trials = (positive_int(settings.get(key), f"settings.{key}")
                              for key in ("batch", "repeats", "trials"))
    if batch % 16:
        raise ValueError("Table 3 batch must be a multiple of 16")
    if profile == "paper" and (batch != 65536 or repeats != 1000):
        raise ValueError("paper conversion requires batch 65536 and R=1000")
    gpu = metadata.get("gpu")
    if gpu not in GPUS:
        raise ValueError("Table 3 source must identify A100, H100, or B200")
    parsed, seen = [], set()
    for row in selected:
        if row["gpu"] != gpu:
            raise ValueError("Table 3 source GPU mismatch")
        if row["passed"] != "true":
            raise ValueError("Table 3 correctness gate did not pass")
        if row.get("candidate", ""):
            raise ValueError("unexpected ThunderFFT candidate")
        if (positive_int(row["batch"], "batch"), positive_int(row["inside_repeats"], "inside_repeats")) != (batch, repeats):
            raise ValueError("Table 3 CSV/settings mismatch")
        try:
            trial = int(row["trial"])
        except ValueError as error:
            raise ValueError("invalid Table 3 trial") from error
        if str(trial) != row["trial"] or not 0 <= trial < trials or trial in seen:
            raise ValueError("duplicate or out-of-range Table 3 trial")
        seen.add(trial)
        time_us, t_r, t_2r = (finite_number(row[key], key) for key in ("time_us", "t_R_us", "t_2R_us"))
        if t_2r <= t_r or not math.isclose(time_us, (t_2r - t_r) / repeats, rel_tol=1e-4, abs_tol=1e-6):
            raise ValueError("Table 3 differential timing is inconsistent")
        error = finite_number(row["rel_l2"], "rel_l2", allow_zero=True)
        if error > 0.005:
            raise ValueError("Table 3 relative L2 gate failed")
        total = FLOPS_PER_FFT * batch
        throughput = total / time_us / 1e6
        if not math.isfinite(throughput):
            raise ValueError("derived throughput is nonfinite")
        parsed.append(dict(gpu=gpu, N=64, batch=batch, precision="tf32", placement="register",
                           library="ThunderFFT", trial=trial, inside_repeats=repeats, time_us=time_us,
                           t_R_us=t_r, t_2R_us=t_2r, rel_l2=error, passed="true",
                           tensor_flops_per_fft=FLOPS_PER_FFT, tensor_flops_per_batch=total,
                           unit_tflops=throughput))
    if len(seen) != trials:
        raise ValueError(f"incomplete Table 3 trials: expected {trials}, got {len(seen)}")
    hashes = {name: sha256(path) for name, path in paths.items()}
    # The runner snapshots Table 3 into its collection before adding the
    # collection_part location to the live status. Table 4 then snapshots that
    # annotated status from the same run. This location is not a measurement
    # identity: accept either copy while retaining and verifying its exact bytes.
    # All other status fields, and all data/metadata bytes, must still agree.
    identity = {"data_sha256": hashes["data.csv"], "metadata_sha256": hashes["metadata.json"],
                "status": {key: value for key, value in status.items() if key != "collection_part"}}
    return {"directory": directory, "rows": sorted(parsed, key=lambda row: row["trial"]),
            "gpu_uuid": unit_gpu_uuid(metadata), "sha256": hashes, "identity": identity}


def preload(source):
    source["contents"] = {name: evidence_file(source["directory"], name).read_bytes() for name in source["sha256"]}
    for name, content in source["contents"].items():
        if hashlib.sha256(content).hexdigest() != source["sha256"][name]:
            raise ValueError("source evidence changed during conversion")


def snapshot_source(output, source, relative):
    snapshot = output / relative
    if not snapshot.resolve().is_relative_to(output):
        raise ValueError("escaping output snapshot directory")
    snapshot.mkdir(parents=True, exist_ok=True)
    for name, content in source["contents"].items():
        target = snapshot / name
        if target.is_symlink() or not target.resolve().is_relative_to(snapshot.resolve()):
            raise ValueError("escaping or symlinked snapshot file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    source["snapshot"] = relative
    return {"original_source": str(source["directory"]), "snapshot": relative,
            "sha256": source["sha256"], "gpu": source.get("gpu", source["rows"][0]["gpu"]),
            "gpu_uuid": source.get("gpu_uuid")}


def generate(source_dirs, output, profile, ideal_dirs=()):
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported Table 4 profile")
    output = Path(output).resolve()
    sources, unused, by_gpu, ideals = [], [], {}, {}
    for directory in sorted({Path(path).expanduser().resolve() for path in source_dirs}):
        if directory == output:
            raise ValueError("output directory must not overwrite a Table 3 source")
        source = load_source(directory, profile)
        if source is None:
            unused.append({"path": str(directory), "reason": "no ThunderFFT N64 TF32 register rows"})
            continue
        gpu = source["rows"][0]["gpu"]
        if gpu in by_gpu:
            if source["identity"] == by_gpu[gpu]["identity"]:
                unused.append({"path": str(directory), "reason": "identical Table 3 measurement already selected"})
                continue
            raise ValueError(f"conflicting {gpu} Table 3 sources; retain one source per GPU")
        by_gpu[gpu] = source
        sources.append(source)
    for directory in sorted({Path(path).expanduser().resolve() for path in ideal_dirs}):
        if directory == output:
            raise ValueError("output directory must not overwrite an ideal MMA source")
        source = load_ideal(directory, profile)
        gpu = source["gpu"]
        if gpu in ideals:
            if source["sha256"] == ideals[gpu]["sha256"]:
                unused.append({"path": str(directory), "reason": "identical ideal MMA evidence already selected"})
                continue
            raise ValueError(f"conflicting {gpu} ideal MMA sources; retain one source per GPU")
        if gpu in by_gpu and by_gpu[gpu]["gpu_uuid"] and by_gpu[gpu]["gpu_uuid"] != source["gpu_uuid"]:
            raise ValueError(f"{gpu}: Table 3 and ideal MMA GPU UUID mismatch; use measurements from the same GPU")
        ideals[gpu] = source
    # Read all inputs before writing, allowing a bundle to be assembled in place.
    for source in [*sources, *ideals.values()]:
        preload(source)
    output.mkdir(parents=True, exist_ok=True)
    data, provenance, ideal_provenance, ideal_data = [], [], [], []
    for index, source in enumerate(sources):
        relative = f"inputs/table3-{index:03d}"
        provenance.append(snapshot_source(output, source, relative))
        for row in source["rows"]:
            data.append({**row, "source_snapshot": relative, "source_data_sha256": source["sha256"]["data.csv"]})
    for index, gpu in enumerate(sorted(ideals)):
        source = ideals[gpu]
        relative = f"inputs/ideal-{index:03d}"
        ideal_provenance.append(snapshot_source(output, source, relative))
        ideal_data.extend(source["rows"])
    table, missing = [], []
    matched = 0
    for gpu in GPUS:
        hardware, ideal_reference, unit_reference = PAPER[gpu]
        row = dict(gpu=gpu, status="pending", N=64, precision="tf32", placement="register",
                   library="ThunderFFT", hardware_peak_tflops=hardware,
                   hardware_peak_source="paper_reference", ideal_mma_tflops="", efficiency_pct="",
                   ideal_mma_status="pending", paper_ideal_mma_tflops=ideal_reference,
                   paper_unit_tflops=unit_reference, efficiency_identity="pending")
        if gpu in by_gpu:
            source = by_gpu[gpu]
            samples = source["rows"]
            median = statistics.median(sample["time_us"] for sample in samples)
            row.update(status="derived", batch=samples[0]["batch"],
                       inside_repeats=samples[0]["inside_repeats"], trials=len(samples),
                       median_time_us=median, max_rel_l2=max(sample["rel_l2"] for sample in samples),
                       tensor_flops_per_fft=FLOPS_PER_FFT,
                       tensor_flops_per_batch=samples[0]["tensor_flops_per_batch"],
                       unit_tflops=samples[0]["tensor_flops_per_batch"] / median / 1e6,
                       source_snapshot=source["snapshot"], unit_gpu_uuid=source["gpu_uuid"] or "")
        else:
            missing.append({"item": f"{gpu}/unit_kernel", "reason": "completed Table 3 N64 TF32 register CSV not collected"})
        if gpu in ideals:
            ideal = ideals[gpu]
            row.update(ideal_mma_status="measured", ideal_mma_tflops=ideal["tflops"],
                       ideal_gpu_uuid=ideal["gpu_uuid"], ideal_selected_candidate=ideal["selected_candidate"],
                       ideal_trials=ideal["trials"], ideal_source_snapshot=ideal["snapshot"])
            if gpu not in by_gpu:
                row["status"] = "partial"
            elif by_gpu[gpu]["gpu_uuid"]:
                matched += 1
                row.update(status="complete", efficiency_pct=row["unit_tflops"] / ideal["tflops"] * 100,
                           efficiency_identity="same_gpu_uuid")
            else:
                row["efficiency_identity"] = "unverified"
                missing.append({"item": f"{gpu}/efficiency_identity", "reason": "Table 3 GPU UUID could not be verified; efficiency is not combined across unidentified devices"})
        else:
            missing.append({"item": f"{gpu}/ideal_mma", "reason": "independent ideal MMA measurement not collected; Table 3 cannot derive it"})
        table.append(row)
    data.sort(key=lambda row: (GPUS.index(row["gpu"]), row["trial"]))
    ideal_data.sort(key=lambda row: (GPUS.index(row["gpu"]), row["candidate"], row["trial"]))
    complete = matched == len(GPUS)
    state = {"status": "complete" if complete else ("partial" if data or ideals else "pending"),
             "missing": missing, "full_paper_comparison": complete and profile == "paper",
             "collected_unit_gpus": len(by_gpu), "collected_ideal_gpus": len(ideals),
             "matched_gpu_pairs": matched, "expected_unit_gpus": len(GPUS),
             "expected_ideal_gpus": len(GPUS), "complete_declared_subset": complete,
             "complete_unit_subset": len(by_gpu) == len(GPUS)}
    write_csv(output / "data.csv", data, DATA_FIELDS)
    write_csv(output / "ideal.csv", ideal_data, IDEAL_FIELDS)
    write_csv(output / "table.csv", table, TABLE_FIELDS)
    markdown = ["# Table 4: Tensor Core throughput", "",
                "| GPU | Hardware peak (paper reference) | Ideal MMA (measured) | Unit kernel (derived) | Efficiency |",
                "| --- | ---: | ---: | ---: | ---: |"]
    for row in table:
        unit = f"{row['unit_tflops']:.2f}" if "unit_tflops" in row else "pending"
        ideal = f"{row['ideal_mma_tflops']:.2f}" if row["ideal_mma_tflops"] != "" else "pending"
        efficiency = f"{row['efficiency_pct']:.1f}%" if row["efficiency_pct"] != "" else "pending"
        markdown.append(f"| {row['gpu']} | {row['hardware_peak_tflops']:.2f} | {ideal} | {unit} | {efficiency} |")
    markdown.extend(["", "Throughputs are in TFLOPS. Unit throughput is derived from existing Table 3 "
                     "N64 TF32 register timings. Ideal MMA throughput comes from separately recorded "
                     "MMA microbenchmark evidence; assembly performs no GPU measurement.", "",
                     "Efficiency requires both measurements on the same verified GPU UUID. "
                     "Clocks and measurement dates can differ and remain recorded in the source bundles. "
                     "Paper ideal/unit values are retained separately in the CSV for reference.", "", LIMITATION, ""])
    (output / "table.md").write_text("\n".join(markdown))
    metadata = {"schema_version": 1, "derivation_schema": SCHEMA, "evaluation": "table4", "profile": profile,
                **state, "sources": provenance, "ideal_sources": ideal_provenance, "unused_sources": unused,
                "operation": "CPU conversion of Table 3 CSVs and assembly of independent ideal MMA evidence; no GPU measurements performed by assembly",
                "gpu_measurements_performed": False, "limitations": [LIMITATION,
                    "Unit and ideal MMA measurements may have different clock states and dates; source metadata is retained. The paper does not specify the original ideal MMA launch parameters."],
                "formula": {"tensor_flops_per_fft": FLOPS_PER_FFT,
                            "definition": "3 stages * 16 MMA/warp/stage * (2*16*8*8) FLOPs/MMA / 16 FFTs/warp",
                            "unit_tflops": "tensor_flops_per_fft * batch / (median(time_us) * 1e6)",
                            "ideal_mma_tflops": "max candidate median(flop_count / time_us / 1e6); lexical candidate tie break",
                            "efficiency_pct": "100 * unit_tflops / ideal_mma_tflops, same verified GPU UUID",
                            "timing": "Table 3 time_us = (t_2R_us - t_R_us) / inside_repeats",
                            "count_scope": "issued Tensor Core arithmetic, not useful 5*N*log2(N) FFT operations"},
                "paper_reference": {"source": "Supplied paper Table 4, page 8", "pdf_sha256": PAPER_SHA256,
                                    "hardware_peak_kind": "TF32 dense, published chip peak as reported by paper",
                                    "ideal_mma_is_remeasured": bool(ideals)},
                "outputs": {name: sha256(output / name) for name in ("data.csv", "ideal.csv", "table.csv", "table.md")}}
    write_json(output / "metadata.json", metadata)
    return state


def default_sources():
    explicit = os.environ.get("THUNDERFFT_TABLE4_SOURCE_DIR")
    if explicit is not None:
        if not explicit.strip():
            raise ValueError("THUNDERFFT_TABLE4_SOURCE_DIR is empty")
        return [Path(explicit).expanduser()]
    dependencies = json.loads(os.environ.get("THUNDERFFT_DEPENDENCY_RESULTS", "{}"))
    if not isinstance(dependencies, dict):
        raise ValueError("THUNDERFFT_DEPENDENCY_RESULTS must be a JSON object")
    source = dependencies.get("table3")
    if source is None:
        return []
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Table 3 dependency must be one result directory")
    return [Path(source).expanduser()]


def default_ideal_sources():
    value = os.environ.get("THUNDERFFT_TABLE4_IDEAL_DIR")
    if value is None:
        return []
    if not value.strip():
        raise ValueError("THUNDERFFT_TABLE4_IDEAL_DIR is empty")
    return [Path(value).expanduser()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", type=Path)
    parser.add_argument("--ideal-dir", action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "paper"), default="smoke")
    args = parser.parse_args()
    try:
        state = generate(args.source_dir if args.source_dir is not None else default_sources(),
                         args.output_dir, args.profile,
                         ideal_dirs=args.ideal_dir if args.ideal_dir is not None else default_ideal_sources())
    except (ValueError, OSError) as error:
        print(f"Table 4 conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"Table 4: {state['collected_unit_gpus']}/3 unit-kernel rows, "
          f"{state['collected_ideal_gpus']}/3 ideal MMA rows, {state['matched_gpu_pairs']}/3 matched pairs.")
    return 0 if state["collected_unit_gpus"] or state["collected_ideal_gpus"] else 78


if __name__ == "__main__":
    raise SystemExit(main())
