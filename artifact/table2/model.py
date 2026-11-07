#!/usr/bin/env python3
"""Generate Table 2: calculated model rows plus the measured cuFFTDx row.

The calculated rows (ideal CUDA Core FFT, Tensor Core radix-4/8/16) come from
assumptions.json only. The measured cuFFTDx row is derived from a completed
Table 3 run (the runner dependency, or --source-dir): the fastest validated
cuFFTDx candidate for the configuration in assumptions.json
["measured_reference"]. Its relative time is t_measured / t_ideal with
t_ideal = 5 N log2 N batch / CUDA-Core peak, and utilization is the inverse.
Without a Table 3 source that measured cuFFTDx, the measured row is omitted
and the outputs say so.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import sys


HERE = Path(__file__).resolve().parent
FIELDS = ("gpu", "precision", "origin", "engine", "design", "mma_shape", "peak_tflops",
          "arithmetic_coefficient", "shared_bytes_per_mma", "mma_issue_cycles",
          "required_shared_bytes_per_cycle", "utilization_ceiling", "relative_time",
          "N", "batch", "measured_time_us", "ideal_time_us", "candidate")
SOURCE_REQUIRED = {"gpu", "N", "batch", "placement", "library", "precision", "trial",
                   "time_us", "rel_l2", "passed", "inside_repeats", "candidate"}


class Pending(Exception):
    pass


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def calculate(config):
    for key in ("cuda_peak_tflops", "tensor_peak_tflops", "shared_bytes_per_sm_cycle",
                "tensor_ops_per_sm_cycle", "mma_m", "component_bytes"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    ideal_cost = 5.0 / config["cuda_peak_tflops"]
    rows = [{"gpu": config["gpu"], "precision": config["precision"],
             "origin": "calculated", "engine": "CUDA Core", "design": "Ideal FFT",
             "mma_shape": "", "peak_tflops": config["cuda_peak_tflops"],
             "arithmetic_coefficient": 5.0, "shared_bytes_per_mma": "",
             "mma_issue_cycles": "", "required_shared_bytes_per_cycle": "",
             "utilization_ceiling": 1.0, "relative_time": 1.0}]
    for radix in config["radices"]:
        if radix not in (4, 8, 16):
            raise ValueError("This paper model defines radices 4, 8 and 16 only")
        m, n, k = config["mma_m"], 2 * radix, 2 * radix
        ops = 2 * m * n * k
        issue_cycles = ops / config["tensor_ops_per_sm_cycle"]
        traffic = (m * k + m * n) * config["component_bytes"]
        required_bw = traffic / issue_cycles
        utilization = min(1.0, config["shared_bytes_per_sm_cycle"] / required_bw)
        coefficient = 8.0 * radix / math.log2(radix)
        rows.append({"gpu": config["gpu"], "precision": config["precision"],
                     "origin": "calculated", "engine": "Tensor Core",
                     "design": f"Radix-{radix}", "mma_shape": f"m{m}n{n}k{k}",
                     "peak_tflops": config["tensor_peak_tflops"],
                     "arithmetic_coefficient": coefficient,
                     "shared_bytes_per_mma": traffic, "mma_issue_cycles": issue_cycles,
                     "required_shared_bytes_per_cycle": required_bw,
                     "utilization_ceiling": utilization,
                     "relative_time": coefficient / (config["tensor_peak_tflops"] * utilization) / ideal_cost})
    return rows


def measured_row(config, time_us, batch, candidate):
    """Table 2 measured row from a cuFFTDx compute time for the whole batch."""
    reference = config["measured_reference"]
    n = reference["N"]
    if not math.isfinite(time_us) or time_us <= 0 or batch <= 0:
        raise ValueError("measured time and batch must be finite and positive")
    ideal_us = 5.0 * n * math.log2(n) * batch / (config["cuda_peak_tflops"] * 1e12) * 1e6
    relative = time_us / ideal_us
    return {"gpu": reference["gpu"], "precision": reference["precision"].upper(),
            "origin": "measured", "engine": "CUDA Core", "design": "cuFFTDx",
            "mma_shape": "", "peak_tflops": config["cuda_peak_tflops"],
            "arithmetic_coefficient": "", "shared_bytes_per_mma": "", "mma_issue_cycles": "",
            "required_shared_bytes_per_cycle": "", "utilization_ceiling": 1.0 / relative,
            "relative_time": relative, "N": n, "batch": batch, "measured_time_us": time_us,
            "ideal_time_us": ideal_us, "candidate": candidate}


def source_directory(explicit):
    if explicit:
        return explicit.resolve()
    dependencies = json.loads(os.environ.get("THUNDERFFT_DEPENDENCY_RESULTS", "{}"))
    if not isinstance(dependencies, dict):
        raise ValueError("THUNDERFFT_DEPENDENCY_RESULTS must be a JSON object")
    source = dependencies.get("table3")
    if source is None:
        return None
    if not isinstance(source, str) or not source:
        raise ValueError("Table 3 dependency must be a result-directory path")
    return Path(source).resolve()


def read_measurement(directory, config, profile):
    """Return (time_us, batch, candidate, provenance), or raise Pending when the
    Table 3 source did not measure the reference cuFFTDx configuration."""
    reference = config["measured_reference"]
    status_path, data_path = directory / "status.json", directory / "data.csv"
    if not status_path.is_file() or not data_path.is_file():
        raise ValueError(f"Table 3 status.json or data.csv is missing: {directory}")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("evaluation") != "table3" or status.get("status") != "completed" \
            or status.get("exit_code") != 0:
        raise ValueError("source must be a completed Table 3 evaluation with exit_code 0")
    if status.get("profile") != profile:
        raise ValueError("Table 3 source profile does not match requested Table 2 profile")
    with data_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    if not any(row.get("library") == "cuFFTDx" for row in rows):
        raise Pending("Table 3 source has no cuFFTDx measurements (run it with MATHDX_ROOT)")
    missing = SOURCE_REQUIRED - fields
    if missing:
        raise ValueError("missing Table 3 fields: " + ", ".join(sorted(missing)))
    matches = [row for row in rows if row["library"] == "cuFFTDx" and row["gpu"] == reference["gpu"]
               and int(row["N"]) == reference["N"] and row["precision"] == reference["precision"]
               and row["placement"] == reference["placement"]]
    if not matches:
        raise Pending(f"Table 3 source has no {reference['gpu']} cuFFTDx N={reference['N']} "
                      f"{reference['precision']} {reference['placement']} rows")
    candidates, batches, repeats = {}, set(), set()
    for row in matches:
        if row["passed"].strip().lower() not in ("true", "1"):
            raise ValueError(f"unvalidated cuFFTDx source row: {row['candidate']} trial {row['trial']}")
        time_us, rel_l2 = float(row["time_us"]), float(row["rel_l2"])
        if not math.isfinite(time_us) or time_us <= 0:
            raise ValueError("source compute time must be finite and positive")
        if not math.isfinite(rel_l2) or not 0 <= rel_l2 <= 0.005:
            raise ValueError("source relative L2 must be finite and at most 0.005")
        if not row["candidate"]:
            raise ValueError("cuFFTDx row without a candidate label")
        trials = candidates.setdefault(row["candidate"], {})
        if row["trial"] in trials:
            raise ValueError(f"duplicate source trial: {row['candidate']} trial {row['trial']}")
        trials[row["trial"]] = time_us
        batches.add(int(row["batch"]))
        repeats.add(int(row["inside_repeats"]))
    if len(batches) != 1 or len(repeats) != 1:
        raise ValueError("cuFFTDx reference rows need one common batch and inside_repeats value")
    batch, repeat = batches.pop(), repeats.pop()
    if profile == "paper" and (batch != reference["batch"] or repeat != 1000):
        raise ValueError(f"paper Table 2 requires batch {reference['batch']} and R=1000")
    if len({tuple(sorted(trials)) for trials in candidates.values()}) != 1:
        raise ValueError("cuFFTDx candidates have different trial sets")
    # Table 3's selection rule: smallest median compute time; ties keep source order.
    best = min(candidates, key=lambda name: statistics.median(candidates[name].values()))
    time_us = statistics.median(candidates[best].values())
    check_table3_selection(directory, reference, best)
    provenance = {"source_run": directory.name, "source_result_dir": str(directory),
                  "source_file": str(data_path), "source_sha256": sha256(data_path),
                  "source_status_sha256": sha256(status_path), "inside_repeats": repeat,
                  "trials": len(candidates[best]), "candidates_compared": len(candidates),
                  "trial_time_us": sorted(candidates[best].values())}
    if (directory / "metadata.json").is_file():
        provenance["source_metadata_sha256"] = sha256(directory / "metadata.json")
        source_metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        provenance["cufftdx_build"] = source_metadata.get("cufftdx_build")
        provenance["gpu_name"] = source_metadata.get("gpu_name")
    return time_us, batch, best, provenance


def check_table3_selection(directory, reference, best):
    """When Table 3 wrote its candidate table, its choice must agree."""
    path = directory / "cufftdx_candidates.csv"
    if not path.is_file():
        return
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["gpu"], int(row["N"]), row["precision"], row["placement"], row["selected"]) == \
                    (reference["gpu"], reference["N"], reference["precision"], reference["placement"], "true") \
                    and row["candidate"] != best:
                raise ValueError("cuFFTDx selection disagrees with Table 3 cufftdx_candidates.csv")


def snapshot_source(output, directory, provenance):
    """Keep the native source with a derived result so downloads are portable."""
    directory = directory.resolve()
    relative = Path("source/table3")
    snapshot = output / relative
    snapshot.mkdir(parents=True, exist_ok=True)
    names = {name for name in ("data.csv", "status.json", "metadata.json", "cufftdx_candidates.csv")
             if (directory / name).is_file()}
    if (directory / "metadata.json").is_file():
        metadata = json.loads((directory / "metadata.json").read_text())
        for command in metadata.get("commands", []):
            names.update(command[key] for key in ("stdout", "stderr")
                         if isinstance(command.get(key), str) and (directory / command[key]).is_file())
    hashes = {}
    for name in sorted(names):
        source = (directory / name).resolve()
        target = snapshot / name
        if not source.is_relative_to(directory) or not target.resolve().is_relative_to(snapshot.resolve()):
            raise ValueError("Table 3 snapshot evidence path escapes its result directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target.resolve():
            shutil.copy2(source, target)
        hashes[name] = sha256(target)
    return {**provenance, "original_source_result_dir": str(directory),
            "source_result_dir": relative.as_posix(), "source_file": (relative / "data.csv").as_posix(),
            "snapshot_sha256": hashes}


def write_outputs(output, config, inputs, rows, measured, provenance, profile, pending):
    for name in ("data.csv", "table.csv"):
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, restval="")
            writer.writeheader()
            writer.writerows(rows)
    lines = ["# Table 2: A100 FP16 model" + ("" if measured else " (calculated rows only)"), "",
             "| Design | Origin | Arithmetic coefficient | Utilization | Relative time |",
             "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        coefficient = row["arithmetic_coefficient"]
        coefficient = f"{coefficient:.2f}" if coefficient != "" else "—"
        lines.append(f"| {row['design']} | {row['origin']} | {coefficient} | "
                     f"{100 * row['utilization_ceiling']:.1f}% | {row['relative_time']:.2f} |")
    reference = config["measured_reference"]
    if measured:
        lines += ["", f"cuFFTDx: {reference['gpu']} {reference['precision'].upper()} {reference['placement']} "
                      f"N={reference['N']}, batch={measured['batch']:,}, candidate `{measured['candidate']}`; "
                      f"median compute {measured['measured_time_us']:.3f} us vs ideal "
                      f"{measured['ideal_time_us']:.3f} us (Table 3 run {provenance['source_run']}).",
                  f"Paper: relative time {reference['paper_relative_time']}, "
                  f"utilization {100 * reference['paper_utilization']:.1f}%."]
        if measured["batch"] != reference["batch"]:
            lines.append(f"Warning: batch {measured['batch']:,} is not the paper's {reference['batch']:,}; "
                         "a small batch does not fill the GPU, so this row is not comparable to the paper.")
    else:
        lines += ["", f"The measured cuFFTDx row is excluded: {pending}"]
    lines += ["Calculated rows use the supplied paper's model assumptions, not GPU measurements.",
              "The radix-8 calculation retains 64/3 rather than rounding it to 21.3."]
    (output / "table.md").write_text("\n".join(lines) + "\n")
    metadata = {"schema_version": 2, "evaluation": "table2", "profile": profile,
                "scope": "calculated rows and measured cuFFTDx row" if measured else "calculated rows only",
                "full_paper_comparison": bool(measured) and profile == "paper",
                "excluded": [] if measured else ["measured cuFFTDx row"],
                "excluded_reason": None if measured else pending,
                "assumptions": config, "assumptions_sha256": sha256(inputs),
                "generator_sha256": sha256(Path(__file__)),
                "formulae": {"arithmetic": "8*r/log2(r) times N*log2(N)",
                             "mma_ops": "2*m*n*k; n=k=2*r",
                             "shared_traffic": "(m*k+m*n)*component_bytes; B remains in registers",
                             "utilization": "min(1, available_shared_B/cycle / required_shared_B/cycle)",
                             "relative_time": "(coefficient/(peak_TFLOPS*utilization))/(5/cuda_peak_TFLOPS)",
                             "measured_ideal_time": "5*N*log2(N)*batch / (cuda_peak_TFLOPS*1e12) s",
                             "measured_relative_time": "Table 3 median compute_us / measured_ideal_time",
                             "measured_utilization": "1 / measured_relative_time"},
                "measured_source": provenance,
                "measured_vs_paper": ({"relative_time": measured["relative_time"],
                                       "paper_relative_time": reference["paper_relative_time"],
                                       "utilization": measured["utilization_ceiling"],
                                       "paper_utilization": reference["paper_utilization"]}
                                      if measured else None),
                "scientific_validation": ("calculated rows from declared paper assumptions; measured row "
                                          "from validated Table 3 cuFFTDx timing" if measured
                                          else "calculated from declared paper assumptions")}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=("smoke", "paper"),
                        default=os.environ.get("THUNDERFFT_PROFILE", "smoke"))
    parser.add_argument("--source-dir", type=Path,
                        help="completed Table 3 result directory (default: runner dependency)")
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        inputs = HERE / "assumptions.json"
        config = json.loads(inputs.read_text())
        rows = calculate(config)
        measured, provenance, pending = None, None, None
        directory = source_directory(args.source_dir)
        if directory is None:
            pending = "no Table 3 source (use ./run.sh table2 or --source-dir)"
        else:
            try:
                time_us, batch, candidate, provenance = read_measurement(directory, config, args.profile)
                measured = measured_row(config, time_us, batch, candidate)
                rows.insert(1, measured)
                provenance = snapshot_source(args.output, directory, provenance)
            except Pending as error:
                pending = str(error)
        write_outputs(args.output, config, inputs, rows, measured, provenance, args.profile, pending)
        if measured:
            print(f"Table 2: {len(rows)} rows; cuFFTDx relative time {measured['relative_time']:.2f}, "
                  f"utilization {100 * measured['utilization_ceiling']:.1f}% "
                  f"(Table 3 run {provenance['source_run']}). Outputs: {args.output}")
        else:
            print(f"Calculated {len(rows)} model rows: {args.output / 'table.csv'}; "
                  f"cuFFTDx measurement excluded: {pending}")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"FAILED [table2]: {error}", file=sys.stderr)
        return 65


if __name__ == "__main__":
    sys.exit(main())
