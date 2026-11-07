#!/usr/bin/env python3
"""Project N256 FP16 Table 6 timings onto an N64 TF32 work model, without GPU work."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


HERE = Path(__file__).resolve().parent
TABLE6 = HERE.parent / "table6"
BATCH = 65536
SOURCE_FLOPS = 65536
TARGET_FLOPS = 12288
SOURCE_PEAK = 312.0
TARGET_PEAK = 156.0
LIBRARIES = {"tcFFT": "layer_256_0_A100", "FlashFFTConv": "conv_kernel"}
ESTIMATE_FIELDS = (
    "library", "source_evidence", "source_time_us", "source_N", "source_precision",
    "source_batch", "source_tensor_flops_per_fft", "source_peak_tflops",
    "source_achieved_tflops", "source_utilization_pct", "target_N", "target_precision",
    "target_batch", "target_model_flops_per_slot", "target_peak_tflops",
    "projected_time_us", "projected_utilization_pct", "estimation_method",
    "target_work_model", "timing_status", "accuracy_status", "full_paper_comparison",
)


def positive(value, name):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_csv(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError(f"Missing or duplicate CSV fields: {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return fields, rows


def is_a100(name):
    return isinstance(name, str) and re.search(r"\bA100\b", name) is not None


def documented_sources(path=TABLE6 / "README.md"):
    """Read the public rounded timing record; do not invent original raw trials."""
    path = Path(path)
    text = path.read_text()
    if not re.search(r"256-point FP16 forward FFT, batch 65,536, on A100", text):
        raise ValueError("Table 6 README does not declare the required N256 FP16 A100 batch65536 contract")
    sections = re.findall(r"^## A100 result \(([^\n)]+)\)\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    if len(sections) != 1:
        raise ValueError("Expected exactly one documented A100 timing result section")
    date, section = sections[0]
    sources = []
    for library in LIBRARIES:
        matches = re.findall(r"^\|\s*" + re.escape(library) + r"\s*\|\s*([0-9.eE+-]+)\s*\|", section, re.M)
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous documented {library} timing")
        sources.append({"library": library, "time_us": positive(matches[0], "documented time_ms") * 1000,
                        "evidence": "documented_rounded_timing", "source_date": date,
                        "note": "Rounded Table 6 README timing; original raw timing CSV is not supplied or revalidated."})
    return sources, [file_record(path)]


def table6_sources(path):
    """The Table 6 summary omits precision; its adjacent data.csv supplies that contract."""
    path = Path(path)
    _, rows = read_csv(path)
    data_path = path.with_name("data.csv")
    if not data_path.is_file():
        raise ValueError("Table 6 table.csv requires adjacent data.csv to confirm FP16 input and kernel identity")
    _, raw_rows = read_csv(data_path)
    sources = []
    for library, variant in LIBRARIES.items():
        selected = [row for row in rows if row.get("library") == library]
        if not selected:
            continue
        if len(selected) != 1:
            raise ValueError(f"Expected one Table 6 summary row for {library}")
        row = selected[0]
        if (not is_a100(row.get("gpu")) or row.get("N") != "256" or row.get("batch") != str(BATCH)
                or row.get("variant") != variant or row.get("selected_candidate") != variant):
            raise ValueError(f"{library} summary must describe the Table 6 A100 N256 batch65536 kernel")
        raw = [item for item in raw_rows if item.get("library") == library]
        if not raw or any(not is_a100(item.get("gpu")) or item.get("N") != "256"
                          or item.get("batch") != str(BATCH) or item.get("precision") != "fp16"
                          or item.get("input") != "complex" or item.get("variant") != variant
                          or item.get("candidate") != variant for item in raw):
            raise ValueError(f"{library} data.csv must confirm N256 FP16 complex batch65536 and the supported kernel")
        # Compare the rounded summary against its trials, avoiding a detached table.csv.
        median_ms = statistics.median(positive(item.get("time_ms"), "Table 6 raw time_ms") for item in raw)
        summary_ms = positive(row.get("median_time_ms"), "Table 6 median_time_ms")
        if not math.isclose(summary_ms, round(median_ms, 6), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"{library} summary timing disagrees with data.csv")
        sources.append({"library": library, "time_us": summary_ms * 1000,
                        "evidence": "table6_summary_with_raw_contract",
                        "note": "Uses rounded median_time_ms from Table 6 table.csv; adjacent trials confirm the workload, without new numerical validation."})
    if not sources:
        raise ValueError("No supported tcFFT or FlashFFTConv Table 6 rows")
    return sources, [file_record(path), file_record(data_path)]


def project(source):
    library = source["library"]
    if library not in LIBRARIES:
        raise ValueError("Only the patched Table 6 tcFFT and FlashFFTConv kernels have this operation count")
    source_us = positive(source["time_us"], "source_time_us")
    throughput = SOURCE_FLOPS * BATCH / source_us / 1e6
    utilization = throughput / SOURCE_PEAK * 100.0
    projected_us = source_us * TARGET_FLOPS / SOURCE_FLOPS * SOURCE_PEAK / TARGET_PEAK
    positive(throughput, "derived throughput")
    positive(projected_us, "projected time")
    return dict(zip(ESTIMATE_FIELDS, (
        library, source["evidence"], source_us, 256, "fp16", BATCH, SOURCE_FLOPS, SOURCE_PEAK,
        throughput, utilization, 64, "tf32", BATCH, TARGET_FLOPS, TARGET_PEAK,
        projected_us, utilization, "constant_hardware_peak_utilization",
        "32*N*log2(N); theoretical normalization, not an implemented N64 radix16 kernel",
        "estimated", "not_applicable_projection", False,
    )))


def comparison_rows(path, estimates):
    path = Path(path)
    fields, rows = read_csv(path)
    meta_path = path.with_name("metadata.json")
    meta = json.loads(meta_path.read_text())
    if meta.get("evaluation") != "table7" or not is_a100(meta.get("gpu", {}).get("name")):
        raise ValueError("Table 7 comparison requires adjacent metadata.json identifying an A100 Table 7 run")
    if not rows:
        raise ValueError("The Table 7 kernel comparison contains no measured rows")
    seen = set()
    for row in rows:
        if not row.get("variant") or row["variant"] in seen:
            raise ValueError("Table 7 variants must be nonempty and unique")
        seen.add(row["variant"])
        if (row.get("N") != "64" or row.get("batch") != str(BATCH) or row.get("precision") != "tf32"
                or row.get("timing_status") not in ("collected", "collected_unvalidated")
                or row.get("accuracy_status") not in ("passed", "unchecked")):
            raise ValueError("Table 7 rows must be collected N64 TF32 batch65536 timings with declared accuracy status")
        positive(row.get("median_time_ms"), "Table 7 median_time_ms")
    extra = ["comparison_record_type", "comparison_time_us", "projection_source_library"]
    speedups = {item["library"]: "speedup_vs_projected_" + item["library"].lower() for item in estimates}
    extra.extend(speedups.values())
    if any(field in fields for field in extra):
        raise ValueError("Input must be an original kernel_comparison.csv, not a previous projection export")
    fields = fields + extra
    result = []
    for estimate in estimates:
        row = dict.fromkeys(fields, "")
        row.update(variant="projected_" + estimate["library"].lower() + "_n64_tf32",
                   N=64, batch=BATCH, precision="tf32", timing_status="estimated",
                   accuracy_status="not_applicable_projection", paper_identity="unconfirmed",
                   comparison_record_type="projected_baseline", comparison_time_us=estimate["projected_time_us"],
                   projection_source_library=estimate["library"])
        # Measured median_time_ms remains blank for a model projection.
        if "utilization_pct" in row:
            row["utilization_pct"] = estimate["projected_utilization_pct"]
        for baseline in estimates:
            row[speedups[baseline["library"]]] = baseline["projected_time_us"] / estimate["projected_time_us"]
        result.append(row)
    for original in rows:
        row = dict(original)
        time_us = float(row["median_time_ms"]) * 1000
        row.update(comparison_record_type="measured_unit_kernel", comparison_time_us=time_us,
                   projection_source_library="")
        for estimate in estimates:
            row[speedups[estimate["library"]]] = estimate["projected_time_us"] / time_us
        result.append(row)
    return fields, result, [file_record(path), file_record(meta_path)]


def write_outputs(output, sources, inputs, comparison=None):
    estimates = [project(source) for source in sources]
    comparison_data = None
    inputs = list(inputs)
    if comparison is not None:
        fields, rows, records = comparison_rows(comparison, estimates)
        comparison_data = (fields, rows)
        inputs.extend(records)
    output = Path(output)
    names = ["baseline_estimates.csv", "projection_metadata.json"]
    if comparison_data is not None:
        names.append("comparison.csv")
    input_paths = {Path(record["path"]).resolve() for record in inputs}
    for name in names:
        target = output / name
        if target.resolve() in input_paths or target.exists():
            raise ValueError(f"Refusing to overwrite an input or existing output: {target}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "evaluation": "table7_baseline_projection", "full_paper_comparison": False,
        "status": "projection_complete", "gpu_work_performed": False,
        "sources": sources, "inputs": inputs, "calculation": file_record(Path(__file__)),
        "source_contract": {"gpu_family": "A100", "N": 256, "precision": "fp16", "batch": BATCH,
                            "kernel_variants": LIBRARIES, "tensor_flops_per_fft": SOURCE_FLOPS,
                            "hardware_peak_tflops": SOURCE_PEAK,
                            "timing_scope": "(time(2R)-time(R))/R; FFT-only resident compute"},
        "source_operation_count": "2 complex matrix products * 4 WMMA per product * (2*16*16*16) = 65536 FLOPs/FFT",
        "target_model": {"N": 64, "precision": "tf32", "batch": BATCH,
                         "model_flops_per_slot": TARGET_FLOPS, "hardware_peak_tflops": TARGET_PEAK,
                         "implemented_kernel": False,
                         "description": "32*N*log2(N) is a theoretical normalization; no N64 radix16 execution is asserted."},
        "assumption": "Source and target retain the same fraction of their respective dense hardware peaks.",
        "formula": "projected_us = source_us * (12288/65536) * (312/156)",
        "hardware_peak_reference": "https://images.nvidia.com/data-center/a100/a100-datasheet.pdf",
        "limitations": ["Projected times are not measured N64 latencies or full paper reproduction.",
                        "Unvalidated historical kernel rows keep their original source IDs and accuracy/provenance fields.",
                        "No utilization value is forced to 70%; it is calculated from the supplied source time."],
        "outputs": names,
    }
    with (output / "baseline_estimates.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ESTIMATE_FIELDS)
        writer.writeheader()
        writer.writerows(estimates)
    if comparison_data is not None:
        with (output / "comparison.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=comparison_data[0])
            writer.writeheader()
            writer.writerows(comparison_data[1])
    with (output / "projection_metadata.json").open("x") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    return estimates


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--documented-table6", action="store_true", help="Use the rounded public Table 6 README timings, with that evidence label")
    mode.add_argument("--table6-csv", type=Path, help="Table 6 table.csv; requires adjacent data.csv")
    mode.add_argument("--library", choices=LIBRARIES, help="Explicit supported Table 6 kernel timing; assumes A100 N256 FP16 complex batch65536")
    parser.add_argument("--source-time-us", type=float, help="Explicit differential resident-compute time for one batch, already divided by R")
    parser.add_argument("--source-note", help="Required provenance description for an explicit timing")
    parser.add_argument("--table7-csv", "--kernel-comparison", dest="kernel_comparison", type=Path,
                        help="Optional A100 Table 7 kernel_comparison.csv; requires adjacent metadata.json")
    parser.add_argument("--output", type=Path, required=True, help="Directory for new outputs; existing output files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.library:
            if args.source_time_us is None or not args.source_note or not args.source_note.strip():
                raise ValueError("--library requires --source-time-us and a nonempty --source-note")
            sources = [{"library": args.library, "time_us": positive(args.source_time_us, "source_time_us"),
                        "evidence": "user_supplied_timing", "note": args.source_note}]
            inputs = []
        else:
            if args.source_time_us is not None or args.source_note is not None:
                raise ValueError("--source-time-us/--source-note require --library")
            sources, inputs = documented_sources() if args.documented_table6 else table6_sources(args.table6_csv)
        estimates = write_outputs(args.output, sources, inputs, args.kernel_comparison)
    except (ValueError, OSError, TypeError, KeyError) as error:
        parser.error(str(error))
    for item in estimates:
        print(f"{item['library']}: {item['source_utilization_pct']:.4f}% source utilization; "
              f"{item['projected_time_us']:.4f} us estimated N64 TF32 time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
