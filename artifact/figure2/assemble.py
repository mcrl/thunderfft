#!/usr/bin/env python3
"""Build Figure 2 from a validated offline Table 3 assembly, including gaps."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import statistics


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
PLOT = module("figure2_offline_plot", HERE / "plot.py")
TABLE3 = module("figure2_table3_assembly", HERE.parent / "table3/assemble.py")
SUMMARY_FIELDS = ("gpu", "N", "batch", "placement", "library", "precision", "status", "trials",
                  "median_us", "min_us", "max_us", "median_ms", "source_run", "source_sha256")
EXTRA_FIELDS = ("assembly_source_dir", "assembly_data_sha256", "time_ms")


def read_assembled(directory, profile):
    metadata_path, data_path = directory / "metadata.json", directory / "data.csv"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("assembly_schema") != "table3-validated-parts-v1" or metadata.get("evaluation") != "table3":
        raise ValueError("Figure 2 assembly requires a validated Table 3 assembly")
    if metadata.get("profile") != profile or metadata.get("status") not in ("complete", "partial", "pending"):
        raise ValueError("Table 3 assembly profile/status mismatch")
    if metadata.get("outputs", {}).get("data.csv") != PLOT.sha256(data_path):
        raise ValueError("assembled Table 3 data hash mismatch")
    actual = TABLE3.strict_csv(data_path, TABLE3.DATA_FIELDS)
    expected, identities, configurations = [], {}, set()
    common_source = None
    for source in metadata.get("sources", []):
        path = Path(source["path"])
        for relative, digest in source["sha256"].items():
            if PLOT.sha256(TABLE3.evidence_file(path, relative)) != digest:
                raise ValueError("Table 3 source evidence changed after assembly")
        measured, identity, _ = TABLE3.load_part(path, profile)
        if not measured:
            raise ValueError("Table 3 assembly contains an empty measurement source")
        gpu = measured[0]["gpu"]
        if common_source is not None and identity["source_sha256"] != common_source:
            raise ValueError("incompatible Table 3 source revisions across GPU families")
        common_source = identity["source_sha256"]
        if gpu in identities and identities[gpu] != identity:
            raise ValueError("incompatible Table 3 source parts")
        identities[gpu] = identity
        keys = {(row["gpu"], row["N"], row["precision"], row["placement"]) for row in measured}
        if configurations & keys:
            raise ValueError("overlapping Table 3 source configurations")
        configurations |= keys
        expected.extend(measured)
    expected.sort(key=lambda row: (TABLE3.GPUS.index(row["gpu"]), row["N"], row["precision"], row["placement"], row["trial"]))
    canonical = [{key: str(row[key]) for key in TABLE3.DATA_FIELDS} for row in expected]
    if actual != canonical:
        raise ValueError("assembled Table 3 rows disagree with validated original measurement evidence")
    if metadata.get("coverage", {}).get("measured_thunderfft_configurations") != len(configurations):
        raise ValueError("assembled Table 3 coverage disagrees with its data")
    expected_status = "partial" if configurations else "pending"
    if metadata["status"] != expected_status:
        raise ValueError("assembled Table 3 status disagrees with its coverage")
    rows = [row for row in actual if row["gpu"] == "A100" and row["placement"] == "register"]
    groups = {}
    for row in rows:
        key = "ThunderFFT", int(row["N"]), row["precision"].upper()
        groups.setdefault(key, {})[int(row["trial"])] = float(row["time_us"])
    provenance = {"source_result_dir": str(directory), "source_run": directory.name,
                  "source_data_sha256": PLOT.sha256(data_path), "source_metadata_sha256": PLOT.sha256(metadata_path)}
    return rows, groups, provenance


def assemble(parts, output, profile):
    output.mkdir(parents=True, exist_ok=True)
    parts = sorted({Path(part).resolve() for part in parts})
    if len(parts) > 1:
        raise ValueError("assemble Table 3 parts first; Figure 2 expects one assembled Table 3 directory")
    rows, groups, provenance = [], {}, {}
    if parts:
        rows, groups, provenance = read_assembled(parts[0], profile)
        # The runner atomically renames its assembly staging directory afterward.
        provenance["source_result_dir"] = os.path.relpath(parts[0], output)
    missing = [{"item": f"A100/N={size}/{precision.lower()}/register", "reason": "Table 3 measurement CSV not collected"}
               for size in PLOT.SIZES for precision in PLOT.PRECISIONS if ("ThunderFFT", size, precision) not in groups]
    missing.append({"item": "cuFFTDx FP16/FP32 series", "reason": "tuned baseline measurements are not supplied"})
    status = "partial" if groups else "pending"
    result = {"status": status, "missing": missing, "full_paper_comparison": False,
              "complete_declared_subset": len(groups) == 14,
              "excluded": ["cuFFTDx FP16", "cuFFTDx FP32"],
              "collected_configurations": len(groups), "expected_configurations": 14}
    summaries = []
    for size in PLOT.SIZES:
        for precision in PLOT.PRECISIONS:
            values = list(groups.get(("ThunderFFT", size, precision), {}).values())
            row = {"gpu": "A100", "N": size, "placement": "register", "library": "ThunderFFT",
                   "precision": precision, "status": "collected" if values else "pending"}
            if values:
                row.update(batch=rows[0]["batch"], trials=len(values), median_us=statistics.median(values),
                           min_us=min(values), max_us=max(values), median_ms=statistics.median(values) / 1000,
                           source_run=provenance["source_run"], source_sha256=provenance["source_data_sha256"])
            summaries.append(row)
    data = [{**row, "assembly_source_dir": provenance["source_result_dir"],
             "assembly_data_sha256": provenance["source_data_sha256"], "time_ms": float(row["time_us"]) / 1000}
            for row in rows]
    TABLE3.MEASURE.write_csv(output / "data.csv", data, (*TABLE3.DATA_FIELDS, *EXTRA_FIELDS))
    TABLE3.MEASURE.write_csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    names = ["data.csv", "summary.csv"]
    if groups:
        PLOT.make_plot(groups, output / "figure.pdf", False, allow_partial=True)
        names.append("figure.pdf")
    metadata = {"schema_version": 1, "evaluation": "figure2", "profile": profile, **result,
                "scope": "ThunderFFT-only A100/register Figure 2 subset", "source": provenance,
                "operation": "offline plot assembly; no new GPU measurement",
                "series_colors": {f"ThunderFFT {precision}": PLOT.COLORS[("ThunderFFT", precision)]
                                  for precision in PLOT.PRECISIONS}, "missing_points": "gaps, without zero fill or connecting across gaps",
                "timing_metric": "(t_2R_us - t_R_us) / R", "unit_conversion": "time_ms = time_us / 1000",
                "central_statistic": "median of trial compute_us", "error_bars": "trial minimum and maximum",
                "outputs": {name: PLOT.sha256(output / name) for name in names}}
    TABLE3.MEASURE.atomic_json(output / "metadata.json", metadata)
    return result
