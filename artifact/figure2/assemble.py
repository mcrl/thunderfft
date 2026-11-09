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
SUMMARY_FIELDS = ("gpu", "N", "batch", "placement", "library", "precision", "status", "candidate", "trials",
                  "median_us", "min_us", "max_us", "median_ms", "speedup_vs_cufftdx", "source_run", "source_sha256")
EXTRA_FIELDS = ("assembly_source_dir", "assembly_data_sha256", "time_ms")


def read_assembled(directory, profile):
    actual, metadata = TABLE3.read_assembled(directory, profile)
    actual = [row for row in actual if row["gpu"] == "A100" and row["placement"] == "register"]
    _, candidates, _ = TABLE3.MEASURE.summarize(actual, profile)
    selected = {(row["N"], row["precision"].upper()): row["candidate"] for row in candidates
                if row["selected"] == "true"}
    rows = [row for row in actual if row["library"] == "ThunderFFT" or
            row["candidate"] == selected[(row["N"], row["precision"].upper())]]
    groups = {}
    for row in rows:
        key = row["library"], int(row["N"]), row["precision"].upper()
        groups.setdefault(key, {})[int(row["trial"])] = float(row["time_us"])
    provenance = {"source_result_dir": str(directory), "source_run": directory.name,
                  "source_data_sha256": PLOT.sha256(directory / "data.csv"),
                  "source_metadata_sha256": PLOT.sha256(directory / "metadata.json")}
    contract = (metadata.get("cufftdx_selection", {}).get("A100", {"mode": "all"})
                if selected else {"mode": "not_measured"})
    provenance.update(PLOT.selection_provenance(contract))
    return rows, groups, provenance, selected


def assemble(parts, output, profile):
    output.mkdir(parents=True, exist_ok=True)
    parts = sorted({Path(part).resolve() for part in parts})
    if len(parts) > 1:
        raise ValueError("assemble Table 3 parts first; Figure 2 expects one assembled Table 3 directory")
    rows, groups, provenance, selected = [], {}, {}, {}
    if parts:
        rows, groups, provenance, selected = read_assembled(parts[0], profile)
        # The runner atomically renames its assembly staging directory afterward.
        provenance["source_result_dir"] = os.path.relpath(parts[0], output)
    missing = [{"item": f"A100/{library}/N={size}/{precision.lower()}/register",
                "reason": "Table 3 measurement CSV not collected"}
               for size in PLOT.SIZES for library, precision in PLOT.SERIES if (library, size, precision) not in groups]
    status = "complete" if not missing else "partial" if groups else "pending"
    result = {"status": status, "missing": missing, "full_paper_comparison": not missing and profile == "paper",
              "complete_declared_subset": sum(key[0] == "ThunderFFT" for key in groups) == 14,
              "excluded": [f"{library} {precision}" for library, precision in PLOT.SERIES
                           if not any((library, size, precision) in groups for size in PLOT.SIZES)],
              "collected_configurations": len(groups), "expected_configurations": 28}
    summaries = []
    for size in PLOT.SIZES:
        for library, precision in PLOT.SERIES:
            values = list(groups.get((library, size, precision), {}).values())
            row = {"gpu": "A100", "N": size, "placement": "register", "library": library,
                   "precision": precision, "status": "collected" if values else "pending"}
            if values:
                row.update(batch=rows[0]["batch"], trials=len(values), median_us=statistics.median(values),
                           min_us=min(values), max_us=max(values), median_ms=statistics.median(values) / 1000,
                           candidate=selected.get((size, precision), "") if library == "cuFFTDx" else "",
                           source_run=provenance["source_run"], source_sha256=provenance["source_data_sha256"])
                if library == "ThunderFFT":
                    baseline = list(groups.get(("cuFFTDx", size, PLOT.MATCHED[precision]), {}).values())
                    if baseline:
                        row["speedup_vs_cufftdx"] = statistics.median(baseline) / statistics.median(values)
            summaries.append(row)
    data = [{**row, "source_part": TABLE3.relative_source(output, row["source_part"]),
             "assembly_source_dir": provenance["source_result_dir"],
             "assembly_data_sha256": provenance["source_data_sha256"], "time_ms": float(row["time_us"]) / 1000}
            for row in rows]
    TABLE3.MEASURE.write_csv(output / "data.csv", data, (*TABLE3.DATA_FIELDS, *EXTRA_FIELDS))
    TABLE3.MEASURE.write_csv(output / "summary.csv", summaries, SUMMARY_FIELDS)
    names = ["data.csv", "summary.csv"]
    if groups:
        PLOT.make_plot(groups, output / "figure.pdf", bool(selected), allow_partial=True)
        names.append("figure.pdf")
    metadata = {"schema_version": 2, "evaluation": "figure2", "profile": profile, **result,
                "scope": "A100/register ThunderFFT and cuFFTDx; missing measurements remain blank", "source": provenance,
                "source_path_base": "assembly_directory",
                "operation": "offline plot assembly; no new GPU measurement",
                "cufftdx_selection": {f"N{n}-{precision}": candidate for (n, precision), candidate in sorted(selected.items())},
                "cufftdx_mode": provenance.get("cufftdx_mode", "not_measured"),
                "cufftdx_preset_sha256": provenance.get("cufftdx_preset_sha256"),
                "cufftdx_selection_rule": PLOT.selection_rule(provenance.get("cufftdx_mode", "not_measured")),
                "pairing": PLOT.MATCHED,
                "series_colors": {f"{library} {precision}": PLOT.COLORS[(library, precision)]
                                  for library, precision in PLOT.SERIES},
                "missing_points": "gaps, without zero fill or connecting across gaps",
                "timing_metric": "(t_2R_us - t_R_us) / R", "unit_conversion": "time_ms = time_us / 1000",
                "central_statistic": "median of trial compute_us", "error_bars": "trial minimum and maximum",
                "outputs": {name: PLOT.sha256(output / name) for name in names}}
    TABLE3.MEASURE.atomic_json(output / "metadata.json", metadata)
    return result
