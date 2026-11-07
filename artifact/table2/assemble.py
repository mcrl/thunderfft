#!/usr/bin/env python3
"""Assemble Table 2 from its declared model and validated Table 3 parts."""

import csv
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


HERE = Path(__file__).resolve().parent
MODEL = module("table2_assembly_model", HERE / "model.py")
TABLE3 = module("table2_assembly_table3", HERE.parent / "table3/assemble.py")


def validate_model(part, metadata, config, profile, fallback_sources=()):
    if metadata.get("profile") != profile or metadata.get("assumptions") != config \
            or metadata.get("assumptions_sha256") != MODEL.sha256(HERE / "assumptions.json"):
        raise ValueError("Table 2 model assumptions/profile mismatch")
    rows = TABLE3.strict_csv(part / "data.csv", MODEL.FIELDS)
    if TABLE3.strict_csv(part / "table.csv", MODEL.FIELDS) != rows:
        raise ValueError("Table 2 data and table disagree")
    calculated = [row for row in rows if row["origin"] == "calculated"]
    expected = [{key: str(row.get(key, "")) for key in MODEL.FIELDS} for row in MODEL.calculate(config)]
    if calculated != expected:
        raise ValueError("Table 2 calculated rows disagree with declared model")
    measured = [row for row in rows if row["origin"] == "measured"]
    if len(rows) != len(calculated) + len(measured) or len(measured) > 1:
        raise ValueError("Table 2 has unexpected or duplicate measured rows")
    if not measured:
        return None
    row = measured[0]
    recomputed = MODEL.measured_row(config, float(row["measured_time_us"]), int(row["batch"]), row["candidate"])
    if row != {key: str(recomputed.get(key, "")) for key in MODEL.FIELDS}:
        raise ValueError("Table 2 measured row disagrees with its formula")
    source = metadata.get("measured_source") or {}
    directory = Path(source.get("source_result_dir", ""))
    if not directory.is_absolute():
        directory = part / directory
    if not (directory / "data.csv").is_file():
        matches = [candidate for candidate in fallback_sources if (candidate / "data.csv").is_file()
                   and MODEL.sha256(candidate / "data.csv") == source.get("source_sha256")
                   and (not source.get("source_metadata_sha256") or
                        MODEL.sha256(candidate / "metadata.json") == source["source_metadata_sha256"])]
        if not matches:
            raise ValueError("Table 2 measured source is missing; import its Table 3 bundle as well")
        directory = matches[0]
    if MODEL.sha256(directory / "data.csv") != source.get("source_sha256"):
        raise ValueError("Table 2 measured source changed")
    for relative, digest in source.get("snapshot_sha256", {}).items():
        if MODEL.sha256(TABLE3.evidence_file(directory, relative)) != digest:
            raise ValueError("Table 2 source snapshot changed")
    # The original single-run adapter and this assembler use the same selection.
    return directory.resolve(), row


def assemble(parts, output, profile):
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported Table 2 profile")
    output.mkdir(parents=True, exist_ok=True)
    inputs = HERE / "assumptions.json"
    config = json.loads(inputs.read_text())
    source_paths, model_rows, inputs_evidence = [], [], []
    parts = sorted({Path(part).resolve() for part in parts})
    inputs_metadata = {part: json.loads((part / "metadata.json").read_text()) for part in parts}
    fallback_sources = [part for part, metadata in inputs_metadata.items()
                        if metadata.get("evaluation") == "table3" and not metadata.get("assembly_schema")]
    for part, metadata in inputs_metadata.items():
        evaluation = metadata.get("evaluation")
        if evaluation == "table2":
            existing = validate_model(part, metadata, config, profile, fallback_sources)
            if existing:
                source, row = existing
                source_paths.append(source)
                model_rows.append(row)
        elif evaluation == "table3":
            source_paths.append(part)
        else:
            raise ValueError(f"Table 2 cannot consume evaluation {evaluation!r}")
        inputs_evidence.append({"path": str(part), "evaluation": evaluation,
                                "metadata_sha256": MODEL.sha256(part / "metadata.json")})
    raw_sources = []
    for source in source_paths:
        metadata = json.loads((source / "metadata.json").read_text())
        if metadata.get("assembly_schema") == "table3-validated-parts-v1":
            _, verified = TABLE3.read_assembled(source, profile)
            raw_sources += [TABLE3.resolve_source(source, item["path"]) for item in verified["sources"]]
        else:
            raw_sources.append(source)
    # Runner snapshots and original dependency directories may contain exactly
    # the same immutable bundle. Count that evidence once, without averaging it.
    unique = {}
    for source in raw_sources:
        if not (source / "data.csv").is_file():
            continue
        digest = MODEL.sha256(source / "data.csv"), MODEL.sha256(source / "metadata.json")
        unique.setdefault(digest, source)
    source_rows, sources, _, _, _ = TABLE3.collect_parts(unique.values(), profile)
    summaries, candidates, _ = TABLE3.MEASURE.summarize(source_rows, profile)
    reference = config["measured_reference"]
    matches = [row for row in summaries if all(row[key] == reference[key] for key in
               ("gpu", "N", "precision", "placement", "library"))]
    measured, provenance, pending = None, None, None
    if matches:
        row = matches[0]
        measured = MODEL.measured_row(config, row["median_time_us"], row["batch"], row["candidate"])
        native_rows = [item for item in source_rows if all(item[key] == row[key] for key in
                       ("gpu", "N", "precision", "placement", "library", "candidate"))]
        source = Path(native_rows[0]["source_part"])
        provenance = {"source_run": source.name, "source_result_dir": str(source),
                      "source_file": str(source / "data.csv"), "source_sha256": MODEL.sha256(source / "data.csv"),
                      "source_metadata_sha256": MODEL.sha256(source / "metadata.json"),
                      "inside_repeats": row["inside_repeats"], "trials": row["trials"],
                      "candidates_compared": row["cufftdx_candidates"],
                      "trial_time_us": sorted(item["time_us"] for item in native_rows)}
        canonical = {key: str(measured.get(key, "")) for key in MODEL.FIELDS}
        if any(existing != canonical for existing in model_rows):
            raise ValueError("Table 2 measured model part disagrees with validated Table 3 source")
        provenance = MODEL.snapshot_source(output, source, provenance)
    elif model_rows:
        raise ValueError("Table 2 measured model part lacks validated reference configuration")
    else:
        pending = (f"validated Table 3 {reference['gpu']} cuFFTDx N={reference['N']} "
                   f"{reference['precision']} {reference['placement']} not collected")
    rows = MODEL.calculate(config)
    if measured:
        rows.insert(1, measured)
    MODEL.write_outputs(output, config, inputs, rows, measured, provenance, profile, pending)
    result = {"status": "complete" if measured else "partial",
              "missing": [] if measured else [{"item": "cuFFTDx measured row", "reason": pending}],
              "full_paper_comparison": bool(measured) and profile == "paper",
              "collected_configurations": len(rows), "expected_configurations": 5}
    metadata = json.loads((output / "metadata.json").read_text())
    metadata.update(result, assembly_schema="table2-model-and-measurement-v1",
                    inputs=inputs_evidence, native_sources=sources,
                    operation="offline model calculation and native Table 3 evidence validation; no new measurements",
                    outputs={name: MODEL.sha256(output / name) for name in ("data.csv", "table.csv", "table.md")})
    TABLE3.MEASURE.atomic_json(output / "metadata.json", metadata)
    return result
