#!/usr/bin/env python3
"""Assemble independently collected Table 6 libraries without launching a GPU."""
import importlib.util
import json
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
M = module("table6_assembly_measure", HERE / "measure.py")
E = module("table6_assembly_evidence", HERE / "evidence.py")
DATA_FIELDS = (*M.FIELDS, "source_part", "source_raw_sha256")
TABLE_FIELDS = (*M.TABLE_FIELDS, "status", "source_part")


def load_part(part, profile):
    part = Path(part).resolve()
    meta = E.metadata(part, "table6", profile)
    stored = E.strict_csv(E.evidence_file(part, "data.csv"), M.FIELDS)
    if not stored:
        raise ValueError(f"{part}: empty data.csv")
    if meta["settings"]["batch"] % 512:
        raise ValueError("Table 6 requires batch divisible by 512")
    if profile == "paper" and meta["settings"]["repeats"] != 1000:
        raise ValueError("Table 6 paper assembly requires R=1000")
    hashes = {name: E.sha256(part / name) for name in ("metadata.json", "data.csv")}
    parsed, identities, libraries = [], {}, set()
    for record in meta.get("commands", []):
        library = record.get("library")
        if library not in M.SELECTION.LIBRARIES or library in libraries or record.get("exit_code") != 0:
            raise ValueError("Table 6 requires one successful command per selected library, with raw CSV/log evidence")
        libraries.add(library)
        binary_digest = meta.get("binary_sha256", {}).get(library)
        if not E.digest(binary_digest):
            raise ValueError(f"missing {library} binary SHA256")
        csv_path = E.evidence_file(part, record.get("stdout"))
        log_path = E.evidence_file(part, record.get("stderr"))
        log = log_path.read_text()
        identity = E.native_identity(meta, log, M.TABLE3, binary=library)
        E.merge_identity(identities, identity)
        args = [str(meta["settings"][key]) for key in ("repeats", "trials", "batch")]
        if library == "cufftdx":
            args = ["--library", "cufftdx", "--candidate", M.CUFFTDX_CANDIDATE, "256", "fp16", "register"] + args
        elif library == "thunderfft":
            args = ["256", "fp16", "register"] + args
        if record.get("argv", [])[1:] != args:
            raise ValueError("Table 6 native argv disagrees with metadata")
        if library in ("tcfft", "flashfftconv"):
            build = meta.get("table6_build", {})
            if build.get("binaries", {}).get(library, {}).get("sha256") != binary_digest:
                raise ValueError("external binary digest disagrees with build provenance")
            provenance = build.get(library, {})
            if not provenance.get("pinned_commit") or not E.digest(provenance.get("patch", {}).get("sha256")):
                raise ValueError("external library lacks pinned source and patch evidence")
        revision = record.get("revision")
        if not isinstance(revision, str) or not revision:
            raise ValueError("missing library revision")
        parser = M.parse_paper_fft if library in ("thunderfft", "cufftdx") else M.parse_native
        rows = parser(csv_path.read_text(), library=library, batch=identity["batch"], repeats=identity["repeats"],
                      trials=identity["trials"], gpu=identity["gpu"], revision=revision)
        if library == "cufftdx" and {row["candidate"] for row in rows} != {M.CUFFTDX_CANDIDATE}:
            raise ValueError("cuFFTDx candidate differs from the requested Table 6 configuration")
        parsed.extend({**row, "source_part": str(part), "source_raw_sha256": E.sha256(csv_path)} for row in rows)
        hashes[record["stdout"]], hashes[record["stderr"]] = E.sha256(csv_path), E.sha256(log_path)
    if not identities:
        raise ValueError("Table 6 bundle has no raw command evidence; historical summary-only values cannot be imported")
    E.check_saved(stored, parsed, M.FIELDS)
    if set(meta.get("selection", {}).get("libraries", [])) != libraries:
        raise ValueError("collected libraries disagree with declared selection")
    return parsed, next(iter(identities.values())), {"path": str(part), "sha256": hashes,
        "binary_sha256": meta["binary_sha256"], "build_configuration": meta.get("build_configuration"),
        "table6_build": meta.get("table6_build")}


def assemble(parts, output, profile):
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported assembly profile")
    parts, output = E.assembly_paths(parts, output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, identities, configurations = [], [], {}, {}
    for part in parts:
        measured, identity, provenance = load_part(part, profile)
        E.merge_identity(identities, identity)
        keys = {(row["gpu"], row["library"], row["variant"]) for row in measured}
        if keys & configurations.keys():
            raise ValueError("overlapping Table 6 libraries; retain one part per GPU/library")
        configurations.update({key: str(part) for key in keys})
        rows.extend(measured)
        sources.append(provenance)
    rows.sort(key=lambda row: (row["gpu"], row["library"], row["variant"], row["candidate"], row["repeat"]))
    summaries = {(row["gpu"], row["library"], row["variant"]): row for row in M.summarize(rows)}
    table, missing = [], []
    for gpu in sorted(set(identities) | {"A100"}):
        for library, variant in M.VARIANTS:
            key = gpu, library, variant
            if key in summaries:
                table.append({**summaries[key], "status": "collected", "source_part": configurations[key]})
            else:
                table.append({"gpu": gpu, "N": M.N, "library": library, "variant": variant, "status": "pending"})
                missing.append({"item": f"{gpu}/{library}", "reason": "measurement CSV not collected"})
    complete = bool(rows) and not missing
    result = {"status": "complete" if complete else "partial" if rows else "pending", "missing": missing,
              "full_paper_comparison": complete and profile == "paper", "collected_configurations": len(summaries)}
    M.TABLE3.write_csv(output / "data.csv", rows, DATA_FIELDS)
    M.TABLE3.write_csv(output / "table.csv", table, TABLE_FIELDS)
    M.TABLE3.atomic_json(output / "metadata.json", {"schema_version": 1, "evaluation": "table6", "profile": profile,
        "assembly_schema": "table6-validated-parts-v1", **result, "sources": sources, "compatibility_by_gpu": identities,
        "operation": "offline assembly; no new measurements", "outputs": {name: E.sha256(output / name)
        for name in ("data.csv", "table.csv")}})
    return result
