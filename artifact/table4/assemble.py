#!/usr/bin/env python3
"""Assemble Table 4 from Table 3, ideal MMA or portable derived bundles."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location("table4_derive", Path(__file__).with_name("derive.py"))
DERIVE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(DERIVE)


def snapshot_inputs(part, entries, ideal=False):
    if not isinstance(entries, list):
        raise ValueError("Table 4 bundle lacks source snapshots")
    directories = []
    for source in entries:
        if not isinstance(source, dict) or not isinstance(source.get("snapshot"), str):
            raise ValueError("invalid Table 4 source snapshot")
        relative = source["snapshot"]
        expected = source.get("sha256", {})
        if not isinstance(expected, dict) or not expected:
            raise ValueError("invalid Table 4 snapshot hashes")
        if ideal:
            if not {"ideal.csv", "ideal_metadata.json"}.issubset(expected):
                raise ValueError("ideal MMA snapshot metadata and CSV are required")
        elif set(expected) != set(DERIVE.SOURCE_FILES):
            raise ValueError("invalid Table 4 snapshot hashes")
        for name, digest in expected.items():
            if not isinstance(name, str) or not DERIVE.valid_hash(digest):
                raise ValueError("invalid Table 4 snapshot hashes")
            path = DERIVE.evidence_file(part, str(Path(relative) / name))
            if DERIVE.sha256(path) != digest:
                raise ValueError(f"{part}: source snapshot {name} hash mismatch")
        directories.append((part / relative).resolve())
    return directories


def assemble(parts, output, profile):
    sources, ideals = [], []
    for part in sorted({Path(part).resolve() for part in parts}):
        if (part / "ideal_metadata.json").is_file() and not (part / "metadata.json").exists():
            ideals.append(part)
            continue
        metadata = DERIVE.read_json(DERIVE.evidence_file(part, "metadata.json"))
        if metadata.get("evaluation") == "table3":
            sources.append(part)
            continue
        if (metadata.get("evaluation"), metadata.get("profile"), metadata.get("derivation_schema")) != (
                "table4", profile, DERIVE.SCHEMA):
            raise ValueError(f"{part}: unsupported Table 4 bundle or profile")
        outputs = metadata.get("outputs", {})
        if not isinstance(outputs, dict):
            raise ValueError("invalid Table 4 output hashes")
        output_names = ("data.csv", "table.csv", "table.md") + (("ideal.csv",) if "ideal.csv" in outputs else ())
        for name in output_names:
            if DERIVE.sha256(DERIVE.evidence_file(part, name)) != outputs.get(name):
                raise ValueError(f"{part}: derived {name} hash mismatch")
        sources.extend(snapshot_inputs(part, metadata.get("sources")))
        ideals.extend(snapshot_inputs(part, metadata.get("ideal_sources", []), ideal=True))
    # generate() deduplicates measurement identities, whether imported directly
    # or through derived snapshots; exact evidence hashes remain verified above.
    return DERIVE.generate(sources, output, profile, ideal_dirs=ideals)
