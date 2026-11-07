#!/usr/bin/env python3
"""Validate independent Figure 3 raw bundles and assemble normalized plots offline."""
import importlib.util
from pathlib import Path


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


HERE = Path(__file__).resolve().parent
M = module("figure3_assembly_measure", HERE / "measure.py")
PLOT = module("figure3_assembly_plot", HERE / "plot.py")
E = module("figure3_assembly_evidence", HERE.parent / "table6/evidence.py")
DATA_FIELDS = (*M.FIELDS, "source_part", "source_raw_sha256")
SUMMARY_FIELDS = (*M.SUMMARY_FIELDS, "status", "source_part")


def load_part(part, profile):
    part = Path(part).resolve()
    meta = E.metadata(part, "figure3", profile)
    stored = E.strict_csv(E.evidence_file(part, "data.csv"), M.FIELDS)
    if not stored:
        raise ValueError(f"{part}: empty data.csv")
    hashes = {name: E.sha256(part / name) for name in ("metadata.json", "data.csv")}
    parsed, identities, keys, active_binaries = [], {}, set(), {}
    for record in meta.get("commands", []):
        argv = record.get("argv", [])
        if record.get("exit_code") != 0 or len(argv) != 9 or argv[1] != "--library" or argv[2] not in M.NATIVE:
            raise ValueError("Figure 3 requires successful native commands with raw CSV/log evidence")
        library, n, precision = argv[2], int(argv[3]), argv[4]
        if n not in M.SIZES or precision not in M.ELEMENT_BYTES or (n, library, precision) in keys:
            raise ValueError("duplicate or unsupported Figure 3 configuration")
        if library in ("copy", "copy-scalar", "thunderfft") and precision not in M.PRECISIONS:
            raise ValueError("unexpected Figure 3 native precision")
        if library == "cufft" and precision != "fp32":
            raise ValueError("cuFFT Figure 3 measurements require FP32")
        if library == "cufftdx" and precision not in ("fp16", "fp32"):
            raise ValueError("cuFFTDx Figure 3 measurements require FP16 or FP32")
        keys.add((n, library, precision))
        sha = meta.get("cufftdx_binary_sha256" if library == "cufftdx" else "binary_sha256")
        if not E.digest(sha):
            raise ValueError("missing Figure 3 executable SHA256 evidence")
        active_binaries["cufftdx" if library == "cufftdx" else "paper_fft"] = sha
        expected = [M.NATIVE[library][1]] + [str(meta["settings"][key]) for key in ("repeats", "trials", "batch")]
        if argv[5:] != expected:
            raise ValueError("Figure 3 native argv disagrees with metadata")
        csv_path, log_path = (E.evidence_file(part, record.get(key)) for key in ("stdout", "stderr"))
        identity = E.native_identity(meta, log_path.read_text(), M.TABLE3)
        E.merge_identity(identities, identity)
        options = dict(n=n, precision=precision, batch=identity["batch"], repeats=identity["repeats"],
                       trials=identity["trials"], gpu=identity["gpu"], raw_csv=record["stdout"])
        if library == "cufftdx":
            best, rows = M.parse_cufftdx(csv_path.read_text(), **options)
            if meta.get("cufftdx_selection", {}).get(f"N{n}-{precision}") != best:
                raise ValueError("cuFFTDx selection disagrees with validated native CSV")
        else:
            rows = M.parse_native(csv_path.read_text(), library=library, **options)
        parsed.extend({**row, "source_part": str(part), "source_raw_sha256": E.sha256(csv_path)} for row in rows)
        hashes[record["stdout"]], hashes[record["stderr"]] = E.sha256(csv_path), E.sha256(log_path)
    if not identities:
        raise ValueError("no native commands collected")
    E.check_saved(stored, parsed, M.FIELDS)
    selection = meta.get("selection", {})
    if set(selection.get("N", [])) != {key[0] for key in keys}:
        raise ValueError("native sizes disagree with declared selection")
    requested = selection.get("libraries")
    if requested is not None:
        expected = set()
        for n in selection.get("N", []):
            for precision in selection.get("precisions", []):
                for library in requested:
                    if library not in M.NATIVE or precision not in M.PRECISIONS:
                        raise ValueError("invalid declared library or precision")
                    if library == "cufft" and precision != "tf32":
                        continue
                    native_precision = M.TABLE3.MATCHED_PRECISION[precision] if library in ("cufft", "cufftdx") else precision
                    expected.add((n, library, native_precision))
        if expected != keys:
            raise ValueError("native configurations disagree with declared selection")
    return parsed, next(iter(identities.values())), {"path": str(part), "sha256": hashes,
        "binary_sha256": meta.get("binary_sha256"), "cufftdx_binary_sha256": meta.get("cufftdx_binary_sha256"),
        "active_binaries": active_binaries, "build_configuration": meta.get("build_configuration")}


def assemble(parts, output, profile):
    if profile not in ("smoke", "paper"):
        raise ValueError("unsupported assembly profile")
    parts, output = E.assembly_paths(parts, output)
    output.mkdir(parents=True, exist_ok=True)
    rows, sources, identities, configurations, binaries = [], [], {}, {}, {}
    for part in parts:
        measured, identity, provenance = load_part(part, profile)
        E.merge_identity(identities, identity)
        for executable, sha in provenance["active_binaries"].items():
            binary_key = identity["gpu"], executable
            if binary_key in binaries and binaries[binary_key] != sha:
                raise ValueError(f"incompatible {executable} binary hashes across Figure 3 parts")
            binaries[binary_key] = sha
        keys = {(row["gpu"], row["N"], row["library"], row["precision"]) for row in measured}
        if keys & configurations.keys():
            raise ValueError("overlapping Figure 3 configurations; retain one part per GPU/N/library/precision")
        configurations.update({key: str(part) for key in keys})
        rows.extend(measured)
        sources.append(provenance)
    rows.sort(key=lambda row: (row["gpu"], row["N"], row["library"], row["precision"], row["repeat"]))
    summary = M.summarize(rows)
    for row in summary:
        row["status"] = "collected" if row["normalized"] != "" else "collected_unscaled"
        row["source_part"] = configurations[(row["gpu"], row["N"], row["library"], row["precision"])]
    present = set(configurations)
    missing = []
    for gpu in sorted(identities or {"A100": None}):
        precisions = ("tf32", "fp16") if any(row["gpu"] == gpu and row["precision"] == "fp16" for row in rows) else ("tf32",)
        for n, library, precision in M.expected_configurations(precisions):
            if (gpu, n, library, precision) not in present:
                summary.append({"gpu": gpu, "N": n, "library": library, "precision": precision, "status": "pending"})
                missing.append({"item": f"{gpu}/N{n}/{library}/{precision}", "reason": "measurement CSV not collected"})
    M.TABLE3.write_csv(output / "data.csv", rows, DATA_FIELDS)
    M.TABLE3.write_csv(output / "baselines.csv", [row for row in rows if row["library"] in ("cuFFT", "cuFFTDx")], DATA_FIELDS)
    M.TABLE3.write_csv(output / "summary.csv", summary, SUMMARY_FIELDS)
    complete = bool(rows) and not missing
    result = {"status": "complete" if complete else "partial" if rows else "pending", "missing": missing,
              "full_paper_comparison": complete and profile == "paper", "collected_configurations": len(present)}
    try:
        bars = [row for row in summary if row["library"] in PLOT.SERIES and row.get("normalized", "") != ""]
        PLOT.make_plot(bars, output / "figure.pdf")
        plot = {"status": "complete" if bars else "pending", "normalized_bars": len(bars)}
    except PLOT.Pending as error:
        plot = {"status": "pending", "reason": str(error)}
        missing.append({"item": "figure.pdf", "reason": str(error)})
        result.update(status="partial" if rows else "pending", full_paper_comparison=False)
    M.TABLE3.atomic_json(output / "metadata.json", {"schema_version": 1, "evaluation": "figure3", "profile": profile,
        "assembly_schema": "figure3-validated-parts-v1", **result, "sources": sources, "compatibility_by_gpu": identities,
        "plot": plot, "operation": "offline assembly; no new measurements", "outputs": {name: E.sha256(output / name)
        for name in ("data.csv", "baselines.csv", "summary.csv", "figure.pdf") if (output / name).is_file()}})
    return result
