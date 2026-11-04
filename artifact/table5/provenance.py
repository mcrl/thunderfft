#!/usr/bin/env python3
"""Persist the source and build identity used by the Table 5 adapter."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def capture(arguments, cwd=None):
    result = subprocess.run(arguments, cwd=cwd, text=True, capture_output=True, check=False)
    return {"command": arguments, "exit_code": result.returncode,
            "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}


def main():
    result_dir, source_dir, build_dir, binary = map(Path, sys.argv[1:])
    metadata_path = result_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    cache = {}
    for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
        if line.startswith(("#", "//")) or ":" not in line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key in {"CMAKE_BUILD_TYPE", "CMAKE_CUDA_COMPILER", "CMAKE_CXX_COMPILER",
                   "CMAKE_CUDA_FLAGS", "CMAKE_CUDA_FLAGS_RELEASE", "CMAKE_CUDA_ARCHITECTURES",
                   "THUNDERFFT_ARCHS", "CUDAToolkit_ROOT", "CMAKE_GENERATOR"}:
            cache[key] = value
    files = capture(["git", "ls-files", "--", "include", "sample/bench", "artifact/table5", "CMakeLists.txt"], cwd=source_dir)
    paths = set(files["stdout"].splitlines())
    snapshot_path = source_dir / ".thunderfft-cloud-source.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.is_file() else None
    if snapshot:
        paths.update(path for path in snapshot["files"]
                     if path.startswith(("include/", "sample/bench/", "artifact/table5/")) or path == "CMakeLists.txt")
    # Include this adapter's new files before they have been committed, too.
    paths.update(str(path.relative_to(source_dir)) for path in (source_dir / "artifact/table5").glob("*") if path.is_file())
    hashes = {path: hashlib.sha256((source_dir / path).read_bytes()).hexdigest()
              for path in sorted(paths) if (source_dir / path).is_file()}
    metadata["source"] = {
        "git_head": capture(["git", "rev-parse", "HEAD"], cwd=source_dir),
        "git_status": capture(["git", "status", "--short"], cwd=source_dir),
        "sha256": hashes,
    }
    if snapshot:
        metadata["source"]["cloud_snapshot"] = {"git": snapshot["git"], "manifest_sha256": hashlib.sha256(snapshot_path.read_bytes()).hexdigest()}
    metadata["build"] = {
        "cmake_cache": cache,
        "cmake_version": capture(["cmake", "--version"]),
        "cuda_compiler_version": capture([cache["CMAKE_CUDA_COMPILER"], "--version"]),
        "binary": str(binary.resolve()),
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "compile_commands": str((build_dir / "compile_commands.json").resolve())
            if (build_dir / "compile_commands.json").exists() else None,
    }
    compilation = build_dir / "compile_commands.json"
    if compilation.is_file():
        metadata["build"]["actual_compile_commands"] = [
            entry for entry in json.loads(compilation.read_text())
            if Path(entry["file"]).resolve() == (source_dir / "artifact/table5/main.cu").resolve()]
    target_dir = build_dir / "sample/bench/CMakeFiles/benchmark_table5_precision.dir"
    metadata["build"]["target_files"] = {
        name: (target_dir / name).read_text() for name in ("flags.make", "link.txt")
        if (target_dir / name).is_file()}
    metadata["execution"] = {
        "timeout_seconds_per_invocation": int(os.environ["THUNDERFFT_TABLE5_TIMEOUT"])
            if "THUNDERFFT_TABLE5_TIMEOUT" in os.environ else 600,
        "clock_or_power_settings_changed_by_adapter": False,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
