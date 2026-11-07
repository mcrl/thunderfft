#!/usr/bin/env python3
"""Clone tcFFT and FlashFFTConv at pinned commits, apply the patches and build the Table 6 drivers."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from selection import libraries

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

TCFFT = {"url": "https://github.com/rox906/tcFFT.git",
         "commit": "3e87a7083577741815e21b2a3c2bed22520786ce",
         "files": {"tcfft_half.cu": "441e6d959a8bcde190896bb77138a248e8897ed0994930a454e8f3112c7eb7d7",
                   "tcfft_half.h": "0a92c8f478dbc854ed964614e8047531a8ecbf36f76b58abfdb00bd37b571240"}}
FLASH_DIR = "csrc/flashfftconv/monarch_cuda/kernels_fp16"
FLASHFFTCONV = {"url": "https://github.com/HazyResearch/flash-fft-conv.git",
                "commit": "b8771028717f46d5b22cbb8e12833f35033d621b",
                "files": {
                    f"{FLASH_DIR}/monarch_cuda_kernel.h":
                        "9668debd79ae88379cb4b15bd072ec3c52c92049fee9c294d3370693750b492a",
                    f"{FLASH_DIR}/monarch_cuda_shared.h":
                        "fea344041cbabeb84acc63ab888cee1b88f1041f226974cbff5ad3b254ae8d64",
                    f"{FLASH_DIR}/shared/monarch_cuda_shared_fp16_complex_mul.h":
                        "2fe20103ac61fce2c7c3839397b539acf1eea27226bbd9c592d0c52c614a9547",
                    f"{FLASH_DIR}/shared/monarch_cuda_shared_fp16_matmuls.h":
                        "950fb1b2f411670bb4b1572a5f9478647127f05f2a18de0cfa4631475b785e8b",
                    f"{FLASH_DIR}/shared/monarch_cuda_shared_fp16_load_frags.h":
                        "9342d3ddd29c8188d5cda11cc22eca959c58e28d841a60f7302326d7d228d490"}}
PATCH = HERE / "tcfft/layer_256_0_A100_repeat.patch"
FLASH_PATCH = HERE / "flashfftconv/monarch_conv_fft_repeat.patch"
FLAGS = ["-std=c++17", "-O3", "-lineinfo", "--use_fast_math"]
TORCH_EXTENSION_DEFINES = ["-D__CUDA_NO_HALF_OPERATORS__", "-D__CUDA_NO_HALF_CONVERSIONS__",
                           "-D__CUDA_NO_BFLOAT16_CONVERSIONS__", "-D__CUDA_NO_HALF2_OPERATORS__"]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def run(command, **kwargs):
    print("$ " + shlex.join(str(part) for part in command), flush=True)
    result = subprocess.run([str(part) for part in command], stderr=subprocess.STDOUT,
                            check=False, **kwargs)
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {shlex.join(str(p) for p in command)}")


def clone(spec, root):
    """Clone the pinned commit into root, replacing anything there; check sha256; return provenance."""
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    run(["git", "init", "-q"], cwd=root)
    run(["git", "fetch", "-q", "--depth", "1", spec["url"], spec["commit"]], cwd=root)
    run(["git", "checkout", "-q", "--detach", "FETCH_HEAD"], cwd=root)
    provenance = {"root": str(root), "url": spec["url"], "pinned_commit": spec["commit"], "files": {}}
    for path, expected in spec["files"].items():
        digest = sha256((root / path).read_bytes())
        if digest != expected:
            raise RuntimeError(f"{path} does not match pinned commit {spec['commit']} (sha256 {digest})")
        provenance["files"][path] = digest
    return provenance


def torch_include(python):
    code = ("import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include')); "
            "print(torch.__version__)")
    result = subprocess.run([python, "-c", code], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError("PyTorch is required for FlashFFTConv's header-only c10 types: "
                           + result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "import failed")
    include, version = result.stdout.split()
    if not (Path(include) / "c10/util/complex.h").is_file():
        raise RuntimeError(f"{include} does not contain c10/util/complex.h")
    return include, version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--nvcc", default=os.environ.get("CUDACXX") or os.environ.get("NVCC") or "nvcc")
    parser.add_argument("--arch", default=os.environ.get("THUNDERFFT_ARCHS", "80"))
    parser.add_argument("--python", default=os.environ.get("THUNDERFFT_PYTHON", sys.executable))
    parser.add_argument("--libraries", help="Comma-separated Table 6 libraries (default: THUNDERFFT_TABLE6_LIBRARIES)")
    args = parser.parse_args()
    try:
        selected = libraries(args.libraries)
    except ValueError as error:
        parser.error(str(error))
    selected = tuple(name for name in selected if name in ("tcfft", "flashfftconv"))
    if not args.arch.isdigit():
        parser.error("THUNDERFFT_ARCHS must name exactly one architecture, e.g. 80")
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        parser.error(f"nvcc not found: {args.nvcc}")
    out = args.build_dir.resolve() / "artifact_table6"
    out.mkdir(parents=True, exist_ok=True)
    gencode = [f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}"]
    info = {"schema_version": 1, "nvcc": nvcc, "arch": args.arch, "binaries": {}}
    run([nvcc, "--version"])
    info["nvcc_version"] = subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout.strip()
    sources = HERE / "src"
    builds = {}
    if "tcfft" in selected:
        info["tcfft"] = clone(TCFFT, out / "tcfft")
        run(["patch", "-p1", "--no-backup-if-mismatch", "-i", PATCH], cwd=out / "tcfft")
        info["tcfft"]["patch"] = {"path": str(PATCH.relative_to(ROOT)), "sha256": sha256(PATCH.read_bytes()),
                                  "patched_sha256": sha256((out / "tcfft/tcfft_half.cu").read_bytes())}
        builds["tcfft"] = ([sources / "tcfft_bench.cu", "-I", sources, "-I", out / "tcfft"], ["-lcublas"])
    if "flashfftconv" in selected:
        info["flashfftconv"] = clone(FLASHFFTCONV, out / "flashfftconv")
        run(["patch", "-p1", "--no-backup-if-mismatch", "-i", FLASH_PATCH], cwd=out / "flashfftconv" / FLASH_DIR)
        info["flashfftconv"]["patch"] = {
            "path": str(FLASH_PATCH.relative_to(ROOT)), "sha256": sha256(FLASH_PATCH.read_bytes()),
            "patched_sha256": sha256((out / "flashfftconv" / FLASH_DIR / "monarch_cuda_kernel.h").read_bytes())}
        include, version = torch_include(args.python)
        info["flashfftconv"].update(torch_include=include, torch_version=version)
        builds["flashfftconv"] = ([sources / "flashfftconv_bench.cu", "-I", sources / "shim", "-I", include,
                                  "-I", out / "flashfftconv" / FLASH_DIR, "-I", sources] + TORCH_EXTENSION_DEFINES, [])
    for name, (inputs, links) in builds.items():
        binary = out / f"table6_{name}"
        command = [nvcc] + FLAGS + gencode + inputs + ["-o", binary] + links
        run(command)
        info["binaries"][name] = {"path": str(binary), "sha256": sha256(binary.read_bytes()),
                                  "command": [str(part) for part in command]}
    info["driver_sources"] = {str(path.relative_to(ROOT)): sha256(path.read_bytes())
                              for path in sorted(sources.rglob("*")) if path.is_file()}
    (out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(out / "build.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"Table 6 build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
