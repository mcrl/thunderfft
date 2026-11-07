#!/usr/bin/env python3
"""Build the pinned, unmodified FlashFFTConv 4096 CUDA kernel as a native runner."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "table6"))
_spec = importlib.util.spec_from_file_location("figure4_external_build", HERE.parent / "table6/build.py")
BUILD = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(BUILD)
SPEC = dict(BUILD.FLASHFFTCONV, files=dict(BUILD.FLASHFFTCONV["files"]))
SPEC["files"][f"{BUILD.FLASH_DIR}/monarch_cuda_16_16_16_kernel.h"] = "0ae2a622cd8293b09e9d95ab0f9c96c38478de6f30145d155df16cbf10a6abd0"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--python", default=os.environ.get("FLASHFFTCONV_PYTHON", sys.executable))
    parser.add_argument("--nvcc", default=os.environ.get("CUDACXX", "nvcc"))
    parser.add_argument("--arch", default=os.environ.get("THUNDERFFT_ARCHS", "80"))
    args = parser.parse_args()
    if not args.arch.isdigit():
        parser.error("FlashFFTConv requires one numeric architecture, e.g. THUNDERFFT_ARCHS=80")
    nvcc = shutil.which(args.nvcc)
    if not nvcc:
        parser.error(f"nvcc not found: {args.nvcc}")
    include, version = BUILD.torch_include(args.python)
    out = args.build_dir.resolve() / "artifact_figure4"
    out.mkdir(parents=True, exist_ok=True)
    source = out / "flashfftconv"
    # Reuse only exact pinned, unmodified files. A changed source is fetched again.
    if not all((source / name).is_file() and BUILD.sha256((source / name).read_bytes()) == digest
               for name, digest in SPEC["files"].items()):
        provenance = BUILD.clone(SPEC, source)
    else:
        provenance = {"root": str(source), "url": SPEC["url"], "pinned_commit": SPEC["commit"], "files": SPEC["files"]}
    binary = out / "figure4_flashfftconv"
    command = [nvcc, "-std=c++17", "-O3", "-lineinfo", f"-gencode=arch=compute_{args.arch},code=sm_{args.arch}",
               str(HERE / "flashfftconv.cu"), "-I", str(HERE.parent / "table6/src/shim"), "-I", include,
               "-I", str(source / BUILD.FLASH_DIR), *BUILD.TORCH_EXTENSION_DEFINES, "-lcufft", "-o", str(binary)]
    BUILD.run(command)
    info = {"schema_version": 1, "external": provenance, "torch_version": version, "arch": args.arch,
            "command": command, "binary": str(binary), "binary_sha256": BUILD.sha256(binary.read_bytes()),
            "source_sha256": {str(p.relative_to(ROOT)): BUILD.sha256(p.read_bytes())
                              for p in (HERE / "flashfftconv.cu", Path(__file__), ROOT / "sample/utils.h",
                                        HERE.parent / "table6/src/shim/torch/extension.h")}}
    (out / "build.json").write_text(json.dumps(info, indent=2) + "\n")
    print(binary)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError) as error:
        print(f"Figure 4 FlashFFTConv build failed: {error}", file=sys.stderr)
        raise SystemExit(1)
