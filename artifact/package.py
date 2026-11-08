#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact.cloud.transport import SOURCE_MANIFEST, _SECRET_CONTENT, _excluded_source


def git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True).stdout.decode()


def main():
    parser = argparse.ArgumentParser(description="Build the artifact distribution zip from tracked files.")
    parser.add_argument("--output", type=Path, help="default: dist/thunderfft-<revision>.zip")
    args = parser.parse_args()
    head = git("rev-parse", "HEAD").strip()
    status = git("status", "--porcelain=v1", "--untracked-files=no")
    manifest = {"schema_version": 1,
                "git": {"head": head, "branch": git("branch", "--show-current").strip(),
                        "status_porcelain": status, "dirty": bool(status)},
                "files": {}, "excluded_files": []}
    entries = []
    for name in sorted(n for n in git("ls-files", "-z").split("\0") if n):
        path = ROOT / name
        if _excluded_source(PurePosixPath(name)):
            manifest["excluded_files"].append(name)
            continue
        if path.is_symlink() or not path.is_file():
            continue
        data = path.read_bytes()
        if _SECRET_CONTENT.search(data):
            sys.exit(f"package: credential-like content in {name}")
        mode = 0o755 if path.stat().st_mode & 0o111 else 0o644
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "mode": mode}
        entries.append((name, data, mode))
    entries.append((SOURCE_MANIFEST, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(), 0o644))
    output = (args.output or ROOT / "dist" / f"thunderfft-{head[:12]}.zip").absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, data, mode in entries:
            info = zipfile.ZipInfo(f"thunderfft/{name}", (2026, 1, 1, 0, 0, 0))
            info.external_attr = (0o100000 | mode) << 16
            info.create_system = 3
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    print(output)


if __name__ == "__main__":
    main()
