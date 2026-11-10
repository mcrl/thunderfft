"""Fixed cuFFTDx candidates selected from a retained, complete paper sweep."""

import csv
import hashlib
import io
import json
from pathlib import Path
import re


MANIFEST_PATH = Path(__file__).with_suffix(".json")
MANIFEST_RELATIVE = "artifact/table3/cufftdx_presets.json"
GEOMETRY_FIELDS = ("ffts_per_block", "elements_per_thread", "threads_per_block", "shared_bytes")
CAPABILITIES = {"A100": "8.0", "H100": "9.0", "B200": "10.0"}
SM = {"A100": 800, "H100": 900, "B200": 1000}
SIZES = (64, 128, 256, 512, 1024, 2048, 4096)


def load_manifest():
    content = MANIFEST_PATH.read_bytes()
    manifest = json.loads(content)
    if manifest.get("schema_version") != 1 or set(manifest.get("gpus", {})) != set(CAPABILITIES):
        raise ValueError("invalid cuFFTDx preset manifest schema or GPU coverage")
    expected = {(n, precision, placement) for n in SIZES
                for precision in ("fp16", "fp32") for placement in ("register", "shared")}
    for family, gpu in manifest["gpus"].items():
        if (gpu["compute_capability"] != CAPABILITIES[family]
                or gpu["build"] != {"cufftdx_version": 10501, "cufftdx_target_sm": SM[family]}):
            raise ValueError(f"invalid cuFFTDx preset build identity for {family}")
        cells = gpu["cells"]
        if len(cells) != len(expected) or {(r["N"], r["precision"], r["placement"]) for r in cells} != expected:
            raise ValueError(f"incomplete or duplicate cuFFTDx preset cells for {family}")
        for row in cells:
            if not re.fullmatch(r"block_default|block_f\d+_e\d+|thread_t\d+", row["candidate"]):
                raise ValueError("invalid cuFFTDx preset candidate label")
            if set(row["geometry"]) != set(GEOMETRY_FIELDS):
                raise ValueError("incomplete cuFFTDx preset geometry")
            if any(type(v) is not int or v < (0 if key == "shared_bytes" else 1)
                   for key, v in row["geometry"].items()):
                raise ValueError("invalid cuFFTDx preset geometry")
    return manifest, hashlib.sha256(content).hexdigest()


def family_for_device(name, memory_mib):
    normalized = re.sub(r"[ _-]+", " ", name.upper()).removeprefix("NVIDIA ")
    tokens = normalized.split()
    for family in ("A100", "H100"):
        if (family in tokens and "PCIE" in tokens and "SXM" not in tokens and "NVL" not in tokens
                and 79000 <= memory_mib <= 84000):
            return family
    if normalized == "B200" and memory_mib >= 175000:
        return "B200"
    raise ValueError(f"no cuFFTDx preset for GPU {name!r} with {memory_mib} MiB; use --cufftdx-mode all")


def inventory_devices(inventory):
    if inventory.get("exit_code") != 0:
        raise ValueError("cuFFTDx preset selection requires a successful GPU inventory")
    devices = []
    for row in csv.reader(io.StringIO(inventory.get("stdout", "")), skipinitialspace=True):
        if len(row) < 5:
            raise ValueError("malformed GPU inventory for cuFFTDx preset selection")
        devices.append({"index": row[0].strip(), "uuid": row[1].strip(), "name": row[2].strip(),
                        "memory_mib": int(row[4].split()[0])})
    return devices


def select_device(inventory, visibility):
    devices = inventory_devices(inventory)
    if visibility is None:
        if len(devices) != 1:
            raise ValueError("cuFFTDx preset selection needs one explicitly visible CUDA device")
        selected = devices
    else:
        token = visibility.split(",")[0].strip()
        if not token or token == "-1":
            raise ValueError("cuFFTDx preset selection has no visible CUDA device")
        if len(devices) > 1 and token.isdecimal():
            raise ValueError("cuFFTDx-only presets on a multi-GPU host require a UUID visibility mask; CUDA ordinals may differ from nvidia-smi indices")
        selected = [d for d in devices if d["index"] == token or d["uuid"] == token]
    if len(selected) != 1:
        raise ValueError("cannot resolve the visible GPU uniquely for cuFFTDx preset selection")
    device = selected[0]
    family = family_for_device(device["name"], device["memory_mib"])
    return {"gpu": family, "gpu_name": device["name"], "gpu_uuid": device["uuid"],
            "compute_capability": CAPABILITIES[family]}


def descriptor(gpu):
    manifest, digest = load_manifest()
    if gpu not in manifest["gpus"]:
        raise ValueError(f"no cuFFTDx preset for {gpu}; use --cufftdx-mode all")
    return {"manifest_path": MANIFEST_RELATIVE, "manifest_sha256": digest,
            "provenance": manifest["provenance"], "gpu": gpu, "build": manifest["gpus"][gpu]["build"]}


def cell(gpu, n, precision, placement):
    manifest, _ = load_manifest()
    if gpu not in manifest["gpus"]:
        raise ValueError(f"no cuFFTDx preset for {gpu}; use --cufftdx-mode all")
    matched = [r for r in manifest["gpus"][gpu]["cells"]
               if (r["N"], r["precision"], r["placement"]) == (n, precision, placement)]
    if len(matched) != 1:
        raise ValueError(f"no unique cuFFTDx preset for {gpu}/{n}/{precision}/{placement}")
    return matched[0]


def validate_source(metadata):
    manifest, _ = load_manifest()
    for name, expected in manifest["provenance"]["candidate_source_sha256"].items():
        if metadata.get("source_sha256", {}).get(name) != expected:
            raise ValueError(f"cuFFTDx preset candidate source changed: {name}; use --cufftdx-mode all")


def validate_device(metadata):
    family = metadata.get("gpu")
    if family not in CAPABILITIES or metadata.get("compute_capability") != CAPABILITIES[family]:
        raise ValueError("unsupported native GPU/compute capability for cuFFTDx preset")
    devices = [d for d in inventory_devices(metadata.get("gpu_inventory_before", {}))
               if d["uuid"] == metadata.get("gpu_uuid")]
    if len(devices) != 1 or family_for_device(devices[0]["name"], devices[0]["memory_mib"]) != family:
        raise ValueError("native GPU UUID disagrees with cuFFTDx preset inventory")
    if metadata.get("gpu_name") != devices[0]["name"]:
        raise ValueError("native GPU name disagrees with cuFFTDx preset inventory")


def contract(metadata):
    info = metadata.get("cufftdx", {})
    if info == "not measured":
        return {"mode": "all"}
    if not isinstance(info, dict):
        raise ValueError("invalid cuFFTDx selection metadata")
    mode = info.get("mode", "all")
    if mode not in ("preset", "all"):
        raise ValueError("invalid cuFFTDx mode")
    if mode == "all":
        if "preset" in info:
            raise ValueError("all-candidate cuFFTDx metadata must not claim a preset")
        return {"mode": mode}
    expected = descriptor(metadata.get("gpu"))
    if info.get("preset") != expected:
        raise ValueError("cuFFTDx preset provenance or manifest identity mismatch")
    if metadata.get("cufftdx_build") != expected["build"]:
        raise ValueError("native cuFFTDx build is unsupported by the preset manifest")
    if metadata.get("source_sha256", {}).get(MANIFEST_RELATIVE) != expected["manifest_sha256"]:
        raise ValueError("cuFFTDx preset manifest source hash is missing or inconsistent")
    validate_source(metadata)
    validate_device(metadata)
    return {"mode": mode, "preset_sha256": expected["manifest_sha256"]}


def validate_rows(rows, metadata, n, precision, placement):
    if contract(metadata)["mode"] == "all":
        return None
    expected = cell(metadata["gpu"], n, precision, placement)
    if not rows:
        raise ValueError("cuFFTDx preset produced no timing trials")
    for row in rows:
        if (row["gpu"], int(row["N"]), row["precision"], row["placement"], row["library"], row["candidate"]) != (
                metadata["gpu"], n, precision, placement, "cuFFTDx", expected["candidate"]):
            raise ValueError("native cuFFTDx label/configuration disagrees with its preset")
        if any(int(row[key]) != value for key, value in expected["geometry"].items()):
            raise ValueError("native cuFFTDx geometry disagrees with its preset")
    return expected["candidate"]
