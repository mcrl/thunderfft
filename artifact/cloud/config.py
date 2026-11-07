"""Explicit, non-secret cloud settings and allowlisted measurement overrides."""
import json
import math
import os
from pathlib import Path

DEFAULTS = {
    "api_key_file": "~/cloud_key/thunderfft/vast_api_key",
    "ssh_key_file": "~/cloud_key/thunderfft/id_ed25519",
    "ssh_public_key_file": "~/cloud_key/thunderfft/id_ed25519.pub",
    "budget_usd": 1.5, "reserve_usd": 0.25,
    "disk_gb": 32, "startup_seconds": 900, "execution_seconds": 600,
    "download_seconds": 120, "cleanup_seconds": 120, "billing_delay_seconds": 360,
    "upload_gb": 10, "download_gb": 1,
    "images": {"A100": "nvidia/cuda:12.9.1-devel-ubuntu22.04",
               "H100": "nvidia/cuda:13.0.2-devel-ubuntu24.04",
               "B200": "nvidia/cuda:13.0.2-devel-ubuntu24.04"},
    "environment": {},
    "mathdx": True,
}
# NVIDIA MathDx 25.06.1 (cuFFTDx headers), selected by the image's CUDA major version.
MATHDX_VERSION = "25.06.1"
MATHDX_ARCHIVES = {
    "12": ("https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz",
           "d4952f6e1a7864d29d9ec4e01c21628d84086e941cf4a3da2df1e11ba5f13f84"),
    "13": ("https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-25.06.1-cuda13.tar.gz",
           "81646046e7ae57169fcea79fb59df526eaef65e740b66e291a69182c612b8304"),
}
ARCHS = {"A100": "80", "H100": "90", "B200": "100"}
ENVIRONMENT = {
    "THUNDERFFT_TABLE3_" + suffix for suffix in
    ("BATCH", "N_LIST", "PLACEMENTS", "PRECISIONS", "REPEATS", "TIMEOUT", "TRIALS")
} | {
    "THUNDERFFT_TABLE4_" + suffix for suffix in
    ("MODE", "ITERATIONS", "TRIALS", "WARMUPS", "WAVES", "TIMEOUT")
} | {
    "THUNDERFFT_TABLE5_" + suffix for suffix in ("BATCH", "N_LIST", "PHASES", "SEED", "TIMEOUT")
} | {
    "THUNDERFFT_TABLE7_" + suffix for suffix in ("BATCH", "REPEATS", "TRIALS", "VARIANTS", "TIMING_ONLY")
}


def measurement_environment(values):
    if not isinstance(values, dict) or set(values) - ENVIRONMENT:
        raise ValueError("cloud environment accepts only documented measurement selection variables")
    result = {}
    for key, value in values.items():
        if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in "\n\r\x00"):
            raise ValueError("measurement overrides must be short, single-line strings")
        result[key] = value
    return result


def load_config(path=None):
    result = json.loads(json.dumps(DEFAULTS))
    selected = Path(path).expanduser() if path else Path("~/cloud_key/thunderfft/config.json").expanduser()
    if selected.exists() or path:
        supplied = json.loads(selected.read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            raise ValueError("unknown cloud configuration field")
        result.update(supplied)
    for field in ("api_key_file", "ssh_key_file", "ssh_public_key_file"):
        result[field] = str(Path(result[field]).expanduser().absolute())
    for field in ("budget_usd", "reserve_usd", "disk_gb", "startup_seconds", "execution_seconds",
                  "download_seconds", "cleanup_seconds", "billing_delay_seconds", "upload_gb", "download_gb"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be a positive finite number")
    for field in ("disk_gb", "startup_seconds", "execution_seconds", "download_seconds", "cleanup_seconds", "billing_delay_seconds"):
        if int(result[field]) != result[field]:
            raise ValueError(f"{field} must be an integer")
        result[field] = int(result[field])
    ceiling = 5.0
    if result["reserve_usd"] < .25:
        raise ValueError("reserve_usd must be at least 0.25")
    policy = Path("~/.config/vastai/spending-policy.json").expanduser()
    if policy.exists():
        local = json.loads(policy.read_text())
        ceiling = min(ceiling, float(local["max_total_per_execution_session"]))
        result["reserve_usd"] = max(result["reserve_usd"], float(local.get("minimum_cleanup_and_uncertainty_reserve", .25)))
    if result["budget_usd"] > ceiling:
        raise ValueError(f"budget_usd exceeds the per-execution ceiling of ${ceiling:g}")
    if result["execution_seconds"] < 60:
        raise ValueError("execution_seconds must be at least 60, including termination and archive time")
    if result["reserve_usd"] < .25 or result["billing_delay_seconds"] < 360 or result["cleanup_seconds"] < 120:
        raise ValueError("require reserve_usd >= 0.25, billing_delay_seconds >= 360, cleanup_seconds >= 120")
    if not isinstance(result["images"], dict) or set(result["images"]) != set(ARCHS):
        raise ValueError("images must specify A100, H100 and B200")
    for value in result["images"].values():
        if not isinstance(value, str) or not value or any(c.isspace() for c in value):
            raise ValueError("image must be a nonempty container tag or digest without whitespace")
    if not isinstance(result["mathdx"], bool):
        raise ValueError("mathdx must be true or false")
    result["environment"] = measurement_environment(result["environment"])
    return result


def inherited_environment(config):
    values = {key: os.environ[key] for key in ENVIRONMENT if key in os.environ}
    values.update(config["environment"])
    return measurement_environment(values)
