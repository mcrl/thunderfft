"""Explicit, non-secret cloud settings and allowlisted measurement overrides."""
import json
import math
import os
from pathlib import Path

DEFAULTS = {
    "api_key_file": "~/cloud_key/thunderfft/vast_api_key",
    "ssh_key_file": "~/cloud_key/thunderfft/id_ed25519",
    "ssh_public_key_file": "~/cloud_key/thunderfft/id_ed25519.pub",
    "budget_usd": 50.0, "reserve_usd": 0.25,
    "disk_gb": 32, "startup_seconds": 900, "execution_seconds": 3600,
    "download_seconds": 120, "cleanup_seconds": 120, "billing_delay_seconds": 360,
    "upload_gb": 10, "download_gb": 1,
    "exclude_machine_ids": [], "max_startup_attempts": 2, "ssh_auth_failure_seconds": 90,
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
# NVIDIA's redistrib_9.12.0.json, linux-x86_64; checked 2026-11-07.
CUDNN_ARCHIVES = {
    "12": ("https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda12-archive.tar.xz",
           "21fad1b3a8a3db018868184b2082bd492633ae5101752432d655782bd108b19e"),
    "13": ("https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.12.0.46_cuda13-archive.tar.xz",
           "ce92b4058d95c85959b822baa56389126a31678f5068b486523d1aa7f6490fcb"),
}
TORCH_HEADERS_VERSION = "2.8.0"
ARCHS = {"A100": "80", "H100": "90", "B200": "100"}
CUFFTDX_MODE_ENV = "THUNDERFFT_TABLE3_CUFFTDX_MODE"
CUFFTDX_MODES = ("preset", "all")
CUDNN_MODE_ENV = "THUNDERFFT_CUDNN_MODE"
CUDNN_MODES = ("fast", "full")
TABLE5_RANGE_MODE_ENV = "THUNDERFFT_TABLE5_RANGE_MODE"
TABLE5_RANGE_MODES = ("boundary", "full")
ENVIRONMENT = {
    "THUNDERFFT_TABLE3_" + suffix for suffix in
    ("BATCH", "N_LIST", "PLACEMENTS", "PRECISIONS", "REPEATS", "TIMEOUT", "CUFFTDX_TIMEOUT", "CUFFTDX_MODE", "TRIALS", "LIBRARIES")
} | {
    "THUNDERFFT_TABLE4_" + suffix for suffix in
    ("MODE", "ITERATIONS", "TRIALS", "WARMUPS", "WAVES", "TIMEOUT")
} | {
    "THUNDERFFT_TABLE5_" + suffix for suffix in ("BATCH", "N_LIST", "PHASES", "SEED", "TIMEOUT", "RANGE_MODE")
} | {
    "THUNDERFFT_TABLE7_" + suffix for suffix in ("BATCH", "REPEATS", "TRIALS", "VARIANTS", "TIMING_ONLY")
} | {
    "THUNDERFFT_TABLE6_" + suffix for suffix in ("BATCH", "REPEATS", "TRIALS", "TIMEOUT", "LIBRARIES")
} | {
    "THUNDERFFT_FIGURE3_" + suffix for suffix in
    ("BATCH", "N_LIST", "PRECISIONS", "REPEATS", "TIMEOUT", "TRIALS", "LIBRARIES")
} | {
    "THUNDERFFT_FIGURE" + figure + "_" + suffix
    for figure in ("4", "5") for suffix in ("FILTERS", "TRIALS", "TIMEOUT", "LIBRARIES", "SLOW_RATIO")
} | {
    "THUNDERFFT_FIGURE5_VALIDATE_CHANNELS", "THUNDERFFT_FIGURE4_N", CUDNN_MODE_ENV,
}


def required_dependencies(jobs, *, mathdx=True):
    """Install only dependencies of the requested evaluation/library subsets."""
    needed = {"mathdx": False, "cudnn": False, "torch_headers": False}
    for job in jobs:
        environment = job.get("environment", {})
        for evaluation in job["evaluations"]:
            dependency_evaluation = "table3" if evaluation == "figure2" else evaluation
            value = environment.get("THUNDERFFT_" + dependency_evaluation.upper() + "_LIBRARIES", "all")
            libraries = {item.strip().lower() for item in value.split(",")}
            selected = lambda library: bool(libraries & {"all", "auto", library})
            if evaluation == "table5":
                needed["mathdx"] = mathdx
            elif evaluation in ("table3", "table6", "figure2", "figure3", "figure4") and selected("cufftdx"):
                needed["mathdx"] |= mathdx
            if evaluation in ("figure5", "figure4") and selected("cudnn"):
                needed["cudnn"] = True
            if evaluation in ("table6", "figure5") and selected("flashfftconv"):
                needed["torch_headers"] = True
    return needed


def measurement_environment(values):
    if not isinstance(values, dict) or set(values) - ENVIRONMENT:
        raise ValueError("cloud environment accepts only documented measurement selection variables")
    result = {}
    for key, value in values.items():
        if not isinstance(value, str) or len(value) > 1024 or any(c in value for c in "\n\r\x00"):
            raise ValueError("measurement overrides must be short, single-line strings")
        result[key] = value
    if CUFFTDX_MODE_ENV in result:
        cufftdx_mode(environment=result)
    if CUDNN_MODE_ENV in result:
        cudnn_mode(environment=result)
    if TABLE5_RANGE_MODE_ENV in result:
        table5_range_mode(environment=result)
    return result


def cufftdx_mode(explicit=None, environment=None):
    """CLI overrides configured/inherited selection; an absent value uses preset."""
    values = os.environ if environment is None else environment
    value = explicit if explicit is not None else values.get(CUFFTDX_MODE_ENV, "preset")
    if value not in CUFFTDX_MODES:
        raise ValueError(f"{CUFFTDX_MODE_ENV} must be preset or all")
    return value


def cudnn_mode(explicit=None, environment=None):
    """Figure 4/5 cuDNN defaults to fast; explicit CLI overrides configured values."""
    values = os.environ if environment is None else environment
    value = explicit if explicit is not None else values.get(CUDNN_MODE_ENV, "fast")
    if value not in CUDNN_MODES:
        raise ValueError(f"{CUDNN_MODE_ENV} must be fast or full")
    return value


def table5_range_mode(explicit=None, environment=None):
    """Table 5 defaults to boundary checks; explicit CLI overrides configured values."""
    values = os.environ if environment is None else environment
    value = explicit if explicit is not None else values.get(TABLE5_RANGE_MODE_ENV, "boundary")
    if value not in TABLE5_RANGE_MODES:
        raise ValueError(f"{TABLE5_RANGE_MODE_ENV} must be boundary or full")
    return value


def supplied_config(path=None):
    """Read non-secret settings only; suitable for an offline selection preview."""
    selected = Path(path).expanduser() if path else Path("~/cloud_key/thunderfft/config.json").expanduser()
    if selected.exists() or path:
        supplied = json.loads(selected.read_text())
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
            raise ValueError("unknown cloud configuration field")
        return supplied
    return {}


def preview_cufftdx_mode(path=None, explicit=None):
    return preview_modes(path, explicit)["cufftdx_mode"]


def preview_modes(path=None, explicit_cufftdx=None, explicit_cudnn=None, explicit_table5_range=None):
    configured = measurement_environment(supplied_config(path).get("environment", {}))
    environment = inherited_environment({"environment": configured})
    return {"cufftdx_mode": cufftdx_mode(explicit_cufftdx, environment),
            "cudnn_mode": cudnn_mode(explicit_cudnn, environment),
            "table5_range_mode": table5_range_mode(explicit_table5_range, environment)}


def load_config(path=None):
    result = json.loads(json.dumps(DEFAULTS))
    result.update(supplied_config(path))
    for field in ("api_key_file", "ssh_key_file", "ssh_public_key_file"):
        result[field] = str(Path(result[field]).expanduser().absolute())
    for field in ("budget_usd", "reserve_usd", "disk_gb", "startup_seconds", "execution_seconds",
                  "download_seconds", "cleanup_seconds", "billing_delay_seconds", "upload_gb", "download_gb",
                  "max_startup_attempts", "ssh_auth_failure_seconds"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be a positive finite number")
    for field in ("disk_gb", "startup_seconds", "execution_seconds", "download_seconds", "cleanup_seconds", "billing_delay_seconds",
                  "max_startup_attempts", "ssh_auth_failure_seconds"):
        if int(result[field]) != result[field]:
            raise ValueError(f"{field} must be an integer")
        result[field] = int(result[field])
    if result["max_startup_attempts"] > 3:
        raise ValueError("max_startup_attempts must be between 1 and 3; all attempts share one budget")
    excluded = result["exclude_machine_ids"]
    if (not isinstance(excluded, list) or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                                            for value in excluded)):
        raise ValueError("exclude_machine_ids must be a list of positive integer machine IDs")
    result["exclude_machine_ids"] = sorted(set(excluded))
    ceiling = 50.0
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
