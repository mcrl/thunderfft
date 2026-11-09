#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../common.sh"
artifact_init table5

case "${THUNDERFFT_PROFILE}" in
  smoke)
    TABLE5_N_LIST=64
    TABLE5_BATCH="${THUNDERFFT_TABLE5_BATCH:-64}"
    FP16_MIN=8; FP16_MAX=13
    WIDE_MIN=120; WIDE_MAX=127
    ;;
  paper)
    TABLE5_N_LIST=64,256,1024,4096
    TABLE5_BATCH="${THUNDERFFT_TABLE5_BATCH:-65536}"
    FP16_MIN=0; FP16_MAX=127
    WIDE_MIN=0; WIDE_MAX=127
    ;;
  *) echo "Unsupported THUNDERFFT_PROFILE=${THUNDERFFT_PROFILE}" >&2; exit 2 ;;
esac
TABLE5_N_LIST="${THUNDERFFT_TABLE5_N_LIST:-${TABLE5_N_LIST}}"

TABLE5_SEED="${THUNDERFFT_TABLE5_SEED:-1234}"
TABLE5_TIMEOUT="${THUNDERFFT_TABLE5_TIMEOUT:-600}"
if [[ ! "${TABLE5_BATCH}" =~ ^[1-9][0-9]*$ ]] || (( TABLE5_BATCH % 16 != 0 )); then
  echo "THUNDERFFT_TABLE5_BATCH must be a positive multiple of 16." >&2
  exit 2
fi
if [[ ! "${TABLE5_SEED}" =~ ^[0-9]+$ || ! "${TABLE5_TIMEOUT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "THUNDERFFT_TABLE5_SEED must be nonnegative and THUNDERFFT_TABLE5_TIMEOUT must be positive integer seconds." >&2
  exit 2
fi

export TABLE5_N_LIST TABLE5_BATCH TABLE5_SEED FP16_MIN FP16_MAX WIDE_MIN WIDE_MAX
export TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET="${CUDA_VISIBLE_DEVICES+x}"
export TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
# The native executable uses local CUDA device 0. Select the requested ordinal
# within the inherited mask, preserving scheduler assignments and UUID tokens.
if ! TABLE5_EFFECTIVE_CUDA_VISIBLE_DEVICES="$("${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Table 5 cannot select the requested logical CUDA device within the inherited visibility mask. See the device-selection error above."
fi
export CUDA_VISIBLE_DEVICES="${TABLE5_EFFECTIVE_CUDA_VISIBLE_DEVICES}"

artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# The cuFFTDx FP16 accuracy phase is selected by default whenever this build
# directory configured cuFFTDx (MATHDX_ROOT); otherwise it stays uncollected.
TABLE5_CUFFTDX_ENABLED=0
if grep -qx 'THFF_CUFFTDX_ENABLED:INTERNAL=ON' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  TABLE5_CUFFTDX_ENABLED=1
elif [[ -n "${MATHDX_ROOT:-}" ]]; then
  artifact_pending "MATHDX_ROOT is set but the cuFFTDx targets were not configured; THUNDERFFT_ARCHS must name one architecture."
fi
TABLE5_DEFAULT_PHASES=accuracy_reduced,accuracy_fp32_cufft
if (( TABLE5_CUFFTDX_ENABLED )); then
  TABLE5_DEFAULT_PHASES+=",accuracy_fp16_cufftdx"
fi
TABLE5_DEFAULT_PHASES+=",range_fp16,range_tf32,range_fp32_cufft"
if (( TABLE5_CUFFTDX_ENABLED )); then
  TABLE5_DEFAULT_PHASES+=",range_fp16_cufftdx"
fi
TABLE5_PHASES="${THUNDERFFT_TABLE5_PHASES:-${TABLE5_DEFAULT_PHASES}}"
if [[ ",${TABLE5_PHASES}," == *",accuracy_fp16_cufftdx,"* || ",${TABLE5_PHASES}," == *",range_fp16_cufftdx,"* ]] && (( ! TABLE5_CUFFTDX_ENABLED )); then
  artifact_pending "cuFFTDx phases require a build directory configured with MATHDX_ROOT and one THUNDERFFT_ARCHS entry."
fi
export TABLE5_PHASES TABLE5_CUFFTDX_ENABLED

"${THUNDERFFT_PYTHON}" - <<'PY'
import json
import os
import re
from pathlib import Path

raw_sizes = os.environ["TABLE5_N_LIST"]
if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", raw_sizes):
    raise SystemExit("THUNDERFFT_TABLE5_N_LIST must be comma-separated supported FFT sizes.")
sizes = [int(n) for n in raw_sizes.split(",")]
if len(set(sizes)) != len(sizes) or set(sizes) - {64, 128, 256, 512, 1024, 2048, 4096}:
    raise SystemExit("THUNDERFFT_TABLE5_N_LIST contains duplicate or unsupported FFT sizes.")
seed = int(os.environ["TABLE5_SEED"])
if seed > 2**64 - 1:
    raise SystemExit("THUNDERFFT_TABLE5_SEED must fit an unsigned 64-bit integer.")
if int(os.environ["TABLE5_BATCH"]) > 2**31 - 1:
    raise SystemExit("THUNDERFFT_TABLE5_BATCH must fit a positive signed 32-bit integer.")
phases = os.environ["TABLE5_PHASES"].split(",")
allowed_phases = {"accuracy_reduced", "accuracy_fp32_cufft", "range_fp16", "range_tf32", "range_fp32_cufft", "accuracy_fp16_cufftdx", "range_fp16_cufftdx"}
if not phases or len(set(phases)) != len(phases) or set(phases) - allowed_phases:
    raise SystemExit("THUNDERFFT_TABLE5_PHASES must select unique comma-separated Table 5 phase names.")
metadata = {
    "evaluation": "table5",
    "profile": os.environ["THUNDERFFT_PROFILE"],
    "N": sizes,
    "selected_phases": phases,
    "range_mode": os.environ.get("THUNDERFFT_TABLE5_RANGE_MODE", "boundary"),
    "range_checks": {},
    "batch": int(os.environ["TABLE5_BATCH"]),
    "seed": seed,
    "overrides": {key: os.environ[key] for key in ("THUNDERFFT_TABLE5_BATCH", "THUNDERFFT_TABLE5_SEED", "THUNDERFFT_TABLE5_N_LIST", "THUNDERFFT_TABLE5_PHASES") if key in os.environ},
    "complete_paper_size_set": set(sizes) == {64, 256, 1024, 4096},
    "input": "independent Gaussian real/imaginary parts, seed+b for FFT b; each FFT scaled so ||x||_2/sqrt(N) = alpha; stored as FP32",
    "reference": "cuFFT Z2Z FP64 of the FP32 input",
    "implementations": {"fp16": "ThunderFFT", "tf32": "ThunderFFT", "fp32": "cuFFT C2C",
                        "fp16_cufftdx": "cuFFTDx (ElementsPerThread 8, FFTsPerBlock 2)"},
    "cufftdx_configured_in_build": os.environ["TABLE5_CUFFTDX_ENABLED"] == "1",
    "rel_l2": "over all outputs of the batch",
    "paper_conditions": {
        "batch": "65536 (Section 6.1)",
        "not_stated": ["seed", "real/imaginary construction", "normalization scope", "reference input",
                       "rel_l2 aggregation", "exponent scan window", "cuFFTDx parameters"],
        "cufftdx_source": "revised paper text, not in the 2026-10-13 PDF",
    },
    "range_exponents": {
        "fp16": [int(os.environ["FP16_MIN"]), int(os.environ["FP16_MAX"])],
        "fp16_cufftdx": [int(os.environ["FP16_MIN"]), int(os.environ["FP16_MAX"])],
        "tf32": [int(os.environ["WIDE_MIN"]), int(os.environ["WIDE_MAX"])],
        "fp32": [int(os.environ["WIDE_MIN"]), int(os.environ["WIDE_MAX"])],
    },
    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
    "requested_logical_device": int(os.environ["THUNDERFFT_DEVICE"]),
    "previous_CUDA_VISIBLE_DEVICES_was_set": bool(os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET"]),
    "previous_CUDA_VISIBLE_DEVICES": os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES"] if os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET"] else None,
    "scientific_validation": "not_assessed",
    "paper_comparison": "Informational comparison only; paper_conditions lists the Table 5 conditions the paper does not state.",
}
path = Path(os.environ["THUNDERFFT_RESULT_DIR"]) / "metadata.json"
path.write_text(json.dumps(metadata, indent=2) + "\n")
PY

TABLE5_TARGETS=(benchmark_table5_precision)
if (( TABLE5_CUFFTDX_ENABLED )); then
  TABLE5_TARGETS+=(benchmark_table5_precision_cufftdx)
fi
printf '%q ' "${THUNDERFFT_CMAKE}" --build "${THUNDERFFT_BUILD_DIR}" --target "${TABLE5_TARGETS[@]}" -j "${THUNDERFFT_JOBS}" >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
printf '\n' >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${TABLE5_TARGETS[@]}" -j "${THUNDERFFT_JOBS}"
TABLE5_BIN="${THUNDERFFT_BUILD_DIR}/sample/bench/benchmark_table5_precision"
TABLE5_CUFFTDX_BIN="${THUNDERFFT_BUILD_DIR}/sample/bench/benchmark_table5_precision_cufftdx"
TABLE5_PROVENANCE_BINARIES=("${TABLE5_BIN}")
if (( TABLE5_CUFFTDX_ENABLED )); then
  TABLE5_PROVENANCE_BINARIES+=("${TABLE5_CUFFTDX_BIN}")
fi
"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/provenance.py" "${THUNDERFFT_RESULT_DIR}" "${THUNDERFFT_ROOT}" "${THUNDERFFT_BUILD_DIR}" "${TABLE5_PROVENANCE_BINARIES[@]}"
command -v timeout >/dev/null

"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/measure.py" --output "${THUNDERFFT_RESULT_DIR}" \
  --binary "${TABLE5_BIN}" --cufftdx-binary "${TABLE5_CUFFTDX_BIN}" --timeout "${TABLE5_TIMEOUT}"

"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/summarize.py" "${THUNDERFFT_RESULT_DIR}"
echo "Selected Table 5 phases completed and passed numerical sanity checks; unstated paper conditions remain unassessed."
