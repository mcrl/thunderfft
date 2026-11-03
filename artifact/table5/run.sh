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
TABLE5_PHASES="${THUNDERFFT_TABLE5_PHASES:-accuracy_reduced,accuracy_fp32_cufft,range_fp16,range_tf32,range_fp32_cufft}"

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

export TABLE5_N_LIST TABLE5_BATCH TABLE5_SEED TABLE5_PHASES FP16_MIN FP16_MAX WIDE_MIN WIDE_MAX
export TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET="${CUDA_VISIBLE_DEVICES+x}"
export TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
# The native executable uses local CUDA device 0. Select the requested ordinal
# within the inherited mask, preserving scheduler assignments and UUID tokens.
if ! TABLE5_EFFECTIVE_CUDA_VISIBLE_DEVICES="$("${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Table 5 cannot select the requested logical CUDA device within the inherited visibility mask. See the device-selection error above."
fi
export CUDA_VISIBLE_DEVICES="${TABLE5_EFFECTIVE_CUDA_VISIBLE_DEVICES}"

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
allowed_phases = {"accuracy_reduced", "accuracy_fp32_cufft", "range_fp16", "range_tf32", "range_fp32_cufft"}
if not phases or len(set(phases)) != len(phases) or set(phases) - allowed_phases:
    raise SystemExit("THUNDERFFT_TABLE5_PHASES must select unique comma-separated Table 5 phase names.")
metadata = {
    "evaluation": "table5",
    "profile": os.environ["THUNDERFFT_PROFILE"],
    "N": sizes,
    "selected_phases": phases,
    "batch": int(os.environ["TABLE5_BATCH"]),
    "seed": seed,
    "population_provenance": "Operational defaults are batch=65536 and seed=1234 in the retained evaluator source; the supplied paper does not establish its exact batch/seed.",
    "overrides": {key: os.environ[key] for key in ("THUNDERFFT_TABLE5_BATCH", "THUNDERFFT_TABLE5_SEED", "THUNDERFFT_TABLE5_N_LIST", "THUNDERFFT_TABLE5_PHASES") if key in os.environ},
    "complete_paper_size_set": set(sizes) == {64, 256, 1024, 4096},
    "input": "complex Gaussian; per-FFT L2 norm / sqrt(N) = alpha",
    "reference": "cuFFT Z2Z (FP64) applied to the generated FP32 input",
    "implementations": {"fp16": "ThunderFFT", "tf32": "ThunderFFT", "fp32": "cuFFT C2C"},
    "range_exponents": {
        "fp16": [int(os.environ["FP16_MIN"]), int(os.environ["FP16_MAX"])],
        "tf32": [int(os.environ["WIDE_MIN"]), int(os.environ["WIDE_MAX"])],
        "fp32": [int(os.environ["WIDE_MIN"]), int(os.environ["WIDE_MAX"])],
    },
    "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
    "requested_logical_device": int(os.environ["THUNDERFFT_DEVICE"]),
    "previous_CUDA_VISIBLE_DEVICES_was_set": bool(os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET"]),
    "previous_CUDA_VISIBLE_DEVICES": os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES"] if os.environ["TABLE5_PREVIOUS_CUDA_VISIBLE_DEVICES_SET"] else None,
    "scientific_validation": "not_assessed",
    "paper_comparison": "Informational comparison only; published input population and FP32 implementation provenance are not established.",
}
path = Path(os.environ["THUNDERFFT_RESULT_DIR"]) / "metadata.json"
path.write_text(json.dumps(metadata, indent=2) + "\n")
PY

artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
printf '%q ' cmake --build "${THUNDERFFT_BUILD_DIR}" --target benchmark_table5_precision -j "${THUNDERFFT_JOBS}" >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
printf '\n' >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
cmake --build "${THUNDERFFT_BUILD_DIR}" --target benchmark_table5_precision -j "${THUNDERFFT_JOBS}"
TABLE5_BIN="${THUNDERFFT_BUILD_DIR}/sample/bench/benchmark_table5_precision"
"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/provenance.py" "${THUNDERFFT_RESULT_DIR}" "${THUNDERFFT_ROOT}" "${THUNDERFFT_BUILD_DIR}" "${TABLE5_BIN}"
command -v timeout >/dev/null

run_measurement() {
  local name="$1"
  shift
  local args=("${TABLE5_BIN}" --N-list "${TABLE5_N_LIST}" --batch "${TABLE5_BATCH}"
    --mode gaussian --seed "${TABLE5_SEED}" "$@" --csv "${THUNDERFFT_RESULT_DIR}/${name}.csv"
    --environment-json "${THUNDERFFT_RESULT_DIR}/${name}.environment.json")
  printf '%q ' timeout --signal=TERM --kill-after=10s "${TABLE5_TIMEOUT}s" "${args[@]}" >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
  printf '\n' >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
  timeout --signal=TERM --kill-after=10s "${TABLE5_TIMEOUT}s" "${args[@]}" 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/${name}.log"
}

IFS=',' read -r -a selected_phases <<< "${TABLE5_PHASES}"
for phase in "${selected_phases[@]}"; do
  case "${phase}" in
    accuracy_reduced) run_measurement "${phase}" --precision both --alpha 1 --reference fp64 ;;
    accuracy_fp32_cufft) run_measurement "${phase}" --precision fp32 --alpha 1 --reference fp64 ;;
    range_fp16) run_measurement "${phase}" --precision fp16 --alpha-sweep --alpha-exp-min "${FP16_MIN}" --alpha-exp-max "${FP16_MAX}" --fast-finite-only ;;
    range_tf32) run_measurement "${phase}" --precision tf32 --alpha-sweep --alpha-exp-min "${WIDE_MIN}" --alpha-exp-max "${WIDE_MAX}" --fast-finite-only ;;
    range_fp32_cufft) run_measurement "${phase}" --precision fp32 --alpha-sweep --alpha-exp-min "${WIDE_MIN}" --alpha-exp-max "${WIDE_MAX}" --fast-finite-only ;;
  esac
done

"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/summarize.py" "${THUNDERFFT_RESULT_DIR}"
echo "Selected Table 5 phases completed and passed numerical sanity checks; paper provenance remains unassessed."
