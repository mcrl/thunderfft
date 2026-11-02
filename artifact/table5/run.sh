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
    if [[ -z "${THUNDERFFT_TABLE5_BATCH:-}" ]]; then
      artifact_pending "Table 5 paper profile requires explicit THUNDERFFT_TABLE5_BATCH. The source default is 65536; confirm the author's sample population and available host/GPU memory first. FP32 currently uses cuFFT, so its paper provenance also needs confirmation."
    fi
    TABLE5_N_LIST=64,256,1024,4096
    TABLE5_BATCH="${THUNDERFFT_TABLE5_BATCH}"
    FP16_MIN=0; FP16_MAX=127
    WIDE_MIN=0; WIDE_MAX=127
    ;;
  *) echo "Unsupported THUNDERFFT_PROFILE=${THUNDERFFT_PROFILE}" >&2; exit 2 ;;
esac

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

"${THUNDERFFT_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

metadata = {
    "evaluation": "table5",
    "profile": os.environ["THUNDERFFT_PROFILE"],
    "N": [int(n) for n in os.environ["TABLE5_N_LIST"].split(",")],
    "batch": int(os.environ["TABLE5_BATCH"]),
    "seed": int(os.environ["TABLE5_SEED"]),
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
    "remaining_work": ["Confirm paper FP32 implementation provenance", "Confirm paper batch, seed, hardware, and expected tolerances"],
}
path = Path(os.environ["THUNDERFFT_RESULT_DIR"]) / "metadata.json"
path.write_text(json.dumps(metadata, indent=2) + "\n")
PY

artifact_configure
cmake --build "${THUNDERFFT_BUILD_DIR}" --target benchmark_table5_precision -j "${THUNDERFFT_JOBS}"
TABLE5_BIN="${THUNDERFFT_BUILD_DIR}/sample/bench/benchmark_table5_precision"
command -v timeout >/dev/null

run_measurement() {
  local name="$1"
  shift
  local args=("${TABLE5_BIN}" --N-list "${TABLE5_N_LIST}" --batch "${TABLE5_BATCH}"
    --mode gaussian --seed "${TABLE5_SEED}" "$@" --csv "${THUNDERFFT_RESULT_DIR}/${name}.csv")
  printf '%q ' timeout --signal=TERM --kill-after=10s "${TABLE5_TIMEOUT}s" "${args[@]}" >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
  printf '\n' >> "${THUNDERFFT_RESULT_DIR}/commands.txt"
  timeout --signal=TERM --kill-after=10s "${TABLE5_TIMEOUT}s" "${args[@]}" 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/${name}.log"
}

run_measurement accuracy_reduced --precision both --alpha 1 --reference fp64
run_measurement accuracy_fp32_cufft --precision fp32 --alpha 1 --reference fp64
run_measurement range_fp16 --precision fp16 --alpha-sweep --alpha-exp-min "${FP16_MIN}" --alpha-exp-max "${FP16_MAX}" --fast-finite-only
run_measurement range_tf32 --precision tf32 --alpha-sweep --alpha-exp-min "${WIDE_MIN}" --alpha-exp-max "${WIDE_MAX}" --fast-finite-only
run_measurement range_fp32_cufft --precision fp32 --alpha-sweep --alpha-exp-min "${WIDE_MIN}" --alpha-exp-max "${WIDE_MAX}" --fast-finite-only

"${THUNDERFFT_PYTHON}" "${SCRIPT_DIR}/summarize.py" "${THUNDERFFT_RESULT_DIR}"
echo "Table 5 measurements and numerical sanity checks completed; paper provenance remains unassessed."
