#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table7
# Check source availability before building or launching any GPU workload.
"${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" --check-selection \
  --profile "${THUNDERFFT_PROFILE}" --output "${THUNDERFFT_RESULT_DIR}"
bash "${TASK_DIR}/build.sh"
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export TABLE7_PREVIOUS_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export CUDA_VISIBLE_DEVICES="${selected_device}"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/table7/thff_ablation" \
  --build-dir "${THUNDERFFT_BUILD_DIR}/table7" \
  --profile "${THUNDERFFT_PROFILE}" --output "${THUNDERFFT_RESULT_DIR}"
