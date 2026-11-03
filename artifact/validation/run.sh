#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init validation
export THUNDERFFT_VALIDATION_PREVIOUS_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export THUNDERFFT_VALIDATION_PREVIOUS_CUDA_VISIBLE_DEVICES_SET="${CUDA_VISIBLE_DEVICES+x}"
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"
artifact_configure -DTHUNDERFFT_BUILD_VALIDATION=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/configure.log"
cmake --build "${THUNDERFFT_BUILD_DIR}" --target thff_validate -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/evaluate.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/sample/validation/thff_validate" \
  --build-dir "${THUNDERFFT_BUILD_DIR}" --output "${THUNDERFFT_RESULT_DIR}" \
  --profile "${THUNDERFFT_PROFILE}"
