#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init figure4

# The executable uses local CUDA device 0; preserve scheduler visibility masks.
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"

artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/configure.log"
# cuDNN is measured whenever CMake found it (sample/conv1d builds thff_conv1d_compare).
targets=(thff_conv1d_fullfft4096)
cudnn_args=()
if grep -q '^CUDNN_LIB:FILEPATH=/' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  targets+=(thff_conv1d_compare)
  cudnn_args=(--cudnn-binary "${THUNDERFFT_BUILD_DIR}/sample/conv1d/thff_conv1d_compare")
fi
cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${targets[@]}" -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
"${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/sample/conv1d/thff_conv1d_fullfft4096" ${cudnn_args[@]+"${cudnn_args[@]}"} \
  --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/plot.py" --output-dir "${THUNDERFFT_RESULT_DIR}" --evaluation figure4
