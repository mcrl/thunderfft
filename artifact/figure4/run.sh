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
if ! selected="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/selection.py" "${THUNDERFFT_FIGURE4_LIBRARIES:-auto}")"; then
  artifact_pending "Invalid THUNDERFFT_FIGURE4_LIBRARIES selection."
fi
export THUNDERFFT_FIGURE4_LIBRARIES="${selected}"
requested=",${selected},"
targets=()
if [[ "${requested}" == *,auto,* || "${requested}" == *,ThunderFFT,* ]]; then
  targets+=(thff_conv1d_fullfft4096)
fi
cudnn_args=()
flash_args=()
if [[ "${requested}" == *,auto,* || "${requested}" == *,cuDNN,* ]] && grep -q '^CUDNN_LIB:FILEPATH=/' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  targets+=(thff_conv1d_compare)
  cudnn_args=(--cudnn-binary "${THUNDERFFT_BUILD_DIR}/sample/conv1d/thff_conv1d_compare")
fi
if (( ${#targets[@]} )); then
  cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${targets[@]}" -j "${THUNDERFFT_JOBS}" \
    2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
fi
# The standalone adapter needs c10 headers only (CPU PyTorch is sufficient).
flash_python="${FLASHFFTCONV_PYTHON:-${THUNDERFFT_PYTHON}}"
if [[ ",${THUNDERFFT_FIGURE4_LIBRARIES:-auto}," == *,auto,* || ",${THUNDERFFT_FIGURE4_LIBRARIES:-auto}," == *,FlashFFTConv,* ]] && "${flash_python}" -c 'import torch' >/dev/null 2>&1; then
  "${THUNDERFFT_PYTHON}" "${TASK_DIR}/build_flash.py" --build-dir "${THUNDERFFT_BUILD_DIR}" --python "${flash_python}" \
    2>&1 | tee "${THUNDERFFT_RESULT_DIR}/flash-build.log"
  flash_args=(--flash-binary "${THUNDERFFT_BUILD_DIR}/artifact_figure4/figure4_flashfftconv")
else
  echo "FlashFFTConv not selected or unavailable: CPU PyTorch headers and FLASHFFTCONV_PYTHON select its build environment."
fi
"${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/sample/conv1d/thff_conv1d_fullfft4096" ${cudnn_args[@]+"${cudnn_args[@]}"} ${flash_args[@]+"${flash_args[@]}"} \
  --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/plot.py" --output-dir "${THUNDERFFT_RESULT_DIR}" --evaluation figure4
