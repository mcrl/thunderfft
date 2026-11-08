#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init figure5

# The executable uses local CUDA device 0; preserve scheduler visibility masks.
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"

FIGURE5_BUILD="${THUNDERFFT_BUILD_DIR}/figure5"
arch="${THUNDERFFT_ARCHS:-80}"
args=(-S "${TASK_DIR}" -B "${FIGURE5_BUILD}" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
      "-DCMAKE_CUDA_ARCHITECTURES=${arch%%[,;]*}")
if [[ -n "${CUDACXX:-${NVCC:-}}" ]]; then args+=("-DCMAKE_CUDA_COMPILER=${CUDACXX:-${NVCC}}"); fi
if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then args+=("-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}"); fi
if [[ -n "${MATHDX_ROOT+x}" ]]; then args+=("-DMATHDX_ROOT=${MATHDX_ROOT}"); fi
if [[ -n "${CUDNN_HOME:-}" ]]; then args+=("-DCUDNN_ROOT=${CUDNN_HOME}"); fi
if [[ -n "${CUDNN_ROOT:-}" ]]; then args+=("-DCUDNN_ROOT=${CUDNN_ROOT}"); fi
cmake "${args[@]}" 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/configure.log"
available="cuFFT,ThunderFFT"
if grep -qx 'THFF_CONV2D_CUFFTDX:INTERNAL=ON' "${FIGURE5_BUILD}/CMakeCache.txt"; then
  available="${available},cuFFTDx"
fi
if grep -qx 'THFF_CONV2D_CUDNN:INTERNAL=ON' "${FIGURE5_BUILD}/CMakeCache.txt"; then
  available="${available},cuDNN"
fi
cmake --build "${FIGURE5_BUILD}" --target thff_conv2d -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
"${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${FIGURE5_BUILD}/thff_conv2d" --available "${available}" \
  --build-dir "${FIGURE5_BUILD}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/../figure4/plot.py" --output-dir "${THUNDERFFT_RESULT_DIR}" --evaluation figure5
