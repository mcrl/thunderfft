#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table6

if ! selection="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/selection.py")"; then
  exit 2
fi
mapfile -t selected <<< "${selection}"
# All drivers use logical CUDA device zero.
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"
targets=()
measure_args=()
external=()
need_cufftdx=0
for library in "${selected[@]}"; do
  case "${library}" in
    thunderfft) targets+=(thff_paper_fft); measure_args+=(--thunderfft-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft") ;;
    cufftdx) need_cufftdx=1; targets+=(thff_paper_fft_cufftdx); measure_args+=(--cufftdx-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft_cufftdx") ;;
    tcfft|flashfftconv) external+=("${library}") ;;
  esac
done
if (( ${#targets[@]} )); then
  artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  if (( need_cufftdx )) && \
     ! grep -qx 'THFF_CUFFTDX_ENABLED:INTERNAL=ON' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
    artifact_pending "Selected cuFFTDx requires MATHDX_ROOT and a single THUNDERFFT_ARCHS; collect other libraries separately with THUNDERFFT_TABLE6_LIBRARIES."
  fi
  cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${targets[@]}" -j "${THUNDERFFT_JOBS}"
fi
if (( ${#external[@]} )); then
  printf -v external_csv '%s,' "${external[@]}"
  "${THUNDERFFT_PYTHON}" "${TASK_DIR}/build.py" --build-dir "${THUNDERFFT_BUILD_DIR}" \
    --libraries "${external_csv%,}" --python "${FLASHFFTCONV_PYTHON:-${THUNDERFFT_PYTHON}}"
  measure_args+=(--table6-build "${THUNDERFFT_BUILD_DIR}/artifact_table6/build.json")
fi
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" "${measure_args[@]}" \
  --build-dir "${THUNDERFFT_BUILD_DIR}" --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
