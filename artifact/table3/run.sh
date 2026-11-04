#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table3

# The executable uses local CUDA device 0; preserve scheduler visibility masks.
export THUNDERFFT_TABLE3_PREVIOUS_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export THUNDERFFT_TABLE3_PREVIOUS_CUDA_VISIBLE_DEVICES_SET="${CUDA_VISIBLE_DEVICES+x}"
if ! selected_device="$("${THUNDERFFT_PYTHON}" "${TASK_DIR}/../table5/select_device.py" "${THUNDERFFT_DEVICE}")"; then
  artifact_pending "Cannot select the requested logical CUDA device within the inherited visibility mask."
fi
export CUDA_VISIBLE_DEVICES="${selected_device}"

artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/configure.log"
# Build and time cuFFTDx whenever this build directory configured it: MathDX is
# needed only by the configure/build that set MATHDX_ROOT, not by later runs.
targets=(thff_paper_fft)
cufftdx_args=()
if grep -qx 'THFF_CUFFTDX_ENABLED:INTERNAL=ON' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  targets+=(thff_paper_fft_cufftdx)
  cufftdx_args=(--cufftdx-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft_cufftdx")
elif [[ -n "${MATHDX_ROOT:-}" ]]; then
  artifact_pending "MATHDX_ROOT is set but thff_paper_fft_cufftdx was not configured; THUNDERFFT_ARCHS must name one architecture."
fi
cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${targets[@]}" -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft" ${cufftdx_args[@]+"${cufftdx_args[@]}"} \
  --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
