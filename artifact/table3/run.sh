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
# Library selection can collect cuFFTDx after an earlier ThunderFFT-only part.
targets=()
cufftdx_args=()
has_cufftdx=0
if grep -qx 'THFF_CUFFTDX_ENABLED:INTERNAL=ON' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  has_cufftdx=1
elif [[ -n "${MATHDX_ROOT:-}" ]]; then
  artifact_pending "MATHDX_ROOT is set but thff_paper_fft_cufftdx was not configured; THUNDERFFT_ARCHS must name one architecture."
fi
if ! libraries="$("${THUNDERFFT_PYTHON}" - "${TASK_DIR}" "${has_cufftdx}" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from measure import selected_libraries
try:
    print("\n".join(selected_libraries(sys.argv[2] == "1")))
except ValueError as error:
    print(error, file=sys.stderr)
    raise SystemExit(65)
PY
)"; then
  artifact_pending "Invalid or unavailable THUNDERFFT_TABLE3_LIBRARIES selection; cuFFTDx requires MATHDX_ROOT."
fi
while IFS= read -r library; do
  case "${library}" in
    ThunderFFT) targets+=(thff_paper_fft) ;;
    cuFFTDx)
      targets+=(thff_paper_fft_cufftdx)
      cufftdx_args=(--cufftdx-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft_cufftdx")
      ;;
  esac
done <<< "${libraries}"
cmake --build "${THUNDERFFT_BUILD_DIR}" --target "${targets[@]}" -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft" ${cufftdx_args[@]+"${cufftdx_args[@]}"} \
  --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
