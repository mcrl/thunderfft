#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table6

artifact_configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
if ! grep -qx 'THFF_CUFFTDX_ENABLED:INTERNAL=ON' "${THUNDERFFT_BUILD_DIR}/CMakeCache.txt"; then
  artifact_pending "cuFFTDx is the Table 6 baseline: configure with MATHDX_ROOT and a single THUNDERFFT_ARCHS."
fi
cmake --build "${THUNDERFFT_BUILD_DIR}" --target thff_paper_fft thff_paper_fft_cufftdx -j "${THUNDERFFT_JOBS}"
"${THUNDERFFT_PYTHON}" "${TASK_DIR}/build.py" --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --python "${FLASHFFTCONV_PYTHON:-${THUNDERFFT_PYTHON}}"
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
  --thunderfft-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft" \
  --cufftdx-binary "${THUNDERFFT_BUILD_DIR}/sample/bench/thff_paper_fft_cufftdx" \
  --table6-build "${THUNDERFFT_BUILD_DIR}/artifact_table6/build.json" \
  --build-dir "${THUNDERFFT_BUILD_DIR}" \
  --output "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
