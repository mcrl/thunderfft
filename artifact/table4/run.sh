#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table4
case "${THUNDERFFT_TABLE4_MODE:-measure}" in
  derive)
    exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/derive.py" \
      --output-dir "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
    ;;
  measure)
    bash "${TASK_DIR}/build.sh"
    exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/measure.py" \
      --output-dir "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}" \
      --build-dir "${THUNDERFFT_BUILD_DIR}/table4" --device "${THUNDERFFT_DEVICE}"
    ;;
  *) echo "THUNDERFFT_TABLE4_MODE must be measure or derive" >&2; exit 2 ;;
esac
