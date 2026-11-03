#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init figure2
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/plot.py" \
  --output-dir "${THUNDERFFT_RESULT_DIR}" --profile "${THUNDERFFT_PROFILE}"
