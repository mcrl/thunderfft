#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table2
exec "${THUNDERFFT_PYTHON}" "${TASK_DIR}/model.py" --output "${THUNDERFFT_RESULT_DIR}"
