#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init figure2
artifact_pending "Implementation pending. See artifact/figure2/README.md for the assigned work and output contract."
