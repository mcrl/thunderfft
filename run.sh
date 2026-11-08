#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# setup.sh prepares this per-account environment. Explicit Python selections
# (including cloud workers) already supply their own isolated environment.
if [[ -z "${THUNDERFFT_PYTHON:-${PYTHON:-}}" && -f "${HOME}/.thunderfft_env" ]]; then
  source "${HOME}/.thunderfft_env"
fi
exec "${THUNDERFFT_PYTHON:-${PYTHON:-python3}}" "${REPO_ROOT}/artifact/runner.py" "$@"
