#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/thunderfft/build"
if [[ -n "${PYTHON:-}" ]]; then
  PY="${PYTHON}"
elif [[ -x "/home/chaewon/miniconda3/envs/fi-bench/bin/python" ]]; then
  PY="/home/chaewon/miniconda3/envs/fi-bench/bin/python"
else
  PY="python3"
fi

cmake -S "${REPO_ROOT}/thunderfft" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target thff_conv1d_fullfft4096 -j

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <flashfftconv_repo_root> [extra benchmark args]"
  exit 1
fi
FLASH_ROOT="$1"
shift

"${PY}" "${SCRIPT_DIR}/benchmark_figure5_1dfft.py" \
  --flash-root "${FLASH_ROOT}" \
  --L 4096 \
  --B 64 \
  --D 768 \
  --k-start 3 \
  --k-end 63 \
  --k-step 2 \
  --warmup 5 \
  --runs 10 \
  --ours-real-filter 1 \
  --ours-bin "${BUILD_DIR}/sample/conv1d/thff_conv1d_fullfft4096" \
  --csv "${SCRIPT_DIR}/compare_fft4096_realflash_vs_ours_complex_B64_D768_K3_63_step2.csv" \
  "$@"
