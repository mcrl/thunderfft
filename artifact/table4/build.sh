#!/usr/bin/env bash
# Compile the MMA-only benchmark; this script launches no GPU workload.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${TASK_DIR}/../common.sh"
artifact_init table4
TABLE4_BUILD="${THUNDERFFT_BUILD_DIR}/table4"
args=(-S "${TASK_DIR}" -B "${TABLE4_BUILD}" -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "-DCMAKE_CUDA_ARCHITECTURES=${THUNDERFFT_ARCHS:-80}")
if [[ -n "${CUDACXX:-${NVCC:-}}" ]]; then
  args+=("-DCMAKE_CUDA_COMPILER:FILEPATH=${CUDACXX:-${NVCC}}")
fi
if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
  args+=("-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
fi
cmake "${args[@]}" 2>&1 | tee "${THUNDERFFT_RESULT_DIR}/configure.log"
cmake --build "${TABLE4_BUILD}" --target thff_ideal_mma -j "${THUNDERFFT_JOBS}" \
  2>&1 | tee "${THUNDERFFT_RESULT_DIR}/build.log"
