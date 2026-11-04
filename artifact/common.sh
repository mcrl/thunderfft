#!/usr/bin/env bash
# Shared adapter helpers. The root runner supplies a unique result directory.

artifact_init() {
  local evaluation="$1"
  local artifact_dir
  artifact_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  export THUNDERFFT_ROOT="${THUNDERFFT_ROOT:-$(cd -- "${artifact_dir}/.." && pwd)}"
  export THUNDERFFT_BUILD_DIR="${THUNDERFFT_BUILD_DIR:-${THUNDERFFT_ROOT}/build_artifact}"
  export THUNDERFFT_PROFILE="${THUNDERFFT_PROFILE:-smoke}"
  export THUNDERFFT_DEVICE="${THUNDERFFT_DEVICE:-0}"
  export THUNDERFFT_JOBS="${THUNDERFFT_JOBS:-2}"
  export THUNDERFFT_PYTHON="${THUNDERFFT_PYTHON:-${PYTHON:-python3}}"
  export THUNDERFFT_EVALUATION="${evaluation}"
  if [[ -z "${THUNDERFFT_RESULT_DIR:-}" ]]; then
    mkdir -p -- "${THUNDERFFT_ROOT}/result/${evaluation}"
    THUNDERFFT_RESULT_DIR="$(mktemp -d "${THUNDERFFT_ROOT}/result/${evaluation}/$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
    export THUNDERFFT_RESULT_DIR
  fi
  mkdir -p -- "${THUNDERFFT_RESULT_DIR}"
}

artifact_configure() {
  local args=(-S "${THUNDERFFT_ROOT}" -B "${THUNDERFFT_BUILD_DIR}"
              "-DTHUNDERFFT_ARCHS=${THUNDERFFT_ARCHS:-80}")
  if [[ -n "${CUDACXX:-}" ]]; then
    args+=("-DCMAKE_CUDA_COMPILER=${CUDACXX}")
  elif [[ -n "${NVCC:-}" ]]; then
    args+=("-DCMAKE_CUDA_COMPILER=${NVCC}")
  fi
  if [[ -n "${CUDAToolkit_ROOT:-}" ]]; then
    args+=("-DCUDAToolkit_ROOT=${CUDAToolkit_ROOT}")
  fi
  # MATHDX_ROOT is a build-time input stored in the CMake cache; an explicitly
  # empty value clears it.
  if [[ -n "${MATHDX_ROOT+x}" ]]; then
    args+=("-DMATHDX_ROOT=${MATHDX_ROOT}")
  fi
  cmake "${args[@]}" "$@"
}

artifact_pending() {
  printf 'PENDING [%s]: %s\n' "${THUNDERFFT_EVALUATION:-evaluation}" "$*" >&2
  exit 78
}
