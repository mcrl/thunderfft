#!/usr/bin/env bash
# Per-account environment for the artifact: Python venv, CMake, MathDx, cuDNN,
# and a ~/.thunderfft_env file sourced from ~/.bashrc. Installs only under
# $HOME, needs no root, and is safe to run again.
#
#   bash artifact/setup.sh && source ~/.thunderfft_env
#
# Overrides: CUDA_HOME, THUNDERFFT_ARCHS, THUNDERFFT_VENV,
# THUNDERFFT_MATHDX_ARCHIVE and THUNDERFFT_CUDNN_ARCHIVE (local .tar.gz/.tar.xz
# or an extracted directory used instead of downloading).
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${THUNDERFFT_VENV:-${HOME}/venv}"
ENV_FILE="${HOME}/.thunderfft_env"

MATHDX_VERSION=25.06.1

step() { printf '\n== %s\n' "$*"; }
die() { printf 'setup: %s\n' "$*" >&2; exit 1; }

# Fetch (or copy) an archive, check its SHA-256 and unpack into the selected target.
install_archive() {
  local name="$1" url="$2" sha="$3" local_source="$4" target="$5"
  if [[ -d "${local_source}" ]]; then
    mkdir -p -- "${target}"
    cp -a -- "${local_source}/." "${target}/"
    return
  fi
  local tmp
  tmp="$(mktemp -d)"
  trap 'rm -rf -- "${tmp}"' RETURN
  local archive="${tmp}/${url##*/}"
  if [[ -n "${local_source}" ]]; then
    cp -- "${local_source}" "${archive}"
  else
    echo "downloading ${name}: ${url}"
    curl -fsSL --retry 3 -o "${archive}" "${url}"
  fi
  echo "${sha}  ${archive}" | sha256sum -c - >/dev/null || die "${name}: SHA-256 mismatch"
  mkdir -p -- "${target}"
  tar -xf "${archive}" -C "${target}" --strip-components=1
}

step "Python >= 3.9"
python=""
for candidate in python3.13 python3.12 python3.11 python3.10 python3.9 python3; do
  if command -v "${candidate}" >/dev/null &&
     "${candidate}" -c 'import sys, venv; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    python="$(command -v "${candidate}")"
    break
  fi
done
[[ -n "${python}" ]] || die "no python3.9 or newer with the venv module"
[[ -x "${VENV}/bin/python" ]] || "${python}" -m venv "${VENV}"
"${VENV}/bin/python" --version

step "Python packages (matplotlib, CMake, PyTorch headers for Table 6)"
pip=("${VENV}/bin/python" -m pip --disable-pip-version-check -q)
"${pip[@]}" install --upgrade pip
"${pip[@]}" install -r "${ROOT}/artifact/requirements.txt" "cmake>=3.24,<4"
"${VENV}/bin/python" -c 'import torch' 2>/dev/null ||
  "${pip[@]}" install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
"${VENV}/bin/cmake" --version | head -n 1

step "CUDA toolkit"
if [[ -z "${CUDA_HOME:-}" ]]; then
  for candidate in /usr/local/cuda-12.9 /usr/local/cuda; do
    [[ -x "${candidate}/bin/nvcc" ]] && { CUDA_HOME="${candidate}"; break; }
  done
fi
[[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || die "nvcc not found; set CUDA_HOME"
"${CUDA_HOME}/bin/nvcc" --version | tail -n 1
if [[ -z "${THUNDERFFT_ARCHS:-}" ]]; then
  cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i 0 2>/dev/null | head -n 1 || true)"
  THUNDERFFT_ARCHS="${cap//./}"
fi
[[ "${THUNDERFFT_ARCHS}" =~ ^[0-9]+$ ]] || die "cannot detect the GPU architecture; set THUNDERFFT_ARCHS (e.g. 80)"
echo "THUNDERFFT_ARCHS=${THUNDERFFT_ARCHS}"

cuda_major="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p')"
case "${cuda_major}" in
  12)
    MATHDX_SHA256=d4952f6e1a7864d29d9ec4e01c21628d84086e941cf4a3da2df1e11ba5f13f84
    CUDNN_SHA256=21fad1b3a8a3db018868184b2082bd492633ae5101752432d655782bd108b19e ;;
  13)
    MATHDX_SHA256=81646046e7ae57169fcea79fb59df526eaef65e740b66e291a69182c612b8304
    CUDNN_SHA256=ce92b4058d95c85959b822baa56389126a31678f5068b486523d1aa7f6490fcb ;;
  *) die "no pinned MathDx/cuDNN archives for CUDA ${cuda_major}" ;;
esac
# CUDA 12 and 13 archives share their internal top-level directory name.
# Separate install roots prevent reusing the other toolkit's headers on rerun.
MATHDX_INSTALL="${HOME}/nvidia-mathdx-${MATHDX_VERSION}-cuda${cuda_major}"
MATHDX_ROOT="${MATHDX_INSTALL}/nvidia/mathdx/25.06"
MATHDX_URL=https://developer.download.nvidia.com/compute/cuFFTDx/redist/cuFFTDx/cuda${cuda_major}/nvidia-mathdx-${MATHDX_VERSION}-cuda${cuda_major}.tar.gz
CUDNN_NAME=cudnn-linux-x86_64-9.12.0.46_cuda${cuda_major}-archive
CUDNN_HOME="${HOME}/${CUDNN_NAME}"
CUDNN_URL=https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/${CUDNN_NAME}.tar.xz

step "MathDx ${MATHDX_VERSION} (cuFFTDx)"
if [[ ! -f "${MATHDX_ROOT}/include/cufftdx.hpp" ]]; then
  install_archive MathDx "${MATHDX_URL}" "${MATHDX_SHA256}" "${THUNDERFFT_MATHDX_ARCHIVE:-}" \
    "${MATHDX_INSTALL}"
fi
[[ -f "${MATHDX_ROOT}/include/cufftdx.hpp" ]] || die "MathDx install failed: ${MATHDX_ROOT}"
echo "${MATHDX_ROOT}"

step "cuDNN 9.12 (Figures 4 and 5)"
if [[ ! -f "${CUDNN_HOME}/include/cudnn_version.h" ]]; then
  install_archive cuDNN "${CUDNN_URL}" "${CUDNN_SHA256}" "${THUNDERFFT_CUDNN_ARCHIVE:-}" "${CUDNN_HOME}"
fi
grep -q '#define CUDNN_MINOR 12' "${CUDNN_HOME}/include/cudnn_version.h" || die "cuDNN install failed: ${CUDNN_HOME}"
echo "${CUDNN_HOME}"

step "Shell environment"
cat > "${ENV_FILE}" <<EOF
# Generated by artifact/setup.sh; rerun it to update.
. "${VENV}/bin/activate"
export CUDA_HOME="${CUDA_HOME}"
case ":\${PATH}:" in *":${CUDA_HOME}/bin:"*) ;; *) export PATH="${CUDA_HOME}/bin:\${PATH}" ;; esac
export PYTHON="${VENV}/bin/python"
export THUNDERFFT_ARCHS="${THUNDERFFT_ARCHS}"
export MATHDX_ROOT="${MATHDX_ROOT}"
export CUDNN_HOME="${CUDNN_HOME}"
case ":\${LD_LIBRARY_PATH:-}:" in *":${CUDNN_HOME}/lib:"*) ;; *) export LD_LIBRARY_PATH="${CUDNN_HOME}/lib\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" ;; esac
EOF
# Source it before ~/.bashrc's interactive-only guard so ssh commands see it too.
marker="# thunderfft artifact environment"
if ! grep -qF "${marker}" "${HOME}/.bashrc" 2>/dev/null; then
  line='[ -f "$HOME/.thunderfft_env" ] && . "$HOME/.thunderfft_env"'
  { printf '%s\n%s\n\n' "${marker}" "${line}"; cat "${HOME}/.bashrc" 2>/dev/null || true; } > "${HOME}/.bashrc.thunderfft"
  mv -- "${HOME}/.bashrc.thunderfft" "${HOME}/.bashrc"
fi
echo "${ENV_FILE} (sourced from ~/.bashrc)"

step "Done"
echo "Open a new shell or run: source ${ENV_FILE}"
echo "Then: ./run.sh --list"
