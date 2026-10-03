#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${PROJECT_ROOT}"

ENV_NAME="${ENV_NAME:-mwop-ov}"
RECREATE_ENV="${RECREATE_ENV:-0}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"
INSTALL_LMMS_EVAL="${INSTALL_LMMS_EVAL:-1}"
CONDA_CREATE_MODE="${CONDA_CREATE_MODE:-env_file}"
CONDA_CHANNEL_ARGS="${CONDA_CHANNEL_ARGS:---override-channels -c defaults}"
DISABLE_PIP_PROXY="${DISABLE_PIP_PROXY:-0}"

TORCH_VERSION="${TORCH_VERSION:-2.7.0+cu128}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.22.0+cu128}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.7.0+cu128}"
PYTORCH_CUDA_TAG="${PYTORCH_CUDA_TAG:-cu128}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/${PYTORCH_CUDA_TAG}}"
PYTORCH_EXTRA_INDEX_URL="${PYTORCH_EXTRA_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_FIND_LINKS="${PYTORCH_FIND_LINKS:-}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"

ENV_FILE="${ENV_FILE:-${PROJECT_ROOT}/LLaVA-NeXT/ov_workflow/envs/mwop-ov.yml}"
REQUIREMENTS_FILE="${REQUIREMENTS_FILE:-${PROJECT_ROOT}/LLaVA-NeXT/ov_workflow/envs/mwop-ov.requirements.txt}"
LMMS_RUNTIME_FILE="${LMMS_RUNTIME_FILE:-${PROJECT_ROOT}/LLaVA-NeXT/ov_workflow/envs/mwop-ov.lmms-eval-runtime.txt}"

if [[ -n "${CONDA_ROOT:-}" ]]; then
    CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    CONDA_ROOT="$(conda info --base)"
    CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
elif [[ -x "${CONDA_EXE:-}" ]]; then
    CONDA_ROOT="$("${CONDA_EXE}" info --base)"
    CONDA_SH="${CONDA_ROOT}/etc/profile.d/conda.sh"
else
    echo "Cannot find conda. Set CONDA_ROOT=/path/to/miniconda3." >&2
    exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Cannot find conda.sh at ${CONDA_SH}." >&2
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Cannot find conda env file: ${ENV_FILE}" >&2
    exit 1
fi

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    echo "Cannot find requirements file: ${REQUIREMENTS_FILE}" >&2
    exit 1
fi

if [[ ! -f "${LMMS_RUNTIME_FILE}" ]]; then
    echo "Cannot find lmms-eval runtime file: ${LMMS_RUNTIME_FILE}" >&2
    exit 1
fi

if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    unset CUDA_HOME
fi
if [[ -z "${CUDA_HOME:-}" ]] && command -v nvcc >/dev/null 2>&1; then
    nvcc_path="$(readlink -f -- "$(command -v nvcc)")"
    export CUDA_HOME="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
fi
if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
fi

source "${CONDA_SH}"
read -r -a conda_channel_args <<< "${CONDA_CHANNEL_ARGS}"

if [[ "${RECREATE_ENV}" == "1" ]] && conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Removing existing conda env: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "Creating conda env: ${ENV_NAME}"
    if [[ "${CONDA_CREATE_MODE}" == "simple" ]]; then
        conda create -n "${ENV_NAME}" \
            "${conda_channel_args[@]}" \
            python=3.10 pip setuptools wheel ninja packaging -y
    else
        conda env create -f "${ENV_FILE}"
    fi
fi

conda activate "${ENV_NAME}"

echo "Project root: ${PROJECT_ROOT}"
echo "Conda root: ${CONDA_ROOT}"
echo "Conda env: ${CONDA_PREFIX}"
echo "CUDA_HOME=${CUDA_HOME:-unset}"

pip_common_args=(--retries 10 --timeout 120)
if [[ "${DISABLE_PIP_PROXY}" == "1" ]]; then
    pip_common_args+=(--proxy "")
fi

python -m pip install "${pip_common_args[@]}" --upgrade pip setuptools wheel

torch_pip_args=(
    --index-url "${PYTORCH_INDEX_URL}"
    --extra-index-url "${PYTORCH_EXTRA_INDEX_URL}"
)
if [[ -n "${PYTORCH_FIND_LINKS}" ]]; then
    torch_pip_args+=(--find-links "${PYTORCH_FIND_LINKS}")
fi

python -m pip install \
    "${pip_common_args[@]}" \
    "${torch_pip_args[@]}" \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}"

python -m pip install "${pip_common_args[@]}" -r "${REQUIREMENTS_FILE}"

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
    python -m pip install "${pip_common_args[@]}" "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation
fi

python -m pip install "${pip_common_args[@]}" -e "${PROJECT_ROOT}/LLaVA-NeXT" --no-deps --no-build-isolation

if [[ "${INSTALL_LMMS_EVAL}" == "1" ]]; then

    python -m pip install "${pip_common_args[@]}" --no-deps -r "${LMMS_RUNTIME_FILE}"
    python -m pip install "${pip_common_args[@]}" -e "${PROJECT_ROOT}/lmms-eval" --no-deps --no-build-isolation
fi

python "${PROJECT_ROOT}/LLaVA-NeXT/ov_workflow/setup/smoke.py"

echo "mwop-ov installation complete."
