#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: [INSTALL_TARGET=native|speed] bash scripts/01_install.sh [--dry-run]
Create/reuse ENV_NAME (default mwop-ov), install CUDA 12.8 PyTorch and
the root pyproject.toml train/eval extras, then check imports and pip metadata.
Requires Linux x86_64 and Conda. Set CONDA_ROOT if Conda is not on PATH.
INSTALL_FLASH_ATTN=1 also builds the optional FlashAttention package.
INSTALL_TARGET=speed installs the separate SPEED_ENV_NAME environment
with Python 3.12 and CUDA 13.0 PyTorch. FlashAttention requires a matching
CUDA toolkit; INSTALL_CUDA_TOOLKIT=1 installs NVIDIA's compiler wheels.
No models or datasets are downloaded by this command.
EOF
}
original_pythonpath="${PYTHONPATH:-}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ ${#EXTRA_ARGS[@]} == 0 ]] || fail "Unexpected arguments. See --help."
install_target="${INSTALL_TARGET:-native}"
case "${install_target}" in
    native) python_version=3.10.20; python_minor=10 ;;
    speed)
        export ENV_NAME="${SPEED_ENV_NAME:-mwop-ov-speed}"
        export PYTHONPATH="${ROOT}/acceleration:${original_pythonpath}"
        export CC="${SPEED_CC:-gcc}" CXX="${SPEED_CXX:-g++}"
        export CFLAGS="${SPEED_CFLAGS:-}" CXXFLAGS="${SPEED_CXXFLAGS:-}"
        export CPPFLAGS="${SPEED_CPPFLAGS:-}" LDFLAGS="${SPEED_LDFLAGS:-}"
        python_version=3.12.13; python_minor=12 ;;
    *) fail "INSTALL_TARGET must be native or speed." ;;
esac
if [[ "${DRY_RUN}" == 0 ]]; then
    [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || fail "The pinned environment requires Linux x86_64."
    init_conda
    if ! conda run -n "${ENV_NAME}" python -c "import sys; assert sys.version_info[:2] == (3, ${python_minor})" >/dev/null 2>&1; then
        # An existing environment with the wrong Python must not be overwritten.
        if conda run -n "${ENV_NAME}" python --version >/dev/null 2>&1; then
            fail "${ENV_NAME} exists with a different Python. Choose a new ENV_NAME."
        fi
        run conda create -n "${ENV_NAME}" "python=${python_version}" pip -y
    fi
    conda activate "${ENV_NAME}"
else
    run conda create -n "${ENV_NAME}" "python=${python_version}" pip -y
    run conda activate "${ENV_NAME}"
fi
run python -m pip install --upgrade pip 'setuptools>=77,<81' wheel
if [[ "${install_target}" == speed ]]; then
    run python -m pip install ninja==1.13.0 packaging==26.0
    run python -m pip install torch==2.11.0+cu130 torchvision==0.26.0+cu130 --index-url https://download.pytorch.org/whl/cu130
    if [[ "${INSTALL_CUDA_TOOLKIT:-0}" == 1 ]]; then
        run python -m pip install nvidia-cuda-nvcc==13.0.88 nvidia-cuda-crt==13.0.88 nvidia-nvvm==13.0.88
        if [[ "${DRY_RUN}" == 0 ]]; then
            CUDA_HOME="$(python -c 'from importlib.metadata import distribution; from pathlib import Path; d=distribution("nvidia-cuda-nvcc"); roots=[Path(d.locate_file(f)).parents[1] for f in d.files if str(f).endswith("bin/nvcc")]; assert len(roots)==1; print(roots[0])')"
            export CUDA_HOME
            export PATH="${CUDA_HOME}/bin:${PATH}"
            if [[ ! -e "${CUDA_HOME}/lib64" && -d "${CUDA_HOME}/lib" ]]; then
                run ln -s lib "${CUDA_HOME}/lib64"
            fi
        fi
    fi
    run python -m pip install --no-build-isolation -e "${ROOT}/acceleration"
    run python "${ROOT}/scripts/workflow.py" check --kind speed-install
    run python -m pip check
    if [[ "${DRY_RUN}" == 0 ]]; then
        echo "Acceleration environment ready: ${ENV_NAME}. Run scripts/05_speed.sh."
    else
        echo "Preview complete; no commands were executed."
    fi
    exit 0
fi
run python -m pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# DeepSpeed's source build needs to see the Torch installation above.
run python -m pip install --no-build-isolation -e "${ROOT}[train,eval]"
if [[ "${INSTALL_FLASH_ATTN:-0}" == 1 ]]; then
    run python -m pip install flash-attn==2.8.3 --no-build-isolation
fi
run python "${ROOT}/scripts/workflow.py" check --kind install
run python -m pip check
if [[ "${DRY_RUN}" == 0 ]]; then
    echo "Environment ready: ${ENV_NAME}. Training/Taylor/evaluation scripts activate it automatically."
else
    echo "Preview complete; no commands were executed."
fi
