#!/usr/bin/env bash
# Shared configuration and command execution for OV workflows.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
CONFIG_FILE="${MWOP_CONFIG:-${ROOT}/scripts/config.sh}"
if [[ -f "${CONFIG_FILE}" ]]; then
    set -a
    source "${CONFIG_FILE}"
    set +a
elif [[ -n "${MWOP_CONFIG:-}" ]]; then
    echo "Configuration not found: ${CONFIG_FILE}" >&2
    exit 1
fi
export ENV_NAME="${ENV_NAME:-mwop-ov}"
export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT}/models/ov7b}"
export MODEL_REPO="${MODEL_REPO:-lmms-lab/llava-onevision-qwen2-7b-ov}"
export MODEL_REVISION="${MODEL_REVISION:-main}"
export DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export PYTHONPATH="${ROOT}/LLaVA-NeXT:${ROOT}/lmms-eval:${PYTHONPATH:-}"
for name in MODEL_NAME_OR_PATH OUTPUT_ROOT HF_HOME TRAIN_JSONL IMAGE_FOLDER DATA_PATH MASK_CONFIG OUTPUT_DIR EVAL_MODEL MERGED_DIR EVAL_OUTPUT TAYLOR_OUTPUT; do
    if [[ -n "${!name:-}" && "${!name}" != /* ]]; then
        printf -v "${name}" '%s/%s' "${ROOT}" "${!name}"
        export "${name}"
    fi
done
WORKFLOW="${ROOT}/LLaVA-NeXT/ov_workflow"
DRY_RUN=0
EXTRA_ARGS=()
for arg in "$@"; do
    case "${arg}" in
        --dry-run) DRY_RUN=1 ;;
        --help|-h) usage; exit 0 ;;
        *) EXTRA_ARGS+=("${arg}") ;;
    esac
done
run() {
    printf '+ '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" == 0 ]]; then "$@"; fi
}
fail() { echo "Error: $*" >&2; exit 1; }
positive_integer() {
    [[ "$2" =~ ^[1-9][0-9]*$ ]] || fail "$1 must be a positive integer."
}
init_conda() {
    local conda_base="${CONDA_ROOT:-}"
    if [[ -z "${conda_base}" ]] && command -v conda >/dev/null 2>&1; then
        conda_base="$(conda info --base)"
    elif [[ -z "${conda_base}" && -x "${CONDA_EXE:-}" ]]; then
        conda_base="$("${CONDA_EXE}" info --base)"
    fi
    [[ -f "${conda_base}/etc/profile.d/conda.sh" ]] || fail "Set CONDA_ROOT to your Miniconda directory or put conda on PATH."
    source "${conda_base}/etc/profile.d/conda.sh"
}
activate_environment() {
    [[ "${DRY_RUN}" == 1 ]] && return 0
    [[ "$(uname -s)" == Linux ]] || fail "Run these workflows on Linux with an NVIDIA CUDA GPU."
    if [[ -n "${VENV_PATH:-}" ]]; then
        [[ -f "${VENV_PATH}/bin/activate" ]] || fail "Cannot find ${VENV_PATH}/bin/activate."
        source "${VENV_PATH}/bin/activate"
    elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
        :
    elif [[ -z "${CONDA_PREFIX:-}" || "${CONDA_DEFAULT_ENV:-}" != "${ENV_NAME}" ]]; then
        init_conda
        conda activate "${ENV_NAME}"
    fi
}
prepare_model() {
    local args=(--model "${MODEL_NAME_OR_PATH}" --repo "${MODEL_REPO}" --revision "${MODEL_REVISION}")
    if [[ "${DOWNLOAD_MODELS}" == 1 ]]; then args+=(--download); fi
    run python "${ROOT}/scripts/workflow.py" model "${args[@]}"
}
check_runtime() {
    run python "${ROOT}/scripts/workflow.py" check --kind "$1" --gpus "${2:-1}"
}
