#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash scripts/05_speed.sh [--dry-run] [extra run_serial.py arguments]
Benchmark the HF-format OV-7B decoder serially: Dense/MWOP, MWOP +
PyramidDrop, and MWOP + ZOO. Export latency reports and logical/packed FLOPs.
Uses SPEED_ENV_NAME (default mwop-ov-speed) or SPEED_VENV_PATH, independently
of the native training environment. Prepare this acceleration environment
according to acceleration/README.md before running.
SPEED_MODEL defaults to models/hf_ov7b; missing weights are downloaded from
SPEED_MODEL_REPO if DOWNLOAD_MODELS=1. Native LoRA outputs need conversion.
Set SPEED_OUTPUT to a fresh directory for each experiment.
EOF
}
original_pythonpath="${PYTHONPATH:-}"
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
export ENV_NAME="${SPEED_ENV_NAME:-mwop-ov-speed}"
export VENV_PATH="${SPEED_VENV_PATH:-}"
export PYTHONPATH="${ROOT}/acceleration:${original_pythonpath}"
SPEED_MODEL="${SPEED_MODEL:-${ROOT}/models/hf_ov7b}"
SPEED_OUTPUT="${SPEED_OUTPUT:-${OUTPUT_ROOT}/ov_speed}"
SPEED_TILES="${SPEED_TILES:-${ROOT}/acceleration/configs/mwop_validated_tiles.json}"
for name in SPEED_MODEL SPEED_OUTPUT SPEED_TILES; do
    if [[ "${!name}" != /* ]]; then
        printf -v "${name}" '%s/%s' "${ROOT}" "${!name}"
    fi
done
SPEED_ALIGNMENT="${SPEED_ALIGNMENT:-64}"
SPEED_REP="${SPEED_REP:-100}"
SPEED_SAMPLES="${SPEED_SAMPLES:-5}"
for name in SPEED_ALIGNMENT SPEED_REP SPEED_SAMPLES; do
    positive_integer "${name}" "${!name}"
done
for arg in "${EXTRA_ARGS[@]}"; do
    case "${arg}" in
        --model|--model=*|--output|--output=*|--alignment|--alignment=*|--rep|--rep=*|--samples|--samples=*|--tiles|--tiles=*)
            fail "Set SPEED_MODEL/SPEED_OUTPUT/SPEED_ALIGNMENT/SPEED_REP/SPEED_SAMPLES/SPEED_TILES instead of ${arg}." ;;
    esac
done
if [[ "${DRY_RUN}" == 0 && -n "${VIRTUAL_ENV:-}" && -z "${SPEED_VENV_PATH:-}" ]]; then
    fail "Set SPEED_VENV_PATH explicitly for the acceleration venv, or deactivate it before selecting SPEED_ENV_NAME."
fi
activate_environment
[[ "${DRY_RUN}" == 1 || -f "${SPEED_TILES}" ]] || fail "Speed tile plan not found: ${SPEED_TILES}."
prepare_args=(--model "${SPEED_MODEL}" --repo "${SPEED_MODEL_REPO:-llava-hf/llava-onevision-qwen2-7b-ov-hf}" --revision "${SPEED_MODEL_REVISION:-main}")
if [[ "${DOWNLOAD_MODELS}" == 1 ]]; then prepare_args+=(--download); fi
run python "${ROOT}/scripts/workflow.py" speed "${prepare_args[@]}"
run python "${ROOT}/acceleration/run_serial.py" --model "${SPEED_MODEL}" \
    --alignment "${SPEED_ALIGNMENT}" --rep "${SPEED_REP}" --samples "${SPEED_SAMPLES}" \
    --tiles "${SPEED_TILES}" \
    --output "${SPEED_OUTPUT}" "${EXTRA_ARGS[@]}"
run python "${ROOT}/acceleration/theory_ov.py" --ffn-alignment 1 --output "${SPEED_OUTPUT}/theory_logical.json"
run python "${ROOT}/acceleration/theory_ov.py" --ffn-alignment "${SPEED_ALIGNMENT}" --output "${SPEED_OUTPUT}/theory_packed.json"
