#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash scripts/02_train.sh [--dry-run] [extra Trainer arguments]
Run OV LoRA recovery using TRAIN_JSONL + IMAGE_FOLDER (or DATA_PATH YAML).
The default uses the exact frozen attention/FFN masks. Set MASK_CONFIG to
outputs/taylor/joint_mask.json to train with newly computed Taylor masks.
NPROC_PER_NODE defaults to 1; MAX_STEPS=2 enables a short trial.
EOF
}
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
export NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
export NNODES="${NNODES:-${SLURM_NNODES:-1}}"
for name in NPROC_PER_NODE TARGET_GLOBAL_BATCH PER_DEVICE_TRAIN_BATCH_SIZE NNODES; do
    positive_integer "${name}" "${!name}"
done
batch_unit=$(( NPROC_PER_NODE * NNODES * PER_DEVICE_TRAIN_BATCH_SIZE ))
(( TARGET_GLOBAL_BATCH % batch_unit == 0 )) || fail "TARGET_GLOBAL_BATCH must be divisible by GPU count times per-device batch size."
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(( TARGET_GLOBAL_BATCH / batch_unit ))}"
positive_integer GRADIENT_ACCUMULATION_STEPS "${GRADIENT_ACCUMULATION_STEPS}"
(( batch_unit * GRADIENT_ACCUMULATION_STEPS == TARGET_GLOBAL_BATCH )) || fail "GRADIENT_ACCUMULATION_STEPS does not give TARGET_GLOBAL_BATCH."
[[ -n "${IMAGE_FOLDER:-}" ]] || fail "Set IMAGE_FOLDER in scripts/config.sh."
[[ -n "${DATA_PATH:-}${TRAIN_JSONL:-}" ]] || fail "Set TRAIN_JSONL or DATA_PATH in scripts/config.sh."
activate_environment
check_runtime train "${NPROC_PER_NODE}"
data_args=(--images "${IMAGE_FOLDER}" --output "${OUTPUT_ROOT}/data/train.yaml")
if [[ -n "${DATA_PATH:-}" ]]; then
    data_args+=(--yaml "${DATA_PATH}")
else
    data_args+=(--jsonl "${TRAIN_JSONL}")
fi
run python "${ROOT}/scripts/workflow.py" data "${data_args[@]}"
export DATA_PATH="${OUTPUT_ROOT}/data/train.yaml"
prepare_model
if [[ -z "${VISION_TOWER:-}" ]]; then
    if [[ "${DRY_RUN}" == 1 ]]; then
        VISION_TOWER='<vision tower from OV config.json>'
    else
        VISION_TOWER="$(python "${ROOT}/scripts/workflow.py" vision --model "${MODEL_NAME_OR_PATH}")"
    fi
fi
export VISION_TOWER
if [[ -z "${MASK_CONFIG:-}" ]]; then
    run python "${WORKFLOW}/use_frozen_ov_masks.py" --output "${OUTPUT_ROOT}/masks"
    MASK_CONFIG="${OUTPUT_ROOT}/masks/joint_mask.json"
elif [[ "${DRY_RUN}" == 0 && ! -f "${MASK_CONFIG}" ]]; then
    fail "Mask not found: ${MASK_CONFIG}. Run scripts/04_taylor.sh first if using new masks."
fi
export HEAD_MASK_CONFIG="${MASK_CONFIG}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/ov_lora}"
export PROJECT_ROOT="${WORKFLOW}"
export LLAVA_ROOT="${ROOT}/LLaVA-NeXT"
run env MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH}" VISION_TOWER="${VISION_TOWER}" \
    DATA_PATH="${DATA_PATH}" IMAGE_FOLDER="${IMAGE_FOLDER}" \
    HEAD_MASK_CONFIG="${HEAD_MASK_CONFIG}" OUTPUT_DIR="${OUTPUT_DIR}" \
    NPROC_PER_NODE="${NPROC_PER_NODE}" TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH}" \
    GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" \
    bash "${WORKFLOW}/scripts/train_lora.sh" "${EXTRA_ARGS[@]}"
