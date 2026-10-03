#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash scripts/03_test.sh [--dry-run] [extra lmms-eval arguments]
Merge outputs/ov_lora with the base OV model, preserving its training mask,
then evaluate 14 benchmark splits. A verified unchanged merge is reused.
Set EVAL_MODEL to evaluate an existing merged/dense model without merging.
EVAL_TASKS=gqa EVAL_LIMIT=2 runs a short evaluation.
EOF
}
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
activate_environment
check_runtime eval
if [[ -z "${EVAL_MODEL:-}" ]]; then
    prepare_model
    EVAL_MODEL="${MERGED_DIR:-${OUTPUT_ROOT}/ov_merged}"
    run python "${ROOT}/scripts/workflow.py" merge --base "${MODEL_NAME_OR_PATH}" --adapter "${OUTPUT_DIR:-${OUTPUT_ROOT}/ov_lora}" --output "${EVAL_MODEL}"
elif [[ "${DRY_RUN}" == 0 ]]; then
    run python "${ROOT}/scripts/workflow.py" model --model "${EVAL_MODEL}"
fi
eval_args=(--model "${EVAL_MODEL}" --output "${EVAL_OUTPUT:-${OUTPUT_ROOT}/eval_14bench}")
if [[ -n "${EVAL_TASKS:-}" ]]; then eval_args+=(--tasks "${EVAL_TASKS}"); fi
if [[ -n "${EVAL_LIMIT:-}" ]]; then
    positive_integer EVAL_LIMIT "${EVAL_LIMIT}"
    eval_args+=(--limit "${EVAL_LIMIT}")
fi
# run_eval.py reads ablation_config.json from the evaluated model, so the
# training mask is retained even if MASK_CONFIG has since been changed.
run python "${ROOT}/lmms-eval/ov_workflow/run_eval.py" "${eval_args[@]}" "${EXTRA_ARGS[@]}"
