#!/usr/bin/env bash
set -euo pipefail
usage() {
    cat <<'EOF'
Usage: bash scripts/04_taylor.sh [--dry-run]
Collect attention Taylor scores on the dense native OV model, aggregate
four categories, build its path mask, then collect structured FFN Taylor
scores under that mask and export a joint mask. All 14 splits are required.
TAYLOR_SAMPLES defaults to 256 successful samples per split.
Outputs: outputs/taylor/{path,ffn,path_categories,ffn_rankings,*_mask.json}.
EOF
}
source "$(dirname -- "${BASH_SOURCE[0]}")/common.sh" "$@"
[[ ${#EXTRA_ARGS[@]} == 0 ]] || fail "Unexpected arguments. See --help."
TAYLOR_SAMPLES="${TAYLOR_SAMPLES:-256}"
positive_integer TAYLOR_SAMPLES "${TAYLOR_SAMPLES}"
activate_environment
check_runtime taylor
prepare_model
taylor_dir="${TAYLOR_OUTPUT:-${OUTPUT_ROOT}/taylor}"
score_args=(--model "${MODEL_NAME_OR_PATH}" --samples "${TAYLOR_SAMPLES}" --seed "${TAYLOR_SEED:-42}" --device "${TAYLOR_DEVICE:-auto}" --max-image-side "${MAX_IMAGE_SIDE:-768}" --output "${taylor_dir}")
run python "${WORKFLOW}/search_taylor.py" --kind path "${score_args[@]}"
run python "${WORKFLOW}/build_path_categories.py" --input "${taylor_dir}/path" --samples "${TAYLOR_SAMPLES}" --output "${taylor_dir}/path_categories"
run python "${WORKFLOW}/scripts/path_analysis/gen_universal_max_ranking.py" --rankings_dir "${taylor_dir}/path_categories" --agg max --out "${taylor_dir}/path_univmax.csv"
run python "${WORKFLOW}/scripts/path_mask/make_path_mask_config.py" --taylor_csv "${taylor_dir}/path_univmax.csv" --v2v "${V2V_PERCENT:-40}" --t2v "${T2V_PERCENT:-60}" --t2t "${T2T_PERCENT:-10}" --output "${taylor_dir}/path_mask.json"
run python "${WORKFLOW}/search_taylor.py" --kind ffn --mask "${taylor_dir}/path_mask.json" "${score_args[@]}"
run python "${WORKFLOW}/scripts/analysis/build_ffn_fourcat_univmax.py" --input_dir "${taylor_dir}/ffn" --expected_samples "${TAYLOR_SAMPLES}" --source_metric structured --output_dir "${taylor_dir}/ffn_rankings"
run python "${WORKFLOW}/scripts/prune/make_ffn_prune_config.py" --ckpt "${MODEL_NAME_OR_PATH}" --metric taylor --act_scores "${taylor_dir}/ffn_rankings/ffn_univmax_vision_taylor.pt" --token_scope vision --alloc global --ratio "${FFN_RATIO:-0.5}" --out "${taylor_dir}/ffn_mask.json"
run python "${WORKFLOW}/scripts/prune/make_joint_path_ffn_config.py" --path_config "${taylor_dir}/path_mask.json" --ffn_config "${taylor_dir}/ffn_mask.json" --out "${taylor_dir}/joint_mask.json"
echo "Next: MASK_CONFIG=\"${taylor_dir}/joint_mask.json\" bash scripts/02_train.sh"
