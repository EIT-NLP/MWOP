#!/usr/bin/env bash
set -euo pipefail

export CC=gcc
export CXX=g++

ENV_NAME="${ENV_NAME:-mwop-ov}"
CONDA_ROOT="${CONDA_ROOT:-}"
VENV_PATH="${VENV_PATH:-${UV_PROJECT_ENVIRONMENT:-}}"

PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_PATH="${DATA_PATH:?Set DATA_PATH to a training YAML}"
IMAGE_FOLDER="${IMAGE_FOLDER:?Set IMAGE_FOLDER}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH}"
VISION_TOWER="${VISION_TOWER:?Set VISION_TOWER}"

ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
if [[ "${ATTN_IMPLEMENTATION}" != "sdpa" && "${ATTN_IMPLEMENTATION}" != "eager" ]]; then
    echo "Path masks require sdpa or eager" >&2
    exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/post_train_img_ckpts/llava-onevision-qwen2-7b-ov-post-train-lora-img}"
RUN_NAME="${RUN_NAME:-$(basename "${OUTPUT_DIR}")}"

MM_TUNABLE_PARTS="${MM_TUNABLE_PARTS:-mm_mlp_adapter,mm_language_model}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"

PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"

TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
_GPUS=$(( NPROC_PER_NODE * NNODES ))
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-$(( TARGET_GLOBAL_BATCH / (_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE) ))}"
echo "[gbs] target=${TARGET_GLOBAL_BATCH} gpus=${_GPUS} per_device=${PER_DEVICE_TRAIN_BATCH_SIZE} -> grad_accum=${GRADIENT_ACCUMULATION_STEPS} (global=$(( _GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS )))"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
SAVE_STEPS="${SAVE_STEPS:-100}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
REPORT_TO="${REPORT_TO:-none}"
MASTER_PORT="${MASTER_PORT:-29500}"

LLAVA_ROOT="${LLAVA_ROOT:-${PROJECT_ROOT}/..}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${LLAVA_ROOT}/scripts/zero2.json}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${LLAVA_ROOT}/llava/train/train_mem.py}"

if [[ -n "${VENV_PATH}" ]]; then
    if [[ ! -f "${VENV_PATH}/bin/activate" ]]; then
        echo "Cannot find venv activation script: ${VENV_PATH}/bin/activate" >&2
        exit 1
    fi
    source "${VENV_PATH}/bin/activate"
elif [[ -n "${VIRTUAL_ENV:-}" ]]; then
    :
elif [[ -n "${CONDA_PREFIX:-}" && "${CONDA_DEFAULT_ENV:-}" == "${ENV_NAME}" ]]; then
    :
else
    if [[ -z "${CONDA_ROOT}" ]]; then
        if command -v conda >/dev/null 2>&1; then
            CONDA_ROOT="$(conda info --base)"
        elif [[ -x "${CONDA_EXE:-}" ]]; then
            CONDA_ROOT="$("${CONDA_EXE}" info --base)"
        fi
    fi
    if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        echo "Cannot find conda.sh. Set CONDA_ROOT=/path/to/miniconda3 or activate a venv first." >&2
        exit 1
    fi
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
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

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
export PYTHONPATH="${LLAVA_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

for path in \
    "${LLAVA_ROOT}" \
    "${DEEPSPEED_CONFIG}" \
    "${TRAIN_SCRIPT}" \
    "${DATA_PATH}" \
    "${IMAGE_FOLDER}" \
    "${MODEL_NAME_OR_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Missing path: ${path}" >&2
        exit 1
    fi
done

# The OV config may identify its SigLIP tower by a public Hugging Face repo ID.
if [[ ! -d "${VISION_TOWER}" && ! "${VISION_TOWER}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
    echo "VISION_TOWER must be a local directory or a Hugging Face owner/repo ID: ${VISION_TOWER}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Working dir: $(pwd)"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "Data path: ${DATA_PATH}"
echo "Image folder: ${IMAGE_FOLDER}"
echo "Model: ${MODEL_NAME_OR_PATH}"
echo "Vision tower: ${VISION_TOWER}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Attention: ${ATTN_IMPLEMENTATION}"
echo "Master port: ${MASTER_PORT}"

TRAIN_ARGS=(
    "${TRAIN_SCRIPT}"
    --deepspeed "${DEEPSPEED_CONFIG}"
    --model_name_or_path "${MODEL_NAME_OR_PATH}"
    --version qwen_1_5
    --data_path "${DATA_PATH}"
    --image_folder "${IMAGE_FOLDER}"
    --mm_tunable_parts "${MM_TUNABLE_PARTS}"
    --vision_tower "${VISION_TOWER}"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --group_by_modality_length True
    --image_aspect_ratio anyres_max_9
    --image_grid_pinpoints "(1x1),...,(6x6)"
    --mm_patch_merge_type spatial_unpad
    --lora_enable True
    --lora_r 128
    --lora_alpha 256
    --lora_dropout 0.05
    --lora_bias none
    --mm_projector_lr 1e-5
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --bf16 True
    --run_name "${RUN_NAME}"
    --output_dir "${OUTPUT_DIR}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size 1
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --evaluation_strategy "no"
    --save_strategy "steps"
    --save_steps "${SAVE_STEPS}"
    --save_total_limit "${SAVE_TOTAL_LIMIT}"
    --save_only_model False
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.
    --warmup_ratio "${WARMUP_RATIO}"
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --tf32 True
    --model_max_length 32768
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to "${REPORT_TO}"
    --torch_compile False
    --torch_compile_backend inductor
    --dataloader_drop_last True
)

HEAD_MASK_CONFIG="${HEAD_MASK_CONFIG:-}"
if [[ -n "${HEAD_MASK_CONFIG}" ]]; then
    if [[ ! -e "${HEAD_MASK_CONFIG}" ]]; then
        echo "HEAD_MASK_CONFIG path does not exist: ${HEAD_MASK_CONFIG}" >&2
        exit 1
    fi
    echo "Head mask config: ${HEAD_MASK_CONFIG}"
    TRAIN_ARGS+=( --head_mask_config_path "${HEAD_MASK_CONFIG}" )
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
    echo "[smoke] MAX_STEPS=${MAX_STEPS} (step-capped run, overrides --num_train_epochs)"
    TRAIN_ARGS+=( --max_steps "${MAX_STEPS}" )
fi

USE_SRUN="${USE_SRUN:-0}"
if [[ -n "${HEAD_MASK_CONFIG}" ]]; then
    if [[ "$(realpath "${HEAD_MASK_CONFIG}")" != "$(realpath -m "${OUTPUT_DIR}/ablation_config.json")" ]]; then
        cp -- "${HEAD_MASK_CONFIG}" "${OUTPUT_DIR}/ablation_config.json"
    fi
fi
if [[ "$(( _GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS ))" != "${TARGET_GLOBAL_BATCH}" ]]; then
    echo "Global batch must equal TARGET_GLOBAL_BATCH" >&2
    exit 1
fi
export NPROC_PER_NODE MASTER_PORT
if [[ "${USE_SRUN}" == "1" ]]; then
    MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "${SLURM_NODELIST}" | head -n 1)}"
    export MASTER_ADDR
    srun --ntasks="${SLURM_NNODES}" --ntasks-per-node=1 bash -c 'exec torchrun --nnodes="${SLURM_NNODES}" --nproc_per_node="${NPROC_PER_NODE}" --node_rank="${SLURM_PROCID}" --rdzv_id="${SLURM_JOB_ID}" --rdzv_backend=c10d --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" "$@"' ov-worker "${TRAIN_ARGS[@]}" "$@"
else
    if [[ "${NNODES}" != "1" ]]; then
        echo "For Slurm multi-node runs set USE_SRUN=1" >&2
        exit 1
    fi
    torchrun --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" "${TRAIN_ARGS[@]}" "$@"
fi
