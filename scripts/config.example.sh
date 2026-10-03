# Copy to scripts/config.sh. Paths are resolved from the repository root.
# Environment variables provided before a command take precedence.
ENV_NAME="${ENV_NAME:-mwop-ov}"
# CONDA_ROOT="${CONDA_ROOT:-/path/to/miniconda3}"
# VENV_PATH="${VENV_PATH:-/path/to/venv}"
# CUDA_HOME="${CUDA_HOME:-/path/to/cuda-toolkit}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-${ROOT}/models/ov7b}"
MODEL_REPO="${MODEL_REPO:-lmms-lab/llava-onevision-qwen2-7b-ov}"
MODEL_REVISION="${MODEL_REVISION:-main}"
DOWNLOAD_MODELS="${DOWNLOAD_MODELS:-1}"  # Download if the local model is absent.
# VISION_TOWER="${VISION_TOWER:-/path/to/siglip}"  # Otherwise read from OV config.

TRAIN_JSONL="${TRAIN_JSONL:-/path/to/train.jsonl}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/path/to/images}"
# LMMS_DATA_ROOT="${LMMS_DATA_ROOT:-/path/to/benchmarks}"  # Optional local HF dataset copies.
# DATA_PATH="${DATA_PATH:-/path/to/train.yaml}"  # Takes precedence over TRAIN_JSONL.
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/outputs}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
TARGET_GLOBAL_BATCH="${TARGET_GLOBAL_BATCH:-256}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
# MASK_CONFIG="${MASK_CONFIG:-${OUTPUT_ROOT}/taylor/joint_mask.json}"
# OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/ov_lora}"
# MAX_STEPS="${MAX_STEPS:-2}"  # Optional short training run.

TAYLOR_SAMPLES="${TAYLOR_SAMPLES:-256}"
TAYLOR_SEED="${TAYLOR_SEED:-42}"
MAX_IMAGE_SIDE="${MAX_IMAGE_SIDE:-768}"
V2V_PERCENT="${V2V_PERCENT:-40}"
T2V_PERCENT="${T2V_PERCENT:-60}"
T2T_PERCENT="${T2T_PERCENT:-10}"
FFN_RATIO="${FFN_RATIO:-0.5}"

# EVAL_MODEL="${EVAL_MODEL:-/path/to/merged-ov}"  # Skip merging and evaluate this model.
# EVAL_TASKS="${EVAL_TASKS:-gqa,ai2d}"  # Default: all 14 task splits.
# EVAL_LIMIT="${EVAL_LIMIT:-2}"  # Unset means full evaluation.
# EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_ROOT}/eval_14bench}"
# MERGED_DIR="${MERGED_DIR:-${OUTPUT_ROOT}/ov_merged}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"  # Native scripts use SDPA/eager.

# Decoder speed testing uses a separately prepared acceleration environment.
SPEED_ENV_NAME="${SPEED_ENV_NAME:-mwop-ov-speed}"
# SPEED_VENV_PATH="${SPEED_VENV_PATH:-/path/to/acceleration-venv}"
SPEED_MODEL="${SPEED_MODEL:-${ROOT}/models/hf_ov7b}"
SPEED_MODEL_REPO="${SPEED_MODEL_REPO:-llava-hf/llava-onevision-qwen2-7b-ov-hf}"
SPEED_MODEL_REVISION="${SPEED_MODEL_REVISION:-main}"
SPEED_OUTPUT="${SPEED_OUTPUT:-${OUTPUT_ROOT}/ov_speed}"
SPEED_ALIGNMENT="${SPEED_ALIGNMENT:-64}"  # Zero-pad retained FFN widths for efficient GEMMs.
SPEED_REP="${SPEED_REP:-100}"
SPEED_SAMPLES="${SPEED_SAMPLES:-5}"
SPEED_TILES="${SPEED_TILES:-${ROOT}/acceleration/configs/mwop_validated_tiles.json}"
