# OV-7B pruning, recovery training, and inference

Run commands from the repository root on Linux/CUDA. Prepare native OV-7B and its SigLIP vision tower.

```bash
CONDA_ROOT=/path/to/miniconda3 bash LLaVA-NeXT/ov_workflow/setup/install.sh
conda activate mwop-ov
```

The repository-root [pyproject.toml](../pyproject.toml) provides the validated native OV dependencies and `train`/`eval` extras. See the root [Preparation guide](../README.md#preparation) for the alternate root-package installation route.

The installation script and `ov_workflow/envs/` pin the native runtime, including Transformers commit `1c39974a4c4036fd641bc1191cc32799f85715a4`. Use this installation route for training dependencies. Masked attention uses `sdpa` or `eager`. HF decoder acceleration uses a separate environment.

## Frozen OV configuration

Use this route to load the same head/channel indices as the acceleration benchmark:

```bash
python LLaVA-NeXT/ov_workflow/use_frozen_ov_masks.py --output outputs
```

It writes `path_mask.json`, `ffn_mask.json`, and `joint_mask.json`. The frozen attention masks select 314/470/78 of 784 heads for V2V/T2V/T2T. The visual FFN mask removes exactly 265,216 of 530,432 channels across all 28 layers, with all channels in zero-based layer 27 removed and text channels retained. Source-file hashes accompany the exported joint configuration.

## New Taylor calibration

This route produces new masks; identical pruning counts do not imply identical frozen indices.

```bash
python LLaVA-NeXT/ov_workflow/search_taylor.py --model /path/to/native-ov7b --kind path --samples 256 --output outputs/search_path
python LLaVA-NeXT/ov_workflow/build_path_categories.py --input outputs/search_path/path --samples 256 --output outputs/path_categories
python LLaVA-NeXT/ov_workflow/scripts/path_analysis/gen_universal_max_ranking.py --rankings_dir outputs/path_categories --agg max --out outputs/new_path_univmax.csv
python LLaVA-NeXT/ov_workflow/scripts/path_mask/make_path_mask_config.py --taylor_csv outputs/new_path_univmax.csv --v2v 40 --t2v 60 --t2t 10 --output outputs/new_path_mask.json
python LLaVA-NeXT/ov_workflow/search_taylor.py --model /path/to/native-ov7b --kind ffn --mask outputs/new_path_mask.json --samples 256 --output outputs/search_ffn
python LLaVA-NeXT/ov_workflow/scripts/analysis/build_ffn_fourcat_univmax.py --input_dir outputs/search_ffn/ffn --expected_samples 256 --source_metric structured --output_dir outputs/ffn_rankings
python LLaVA-NeXT/ov_workflow/scripts/prune/make_ffn_prune_config.py --ckpt /path/to/native-ov7b --metric taylor --act_scores outputs/ffn_rankings/ffn_univmax_vision_taylor.pt --token_scope vision --alloc global --ratio 0.5 --out outputs/new_ffn_mask.json
python LLaVA-NeXT/ov_workflow/scripts/prune/make_joint_path_ffn_config.py --path_config outputs/new_path_mask.json --ffn_config outputs/new_ffn_mask.json --out outputs/new_joint_mask.json
```

Path calibration uses the unmasked model. FFN calibration re-estimates importance with the path mask active. Four-category aggregation averages RefCOCO/RefCOCO+ A/B into their task units, gives each category three equal task units, and takes Universal-Max across category percentile rankings.

Taylor files use schema version 2. `structured` stores whole-neuron first-order Taylor scores in `scores_structured*`; `legacy` stores mean token-wise absolute scores. All-zero layers are marked invalid and excluded by generic FFN allocation. If only layer 27 is invalid, generic global 50% prunes 255,744 channels (48.2143% of all 28 layers), leaving layer 27 intact.

To explicitly use the L27-first budget with newly collected scores whose `invalid_all_zero_layers` is exactly `[27]`:

```bash
python LLaVA-NeXT/ov_workflow/scripts/prune/make_ffn_prune_config.py --ckpt /path/to/native-ov7b --metric taylor --act_scores outputs/ffn_rankings/ffn_univmax_vision_taylor.pt --token_scope vision --alloc global --ratio 0.5 --force_prune_invalid_layers --out outputs/new_ffn_L27first.json
```

This fully prunes the invalid layer first, then selects the remaining channels to reach 265,216 globally. Verify the metadata before enabling it: the flag fully prunes every marked invalid layer. All-zero statistics can arise from the loss, sample, or token scope; they are not evidence of layer redundancy. Use the frozen exporter to preserve the exact released indices.

## Recover with the joint mask

Prepare a training YAML following `ov_workflow/configs/train.example.yaml`:

```bash
MODEL_NAME_OR_PATH=/path/to/native-ov7b VISION_TOWER=/path/to/siglip DATA_PATH=/path/to/train.yaml IMAGE_FOLDER=/path/to/images HEAD_MASK_CONFIG="$(pwd)/outputs/joint_mask.json" OUTPUT_DIR="$(pwd)/outputs/ov_lora" NPROC_PER_NODE=8 bash LLaVA-NeXT/ov_workflow/scripts/train_lora.sh
```

Defaults are one epoch, LoRA rank 128, alpha 256, dropout 0.05, learning rate 1e-5, and global batch 256. `HEAD_MASK_CONFIG` is copied to the output as `ablation_config.json`. For newly searched masks, select the corresponding new joint configuration explicitly.

For Slurm, allocate resources first and set `USE_SRUN=1` and `NPROC_PER_NODE`. Existing checkpoints support resuming training with optimizer state.

## Merge, infer, evaluate

```bash
python LLaVA-NeXT/ov_workflow/merge_lora.py --base /path/to/native-ov7b --adapter outputs/ov_lora --output outputs/ov_merged
python LLaVA-NeXT/ov_workflow/infer.py --model outputs/ov_merged --image /path/to/image.jpg --prompt "Describe this image."
python lmms-eval/ov_workflow/run_eval.py --model outputs/ov_merged --output outputs/eval_14bench
```

Merging handles LoRA and `non_lora_trainables.bin`, and preserves `ablation_config.json`. Native inference still needs this fixed-mask configuration; the entrypoints load it automatically or accept `--mask`. Without a configuration, inference is dense. Unmerged adapters are supported by `infer.py --model /path/to/adapter --base /path/to/native-ov7b`.

Native attention/FFN modality masks require batch size 1 and one contiguous visual region. Multiple adjacent visual spans are accepted, but image–text–image inputs with intervening text raise a clear error under active modality masks. Dense multi-image input follows the native model behavior. Sequence truncation is checked before recording the visual region.

OV uses a Qwen2 language backbone; the `llava_qwen` architecture and SwiGLU `gate_proj` are required. MWOP-specific contributions are licensed under [Apache 2.0](../LICENSE.txt); upstream code retains its original [license](LICENSE). The exact 282K recovery-data manifest and recovered weights are not supplied.
