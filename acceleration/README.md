# OV-7B acceleration

This directory provides Triton attention, structured visual FFN execution, fixed token schedules, numerical checks, and CUDA Graph timing for the OV-7B decoder.

## Environment and model

Use Linux with a CUDA GPU, PyTorch, Transformers, Triton, FlashAttention 2, einops, nvtx, and safetensors. The implementation uses `triton.experimental.gluon` and the recent Transformers cache layer API. Configure a separate environment from native OV training.

The model must use HF `model_type=llava_onevision` format, with 28 decoder layers, hidden size 3584, intermediate size 18944, 28 query heads, and 4 KV heads. Native `llava_qwen` checkpoints require conversion before loading; a conversion utility and recovered HF weights are not supplied.

[pyproject.toml](pyproject.toml) provides the separate acceleration environment for Python 3.12. A fresh installation with public `einops==0.8.2` and compiled FlashAttention 2.8.3 passed OV decoder numerical checks and the MWOP, MWOP + PyramidDrop, and MWOP + ZOO speed runs on A100. This execution trial used the supplied tile plan, 10 repetitions, and one timing sample; use the normal defaults for more stable measurements. It does not install the native OV training package.

The original experiment environment is retained in [environment.reference.json](environment.reference.json): PyTorch 2.11.0+cu130, CUDA 13.0, Triton 3.6.0, Transformers 5.12.1, FlashAttention 2.8.3, einops 0.9.0.dev0, nvtx 0.2.16, and safetensors 0.8.0. Its development einops source is unspecified; the installable profile replaces that dependency with the checked public release. New measurement manifests record the actual runtime versions.

Install from the repository root on Linux x86_64 with Conda and GCC/G++:

```bash
INSTALL_TARGET=speed INSTALL_CUDA_TOOLKIT=1 bash scripts/01_install.sh
```

The installer creates `SPEED_ENV_NAME` (default `mwop-ov-speed`), installs CUDA 13.0 PyTorch wheels, the acceleration profile, and checks imports and package metadata. `INSTALL_CUDA_TOOLKIT=1` installs NVIDIA compiler wheels inside this environment and configures them for FlashAttention. With an existing CUDA 13 toolkit, omit it and set `CUDA_HOME`. The GPU driver must support this CUDA runtime. First-time FlashAttention source compilation can take tens of minutes. For an A100-only source build, add `FLASH_ATTN_CUDA_ARCHS=80 MAX_JOBS=8 NVCC_THREADS=2` to the command; use the appropriate architecture when building for another GPU.

## Run

From the repository root:

```bash
bash scripts/05_speed.sh
```

Set `SPEED_ENV_NAME` or `SPEED_VENV_PATH`, `SPEED_MODEL`, and optionally `SPEED_OUTPUT` in `scripts/config.sh`; see the [root guide](../README.md#usage). The wrapper downloads missing HF weights when `DOWNLOAD_MODELS=1`, runs the serial benchmark, and exports logical and configured packed FLOPs. It does not install the separate acceleration environment. `--help` and `--dry-run` are available.

For direct Python invocation in an already active acceleration environment:

```bash
python acceleration/run_serial.py --model /path/to/hf-ov7b --alignment 64 --tiles acceleration/configs/mwop_validated_tiles.json --output outputs/ov_speed
python acceleration/theory_ov.py --ffn-alignment 1 --output outputs/ov_theory_logical.json
python acceleration/theory_ov.py --ffn-alignment 64 --output outputs/ov_theory_packed64.json
python acceleration/summarize.py --results outputs/ov_speed
```

The default order is Dense/MWOP, MWOP + PyramidDrop, then MWOP + ZOO. Each command completes before the next starts; a failure stops the run. First use compiles the kernels. Use a new output directory for each experiment. `--rep` and `--samples` control timing samples; `--methods` selects methods. Runs including MWOP produce `REPORT.md` and `summary.json` with logical and packed FLOPs. Each method's packing alignment is read from its measurement manifest. Runtime manifests record software versions, GPU details, configuration, and source hashes.

The wrapper and direct Python CLI default to alignment 64, rounding retained FFN widths up and padding added weight entries with zeros. This preserves the mask while improving matrix-multiplication efficiency. `--alignment 1` uses the exact retained widths; in the A100 execution trial it reduced FLOPs slightly but slowed the FFN enough that MWOP was slower than Dense. `05_speed.sh` uses the supplied `configs/mwop_validated_tiles.json` attention tile plan by default; override it with `SPEED_TILES`. For direct Python commands, select this plan explicitly with `--tiles acceleration/configs/mwop_validated_tiles.json`.

## Pruning configuration

The fixed masks prune V2V/T2V/T2T attention at 40%/60%/10% globally, and 50% of visual FFN channels globally. Text FFN channels remain dense. Per-layer indices are in `pruning_config/llava_onevision`; individual layers can have different pruning ratios.

- MWOP + ZOO retains 1607 of 3215 visual tokens before the decoder.
- MWOP + PyramidDrop uses 3215 → 1747 → 950 → 516 visual tokens. Reductions occur at the entrance to zero-based layers 7, 14, and 21, with seven layers in each stage.

The token and channel indices are fixed. Online ZOO/PyramidDrop scoring and sorting are excluded.

## Timing and FLOPs

The timing input is batch=1, prefix=18, visual=3215, suffix=19. Decoder prefill includes fresh KV, final normalization, the last-token LM head, and argmax. It excludes the vision encoder and projector. Attention/FFN component times include their normalization, residual, and projection operations. Component and complete-decoder timing are measured separately, so their sums can differ.

`theory_ov.py` calculates GQA FLOPs with V=3215, T=19, dKV=512, and causal pair count C(n)=n(n+1)/2. Both logical and packed calculations exclude the 18-token system prefix, probes, and LM head. Timing includes the prefix, so these FLOPs are not the complete timed operation count.

`--ffn-alignment 1` gives logical channel widths. `--ffn-alignment 64` adds padding using the same per-layer width rounding as `VisionMLP`, including the layer-dependent visual lengths of both compositions. Every row includes `logical_total_tflops` and `ffn_padding_tflops`.

Dense totals 44.305328922624 TFLOPs. Frozen MWOP has 25.123692657664 logical TFLOPs; 64-channel alignment adds 0.048671293440 and gives 25.172363951104, rounding to 25.17 in the [paper table](https://arxiv.org/abs/2610.01434). This numerical agreement does not establish the historical setting. The supplied environment reference records alignment 1; retain the manifest's actual setting when reporting new measurements.

`summarize.py` always reports logical FLOPs in its original component columns and adds `Packed Total` and `FFN Alignment` from each method's manifest. `summary.json` preserves logical fields and adds `packed_ffn_tflops`, `packed_total_tflops`, and `ffn_padding_tflops`. A missing alignment raises an error instead of silently assuming a packing setting.

The benchmark's own `theory.json` includes the prefix and packed widths. Report its scope separately from the prefix-excluding calculation above.

## Mask checks

```bash
python acceleration/integrated_checks/ffn_mask/test_structured_ffn_backend.py
python acceleration/integrated_checks/region_flex/test_flex_mask_logic.py
```

The sibling `LLaVA-NeXT/llava/model/language_model/` plugins provide the fixed-mask reference implementations used by these checks and the GPU microbenchmarks.

## Source

The acceleration framework builds on [PruningInferSim](https://github.com/EIT-NLP/LLM-Pruning/tree/main/PruningInferSim). Its source snapshot is recorded in [provenance.json](provenance.json).
