# OV-only evaluation

本目录提供 OV 模型入口 `llava_onevision`，以及评测框架、任务工具与 YAML 配置。模型实现位于 `lmms_eval/models/simple/llava_onevision.py`，依赖同级 `LLaVA-NeXT`。

从仓库根目录，在 LLaVA 训练/评测环境中执行：

```bash
python lmms-eval/ov_workflow/run_eval.py --model /path/to/native-ov-merged --output outputs/eval_14bench
```

默认 14bench：GQA、VQAv2、OK-VQA、TextVQA、DocVQA、OCRBench、ScienceQA-IMG、AI2D、MMStar、RefCOCO A/B、RefCOCO+ A/B、RefCOCOg。可用 `--tasks` 传逗号分隔的任务名，`--limit` 设置评测样本数。

多图与视频的七项任务入口如下（Video-MME 两种字幕设置分别计一项）：

```bash
python lmms-eval/ov_workflow/run_eval.py --model /path/to/native-ov-merged --tasks qbench2_dev,mantis,blink,mvbench,videomme,videomme_w_subtitle,nextqa_mc_test --video-frames 32 --output outputs/eval_multi_video
```

具体可用任务名以包内 YAML 的 `task`/`group` 为准。数据集需要自行下载或设置缓存；默认 YAML 已使用公开数据集标识；若使用离线数据，需要自行设置本地位置。任务目录提供数据处理和评分代码，数据集与模型权重需单独准备。

如已有 Hugging Face 数据集的本地副本，在 `scripts/config.sh` 中设置 `LMMS_DATA_ROOT=/path/to/benchmarks`。加载器优先使用对应名称的本地子目录，找不到时使用 YAML 中的公开数据集地址。14 个默认任务所需目录名为 `GQA`、`VQAv2`、`OK-VQA`、`textvqa`、`DocVQA`、`OCRBench`、`ScienceQA`、`ai2d`、`MMStar`、`RefCOCO`、`RefCOCOplus`、`RefCOCOg`。本地目录应保留原始 Hugging Face 数据集布局和配置名称；本设置同时适用于 Taylor 重要性计算。公开任务不再强制要求预先登录 Hugging Face。

入口自动读取模型目录中的 `ablation_config.json`；也可传 `--mask /path/to/mask.json`。请使用合并后的原生 OV checkpoint；本入口不负责将 adapter 合并。需要完整 CLI 时可从本目录运行 `python -m lmms_eval --help`。

原生模态掩码要求 batch=1 且视觉 token 位于连续区间。多图任务中若图像区间之间有文本，开启模态掩码时会明确报错；Dense 模式保留正常多图行为。请按实际任务输入格式选择配置，不要将连续区间内核用于交错多图输入。

MWOP 自研贡献采用 [Apache 2.0](../LICENSE.txt)。上游 lmms-eval 的主评测框架保留 MIT 许可，模型与任务代码保留 Apache 2.0 许可，具体范围见 [LICENSE](LICENSE)。
