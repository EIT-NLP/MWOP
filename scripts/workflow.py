"""Prepare local inputs and validate native and accelerated OV workflows."""
import argparse
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_speed_model(directory):
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    text = config.get("text_config", {})
    shape = tuple(text.get(key) for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"))
    if config.get("model_type") != "llava_onevision" or shape != (28, 3584, 18944, 28, 4):
        raise ValueError("Speed testing requires HF-format OV-7B (llava_onevision). Native llava_qwen/LoRA outputs cannot be used directly.")
    if not any(directory.glob("*.safetensors")) and not any(directory.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"HF OV model weights not found: {directory}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            if any(not (directory / shard).is_file() for shard in shards):
                raise FileNotFoundError(f"Incomplete HF OV download at {directory}")


def prepare_speed(args):
    directory = Path(args.model).expanduser().resolve()
    # Reject incompatible local weights before allocating GPU memory.
    if (directory / "config.json").is_file():
        try:
            check_speed_model(directory)
        except FileNotFoundError:
            pass  # A partially downloaded HF checkpoint can be resumed below.
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Speed testing requires a BF16-capable CUDA GPU and the separate acceleration environment. See acceleration/README.md.")
    try:
        for module in ("triton.experimental.gluon", "flash_attn", "einops", "nvtx", "safetensors", "transformers.cache_utils"):
            importlib.import_module(module)
        from transformers.cache_utils import CacheLayerMixin
        from transformers import LlavaOnevisionForConditionalGeneration
        importlib.import_module("mwop.model")
        importlib.import_module("mwop.methods")
    except (ImportError, AttributeError) as error:
        raise RuntimeError("Acceleration imports failed. Use the separate environment in acceleration/README.md, not the native training environment.") from error
    try:
        check_speed_model(directory)
    except FileNotFoundError:
        if not args.download:
            raise FileNotFoundError(f"Prepare the HF-format OV checkpoint at {directory}, or set DOWNLOAD_MODELS=1.")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(directory))
        check_speed_model(directory)
    print(f"Acceleration imports and HF OV checkpoint ready: {directory}; GPU={torch.cuda.get_device_name(0)}")


def native_model(directory):
    directory = Path(directory).expanduser().resolve()
    config_file = directory / "config.json"
    if not config_file.is_file():
        raise FileNotFoundError(f"OV config.json not found: {directory}")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    shape = tuple(config.get(key) for key in ("num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"))
    native_type = config.get("model_type") == "llava_qwen" or (
        config.get("model_type") == "llava" and "LlavaQwenForCausalLM" in config.get("architectures", [])
    )
    if not native_type or shape != (28, 3584, 18944, 28, 4):
        raise ValueError("Use native OV-7B: LlavaQwenForCausalLM, model_type=llava/llava_qwen, 28 layers.")
    if not any(directory.glob("*.safetensors")) and not any(directory.glob("pytorch_model*.bin")):
        raise FileNotFoundError(f"Model weights not found: {directory}")
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index = directory / name
        if index.is_file():
            shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
            missing = [shard for shard in shards if not (directory / shard).is_file()]
            if missing:
                raise FileNotFoundError(f"Incomplete OV download; missing shards: {missing}")
    return directory, config


def prepare_model(args):
    directory = Path(args.model).expanduser().resolve()
    try:
        native_model(directory)
    except FileNotFoundError:
        if not args.download or not args.repo:
            raise FileNotFoundError(f"Prepare the native OV checkpoint at {directory}, or enable DOWNLOAD_MODELS=1.")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=args.repo, revision=args.revision, local_dir=str(directory))
    native_model(directory)
    print(f"Native OV checkpoint ready: {directory}")


def prepare_data(args):
    import yaml
    images = Path(args.images).expanduser().resolve()
    if not images.is_dir():
        raise FileNotFoundError(f"IMAGE_FOLDER is not a directory: {images}")
    if args.jsonl:
        source = Path(args.jsonl).expanduser().resolve()
        document = {"datasets": [{"json_path": str(source), "sampling_strategy": "all"}]}
    else:
        source = Path(args.yaml).expanduser().resolve()
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not document.get("datasets"):
        raise ValueError("Training YAML must contain a nonempty datasets list.")
    for dataset in document["datasets"]:
        path = Path(dataset["json_path"]).expanduser()
        if not path.is_absolute():
            path = source.parent / path
        path = path.resolve()
        dataset["json_path"] = str(path)
        if path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                samples = []
                for line in stream:
                    if line.strip():
                        samples.append(json.loads(line))
                    if len(samples) >= 32:
                        break
        elif path.suffix == ".json":
            samples = json.loads(path.read_text(encoding="utf-8"))[:32]
        else:
            raise ValueError(f"Training data must be JSONL or JSON: {path}")
        if not samples:
            raise ValueError(f"Empty training dataset: {path}")
        for sample in samples:
            turns = sample.get("conversations", [])
            if not turns or not any(turn.get("from") in ("gpt", "assistant") and turn.get("value") for turn in turns):
                raise ValueError(f"Missing supervised gpt/assistant conversation in {path}")
            if any(turn.get("from") not in ("human", "gpt", "user", "assistant", "system") or not isinstance(turn.get("value"), str) for turn in turns):
                raise ValueError(f"Use conversations with from=human/gpt or user/assistant and string value: {path}")
            names = sample.get("image", sample.get("images", []))
            if isinstance(names, str):
                names = [names]
            for name in names:
                if not (images / name).is_file():
                    raise FileNotFoundError(f"Training image not found: {images / name}")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    print(f"Training YAML ready: {output} (checked the first 32 samples per dataset)")


def check_runtime(args):
    import torch
    import transformers
    if args.kind == "speed-install":
        if torch.__version__.split("+")[0] != "2.11.0" or transformers.__version__ != "5.12.1":
            raise RuntimeError("Use INSTALL_TARGET=speed with scripts/01_install.sh.")
        for module in ("triton.experimental.gluon", "flash_attn", "einops", "nvtx", "safetensors", "mwop.model", "mwop.methods"):
            importlib.import_module(module)
        from transformers.cache_utils import CacheLayerMixin
        print(f"Acceleration imports OK: torch={torch.__version__}, transformers={transformers.__version__}")
        return
    if torch.__version__.split("+")[0] != "2.7.0" or not transformers.__version__.startswith("4.40.0"):
        raise RuntimeError("Use the native environment from scripts/01_install.sh: Torch 2.7.0 and pinned Transformers 4.40.0.dev0.")
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM
    from peft import PeftModel
    if args.kind in ("install", "train"):
        from llava.train.train import train
    if args.kind in ("install", "eval", "taylor"):
        importlib.import_module("lmms_eval.__main__")
        importlib.import_module("lmms_eval.models.simple.llava_onevision")
    if args.kind != "install":
        if not torch.cuda.is_available() or torch.cuda.device_count() < args.gpus:
            raise RuntimeError(f"Need {args.gpus} visible CUDA GPU(s). Run on a GPU node and check CUDA_VISIBLE_DEVICES.")
        if args.kind in ("train", "taylor") and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Native training/calibration uses BF16 and requires a BF16-capable GPU.")
    print(f"Runtime imports OK: torch={torch.__version__}, transformers={transformers.__version__}, visible_gpus={torch.cuda.device_count()}")


def merge_adapter(args):
    base, _ = native_model(args.base)
    adapter = Path(args.adapter).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    weights = list(adapter.glob("adapter_model.safetensors")) + list(adapter.glob("adapter_model.bin"))
    required = [adapter / "adapter_config.json", adapter / "config.json", adapter / "ablation_config.json", adapter / "non_lora_trainables.bin"]
    if not weights or any(not path.is_file() for path in required):
        raise FileNotFoundError(f"No complete MWOP LoRA export at {adapter}. Finish scripts/02_train.sh first; use its output root, not an intermediate checkpoint.")
    # Hash the small config files and adapter weights, not the 7B base weights.
    digest = hashlib.sha256()
    digest.update(str(base).encode("utf-8"))
    for path in sorted(required + weights + [base / "config.json"]):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    fingerprint = digest.hexdigest()
    marker = output / ".mwop_adapter.sha256"
    if output.exists():
        if marker.is_file() and marker.read_text().strip() == fingerprint:
            native_model(output)
            source_mask = (adapter / "ablation_config.json").read_bytes()
            if (output / "ablation_config.json").read_bytes() != source_mask:
                raise RuntimeError("Merged model mask differs from its training mask. Choose a new MERGED_DIR.")
            print(f"Reusing unchanged merged adapter: {output}")
            return
        raise FileExistsError(f"{output} already exists without a matching completed merge. Set MERGED_DIR to a new directory, or EVAL_MODEL to an existing model you wish to evaluate.")
    command = [sys.executable, str(ROOT / "LLaVA-NeXT/ov_workflow/merge_lora.py"), "--base", str(base), "--adapter", str(adapter), "--output", str(output)]
    subprocess.run(command, cwd=ROOT, check=True)
    native_model(output)
    marker.write_text(fingerprint + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    model = sub.add_parser("model", help="Download/check the native local OV checkpoint")
    model.add_argument("--model", required=True)
    model.add_argument("--repo")
    model.add_argument("--revision", default="main")
    model.add_argument("--download", action="store_true")
    model.set_defaults(func=prepare_model)
    speed = sub.add_parser("speed", help="Check the acceleration runtime and prepare HF-format OV weights")
    speed.add_argument("--model", required=True)
    speed.add_argument("--repo", default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    speed.add_argument("--revision", default="main")
    speed.add_argument("--download", action="store_true")
    speed.set_defaults(func=prepare_speed)
    vision = sub.add_parser("vision", help="Print the configured SigLIP vision tower")
    vision.add_argument("--model", required=True)
    def print_vision(args):
        _, config = native_model(args.model)
        tower = config.get("mm_vision_tower", config.get("vision_tower"))
        if not tower:
            raise ValueError("OV config has no vision tower. Set VISION_TOWER explicitly.")
        print(tower)
    vision.set_defaults(func=print_vision)
    data = sub.add_parser("data", help="Check data inputs and write an absolute-path YAML")
    data.add_argument("--yaml")
    data.add_argument("--jsonl")
    data.add_argument("--images", required=True)
    data.add_argument("--output", required=True)
    data.set_defaults(func=prepare_data)
    check = sub.add_parser("check", help="Validate imports and optionally CUDA availability")
    check.add_argument("--kind", choices=["install", "train", "eval", "taylor", "speed-install"], required=True)
    check.add_argument("--gpus", type=int, default=1)
    check.set_defaults(func=check_runtime)
    merge = sub.add_parser("merge", help="Merge a trained adapter or reuse its unchanged export")
    merge.add_argument("--base", required=True)
    merge.add_argument("--adapter", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(func=merge_adapter)
    args = parser.parse_args()
    if args.command == "data" and not args.jsonl and not args.yaml:
        parser.error("data requires --jsonl or --yaml")
    args.func(args)


if __name__ == "__main__":
    main()
