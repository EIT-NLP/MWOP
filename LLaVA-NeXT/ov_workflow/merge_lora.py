import argparse
import shutil
from pathlib import Path
import _bootstrap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mask', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from llava.model.language_model.llava_qwen import LlavaQwenForCausalLM, LlavaQwenConfig
    tokenizer = AutoTokenizer.from_pretrained(args.base, use_fast=False)
    # Preserve the training-time vision tower and multimodal configuration.
    config_source = args.adapter if (args.adapter / 'config.json').is_file() else args.base
    config = LlavaQwenConfig.from_pretrained(config_source)
    model = LlavaQwenForCausalLM.from_pretrained(args.base, config=config, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='cpu')
    additional = args.adapter / 'non_lora_trainables.bin'
    if additional.is_file():
        state = torch.load(additional, map_location='cpu', weights_only=True)
        state = {(key[11:] if key.startswith('base_model.') else key): value for key, value in state.items()}
        if any(key.startswith('model.model.') for key in state):
            state = {(key[6:] if key.startswith('model.') else key): value for key, value in state.items()}
        model.load_state_dict(state, strict=False)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    mask = args.mask or args.adapter / 'ablation_config.json'
    if mask.is_file():
        shutil.copy2(mask, args.output / 'ablation_config.json')

if __name__ == '__main__':
    main()
