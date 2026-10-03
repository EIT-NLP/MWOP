import argparse
from pathlib import Path
import _bootstrap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--base')
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--image', nargs='+', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--device-map', default='auto')
    parser.add_argument('--max-new-tokens', type=int, default=128)
    args = parser.parse_args()
    import torch
    from PIL import Image
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.ablation_registry import apply_ablation_config
    name = 'llava-onevision-qwen2-7b-ov-lora' if args.base else 'llava-onevision-qwen2-7b-ov'
    tokenizer, model, processor, _ = load_pretrained_model(args.model, args.base, name, device_map=args.device_map, torch_dtype='bfloat16', attn_implementation='sdpa', multimodal=True)
    model.eval()
    mask = args.mask or Path(args.model) / 'ablation_config.json'
    if args.mask and not mask.is_file():
        raise FileNotFoundError(mask)
    if mask.is_file():
        print(apply_ablation_config(model, str(mask)).describe())
    images = [Image.open(path).convert('RGB') for path in args.image]
    conversation = conv_templates['qwen_1_5'].copy()
    conversation.append_message(conversation.roles[0], '\n'.join([DEFAULT_IMAGE_TOKEN] * len(images)) + '\n' + args.prompt)
    conversation.append_message(conversation.roles[1], None)
    device = model.get_input_embeddings().weight.device
    image_tensor = process_images(images, processor, model.config)
    dtype = next(model.get_vision_tower().parameters()).dtype
    image_tensor = [tensor.to(device=device, dtype=dtype) for tensor in image_tensor] if isinstance(image_tensor, list) else image_tensor.to(device=device, dtype=dtype)
    ids = tokenizer_image_token(conversation.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    with torch.inference_mode():
        result = model.generate(ids, images=image_tensor, image_sizes=[image.size for image in images], do_sample=False, max_new_tokens=args.max_new_tokens, use_cache=True)
    print(tokenizer.batch_decode(result, skip_special_tokens=True)[0].strip())

if __name__ == '__main__':
    main()
