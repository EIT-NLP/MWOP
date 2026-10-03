from __future__ import annotations
import argparse
import collections
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple
import torch
VERSION = 2
SCORE_KEYS = ('token_all', 'token_vision', 'token_text', 'structured_all', 'structured_vision', 'structured_text')

def _add_paths(project_root: Path) -> None:
    for path in (project_root.parent, project_root / 'scripts' / 'prune'):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)

def _sha256_file(path: Optional[Path]) -> str:
    if path is None or not path.is_file():
        return ''
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _atomic_json_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)

def _task_object(task: str) -> Tuple[Any, Any]:
    from lmms_eval.tasks import TaskManager, get_task_dict
    task_dict = get_task_dict([task], TaskManager())
    obj = task_dict[task]
    while isinstance(obj, (tuple, list)):
        obj = obj[-1]
    if isinstance(obj, dict):
        obj = next(iter(obj.values()))
    has_test = getattr(obj, 'has_test_docs', lambda: False)
    docs = obj.test_docs() if has_test() else obj.validation_docs()
    return (obj, docs)

def _first_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ''
    return str(value).strip()

def _prepare_doc(obj: Any, doc: Any) -> Tuple[Any, str, str]:
    visual = obj.doc_to_visual(doc)
    if isinstance(visual, (list, tuple)):
        images = list(visual)
    elif visual is None:
        images = []
    else:
        images = [visual]
    if len(images) != 1:
        raise ValueError(f'expected exactly one image, got {len(images)}')
    question = _first_text(obj.doc_to_text(doc))
    answer = _first_text(obj.doc_to_target(doc))
    if not question or not answer:
        raise ValueError('empty question or target')
    return (images[0], question, answer)

def _empty_accumulators() -> Tuple[Dict[str, Dict[int, torch.Tensor]], Dict[str, Dict[int, int]]]:
    return ({key: {} for key in SCORE_KEYS}, {key: {} for key in SCORE_KEYS})

def _commit_stage(sums: MutableMapping[str, MutableMapping[int, torch.Tensor]], counts: MutableMapping[str, MutableMapping[int, int]], stage: Mapping[int, Mapping[str, Tuple[torch.Tensor, int]]]) -> None:
    for layer, values in stage.items():
        for key, (vector, count) in values.items():
            vector = vector.detach().to(device='cpu', dtype=torch.float64)
            if layer not in sums[key]:
                sums[key][layer] = torch.zeros_like(vector)
                counts[key][layer] = 0
            sums[key][layer].add_(vector)
            counts[key][layer] += int(count)

def _normalise(sums: Mapping[str, Mapping[int, torch.Tensor]], counts: Mapping[str, Mapping[int, int]], key: str) -> Dict[int, torch.Tensor]:
    result: Dict[int, torch.Tensor] = {}
    for layer in sorted(sums[key]):
        count = int(counts[key].get(layer, 0))
        if count <= 0:
            raise RuntimeError(f'zero count for {key}, layer {layer}')
        result[layer] = (sums[key][layer] / count).to(torch.float32)
    return result

def _taylor_contributions(activation: torch.Tensor, gradient: torch.Tensor, visual_start: int, visual_end: int) -> Dict[str, Tuple[torch.Tensor, int]]:
    if activation.shape != gradient.shape or activation.ndim != 3 or activation.shape[0] != 1:
        raise ValueError(f'incompatible h/grad shapes: {activation.shape} vs {gradient.shape}')
    total_tokens = int(activation.shape[1])
    if not 0 <= visual_start < visual_end <= total_tokens:
        raise ValueError(f'invalid visual range [{visual_start},{visual_end}) for T={total_tokens}')
    visual_tokens = visual_end - visual_start
    text_tokens = total_tokens - visual_tokens
    if text_tokens <= 0:
        raise ValueError('sample contains no text tokens')
    product = activation.detach().float() * gradient.detach().float()
    absolute = product.abs()
    all_abs = absolute.sum(dim=(0, 1))
    vision_abs = absolute[:, visual_start:visual_end].sum(dim=(0, 1))
    text_abs = absolute[:, :visual_start].sum(dim=(0, 1)) + absolute[:, visual_end:].sum(dim=(0, 1))
    all_signed = product.sum(dim=(0, 1))
    vision_signed = product[:, visual_start:visual_end].sum(dim=(0, 1))
    text_signed = product[:, :visual_start].sum(dim=(0, 1)) + product[:, visual_end:].sum(dim=(0, 1))
    return {'token_all': (all_abs, total_tokens), 'token_vision': (vision_abs, visual_tokens), 'token_text': (text_abs, text_tokens), 'structured_all': (all_signed.abs(), 1), 'structured_vision': (vision_signed.abs(), 1), 'structured_text': (text_signed.abs(), 1)}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--pretrained', required=True, help='merged Hugging Face checkpoint directory')
    parser.add_argument('--ablation_config', required=True, help='the exact V2V/T2V/T2T runtime mask JSON used during recovery training')
    parser.add_argument('--task', required=True, help='lmms-eval task name')
    parser.add_argument('--n_samples', type=int, default=200, help='number of successful samples required')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--conv_template', default='qwen_1_5')
    parser.add_argument('--device', default='auto', help='device_map value; default shards over visible GPUs')
    parser.add_argument('--attn_implementation', default='sdpa', choices=('sdpa', 'eager'))
    parser.add_argument('--max_image_side', type=int, default=0, help='optional longest-side cap; 0 preserves benchmark images exactly')
    parser.add_argument('--checkpoint_every', type=int, default=10, help='save resumable state every N successful samples; 0 disables periodic saves')
    parser.add_argument('--out', required=True)
    parser.add_argument('--force', action='store_true', help='ignore existing output/partial state')
    return parser.parse_args()

def main() -> int:
    args = _parse_args()
    if args.n_samples <= 0:
        raise SystemExit('--n_samples must be positive')
    project_root = Path(os.environ.get('PROJECT_ROOT', os.getcwd())).resolve()
    _add_paths(project_root)
    pretrained = Path(args.pretrained).expanduser().resolve()
    ablation_config = Path(args.ablation_config).expanduser().resolve()
    output_path = Path(args.out).expanduser().resolve()
    partial_path = output_path.with_name(output_path.name + '.partial.pt')
    progress_path = output_path.with_name(output_path.name + '.progress.json')
    if not (pretrained / 'config.json').is_file():
        raise FileNotFoundError(f'merged checkpoint config.json not found: {pretrained}')
    if not ablation_config.is_file():
        raise FileNotFoundError(f'ablation config not found: {ablation_config}')
    config_hash = _sha256_file(pretrained / 'config.json')
    mask_hash = _sha256_file(ablation_config)
    fingerprint_payload = {'version': VERSION, 'pretrained': str(pretrained), 'model_config_sha256': config_hash, 'ablation_config': str(ablation_config), 'ablation_sha256': mask_hash, 'task': args.task, 'target_successful': args.n_samples, 'seed': args.seed, 'conv_template': args.conv_template, 'max_image_side': args.max_image_side, 'attn_implementation': args.attn_implementation}
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode('utf-8')).hexdigest()
    if args.force:
        print('[ffn-taylor] --force: existing output/partial state will be replaced', flush=True)
    elif output_path.is_file():
        existing = torch.load(output_path, map_location='cpu')
        if existing.get('fingerprint') != fingerprint:
            raise RuntimeError(f'existing output has a different fingerprint: {output_path}; use --force or another path')
        if int(existing.get('num_samples', 0)) >= args.n_samples:
            _atomic_json_save({'status': 'complete', 'task': args.task, 'target_successful': args.n_samples, 'successful': int(existing.get('num_samples', 0)), 'attempted': int(existing.get('attempted', 0)), 'skipped': existing.get('skipped', {}), 'output': str(output_path), 'partial': None, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z')}, progress_path)
            print(f'[ffn-taylor] already complete: {output_path}', flush=True)
            return 0
    task_obj, docs = _task_object(args.task)
    doc_order = list(range(len(docs)))
    random.Random(args.seed).shuffle(doc_order)
    print(f'[ffn-taylor] task={args.task} docs={len(docs)} target_successful={args.n_samples}', flush=True)
    sums, counts = _empty_accumulators()
    next_position = successful = attempted = 0
    skipped: collections.Counter[str] = collections.Counter()
    successful_doc_indices: List[int] = []
    if not args.force and partial_path.is_file():
        saved = torch.load(partial_path, map_location='cpu')
        if saved.get('fingerprint') != fingerprint:
            raise RuntimeError(f'partial checkpoint fingerprint mismatch: {partial_path}; use --force or another path')
        sums = saved['sums']
        counts = saved['counts']
        next_position = int(saved['next_position'])
        successful = int(saved['successful'])
        attempted = int(saved['attempted'])
        skipped.update(saved.get('skipped', {}))
        successful_doc_indices = [int(i) for i in saved.get('successful_doc_indices', [])]
        print(f'[ffn-taylor] resume position={next_position}/{len(doc_order)} successful={successful}', flush=True)
    from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.ablation_registry import apply_ablation_config
    if args.conv_template not in conv_templates:
        raise KeyError(f'unknown conversation template: {args.conv_template}')
    device_map = 'auto' if args.device in ('', 'auto', None) else args.device
    print(f'[ffn-taylor] loading merged model: {pretrained}', flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(str(pretrained), None, get_model_name_from_path(str(pretrained)), device_map=device_map, attn_implementation=args.attn_implementation, multimodal=True, torch_dtype='bfloat16')
    model.eval()
    model.requires_grad_(False)
    mask_summary = apply_ablation_config(model, str(ablation_config))
    print(f'[ffn-taylor] re-applied runtime mask: {mask_summary}', flush=True)
    embedding_device = model.get_input_embeddings().weight.device
    model_dtype = next(model.parameters()).dtype

    def enable_embedding_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
        return output
    embedding_handle = model.get_input_embeddings().register_forward_hook(enable_embedding_grad)
    import re
    down_pattern = re.compile('(?:^|\\.)layers\\.(\\d+)\\.mlp\\.down_proj$')
    layer_modules: Dict[int, torch.nn.Module] = {}
    for name, module in model.named_modules():
        match = down_pattern.search(name)
        if match:
            layer_modules[int(match.group(1))] = module
    expected_layers = sorted(layer_modules)
    if not expected_layers:
        raise RuntimeError("found no '*.layers.N.mlp.down_proj' modules")
    if expected_layers != list(range(len(expected_layers))):
        raise RuntimeError(f'non-contiguous FFN layers: {expected_layers}')
    print(f'[ffn-taylor] hooked {len(expected_layers)} FFN layers', flush=True)
    context: Dict[str, Any] = {'active': False, 'stage': {}, 'errors': []}
    hook_handles: List[Any] = []

    def make_pre_hook(layer: int):

        def pre_hook(_module: Any, inputs: Tuple[torch.Tensor, ...]) -> None:
            if not context.get('active'):
                return
            h = inputs[0]
            if not isinstance(h, torch.Tensor) or h.ndim != 3 or h.shape[0] != 1:
                context['errors'].append(f"L{layer}: unexpected down_proj input shape {getattr(h, 'shape', None)}")
                return
            if not h.requires_grad:
                context['errors'].append(f'L{layer}: down_proj input does not require grad')
                return
            total_tokens = int(h.shape[1])
            placeholder_position = int(context['placeholder_position'])
            original_tokens = int(context['original_tokens'])
            visual_tokens = total_tokens - original_tokens + 1
            visual_start = placeholder_position
            visual_end = visual_start + visual_tokens
            if not (visual_tokens > 0 and 0 <= visual_start < visual_end <= total_tokens):
                context['errors'].append(f'L{layer}: invalid visual range [{visual_start},{visual_end}) for T={total_tokens}, original={original_tokens}')
                return
            h_detached = h.detach()

            def on_grad(gradient: torch.Tensor) -> torch.Tensor:
                try:
                    values = _taylor_contributions(h_detached, gradient, visual_start, visual_end)
                    context['stage'][layer] = {key: (vector.cpu(), count) for key, (vector, count) in values.items()}
                except Exception as exc:
                    context['errors'].append(f'L{layer}: contribution error: {exc}')
                return gradient
            h.register_hook(on_grad)
        return pre_hook
    for layer, module in layer_modules.items():
        hook_handles.append(module.register_forward_pre_hook(make_pre_hook(layer)))
    stop_requested = {'value': False}

    def request_stop(signum: int, _frame: Any) -> None:
        stop_requested['value'] = True
        print(f'[ffn-taylor] signal {signum}: stopping after current sample', flush=True)
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (OSError, ValueError):
            pass

    def save_progress(status: str) -> None:
        _atomic_json_save({'status': status, 'task': args.task, 'target_successful': args.n_samples, 'successful': successful, 'attempted': attempted, 'next_position': next_position, 'total_docs': len(doc_order), 'skipped': dict(skipped), 'output': str(output_path), 'partial': str(partial_path) if partial_path.exists() else None, 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S %z')}, progress_path)

    def save_partial(status: str='partial') -> None:
        state = {'version': VERSION, 'fingerprint': fingerprint, 'fingerprint_payload': fingerprint_payload, 'next_position': next_position, 'successful': successful, 'attempted': attempted, 'skipped': dict(skipped), 'successful_doc_indices': successful_doc_indices, 'sums': sums, 'counts': counts}
        _atomic_torch_save(state, partial_path)
        save_progress(status)
    save_progress('running')
    started = time.time()
    try:
        while successful < args.n_samples and next_position < len(doc_order):
            doc_index = doc_order[next_position]
            next_position += 1
            attempted += 1
            outputs = loss = image_tensor = ids_qa = labels = None
            context['active'] = False
            context['stage'] = {}
            context['errors'] = []
            try:
                image, question, answer = _prepare_doc(task_obj, docs[doc_index])
                if args.max_image_side > 0 and hasattr(image, 'size') and hasattr(image, 'thumbnail') and (max(image.size) > args.max_image_side):
                    image = image.copy()
                    image.thumbnail((args.max_image_side, args.max_image_side))
                user_message = question if DEFAULT_IMAGE_TOKEN in question else DEFAULT_IMAGE_TOKEN + '\n' + question
                prompt_only = conv_templates[args.conv_template].copy()
                prompt_only.append_message(prompt_only.roles[0], user_message)
                prompt_only.append_message(prompt_only.roles[1], None)
                prompt_answer = conv_templates[args.conv_template].copy()
                prompt_answer.append_message(prompt_answer.roles[0], user_message)
                prompt_answer.append_message(prompt_answer.roles[1], answer)
                ids_q = tokenizer_image_token(prompt_only.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(embedding_device)
                ids_qa = tokenizer_image_token(prompt_answer.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(embedding_device)
                placeholders = (ids_qa[0] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
                if int(placeholders.numel()) != 1:
                    raise ValueError(f'expected one image placeholder, got {int(placeholders.numel())}')
                labels = ids_qa.clone()
                labels[:, :ids_q.shape[1]] = IGNORE_INDEX
                if int((labels != IGNORE_INDEX).sum().item()) <= 0:
                    raise ValueError('no supervised answer tokens after prompt masking')
                image_tensor = process_images([image], image_processor, model.config)
                if isinstance(image_tensor, list):
                    image_tensor = [item.to(device=embedding_device, dtype=model_dtype) for item in image_tensor]
                else:
                    image_tensor = image_tensor.to(device=embedding_device, dtype=model_dtype)
                context['placeholder_position'] = int(placeholders[0].item())
                context['original_tokens'] = int(ids_qa.shape[1])
                context['active'] = True
                model.zero_grad(set_to_none=True)
                outputs = model(input_ids=ids_qa, labels=labels, images=image_tensor, image_sizes=[[int(image.size[0]), int(image.size[1])]], use_cache=False)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
                if loss is None or not bool(torch.isfinite(loss).item()):
                    raise FloatingPointError(f'non-finite loss: {loss}')
                if context['errors']:
                    raise RuntimeError('; '.join(context['errors'][:3]))
                loss.backward()
                if context['errors']:
                    raise RuntimeError('; '.join(context['errors'][:3]))
                stage = context['stage']
                if sorted(stage) != expected_layers:
                    missing = sorted(set(expected_layers) - set(stage))
                    raise RuntimeError(f'incomplete backward: got {len(stage)}/{len(expected_layers)} layers; missing={missing}')
                for layer in expected_layers:
                    for key in SCORE_KEYS:
                        vector, count = stage[layer][key]
                        if count <= 0 or not bool(torch.isfinite(vector).all().item()):
                            raise FloatingPointError(f'invalid contribution: layer={layer} key={key}')
                _commit_stage(sums, counts, stage)
                successful += 1
                successful_doc_indices.append(doc_index)
                if successful <= 3 or successful % 10 == 0:
                    elapsed = time.time() - started
                    print(f'[ffn-taylor] task={args.task} successful={successful}/{args.n_samples} attempted={attempted} elapsed={elapsed / 60:.1f}m loss={float(loss):.4g}', flush=True)
                if args.checkpoint_every > 0 and successful % args.checkpoint_every == 0:
                    save_partial()
            except Exception as exc:
                reason = 'oom' if isinstance(exc, torch.cuda.OutOfMemoryError) else type(exc).__name__
                skipped[reason] += 1
                if sum(skipped.values()) <= 10 or skipped[reason] <= 3:
                    print(f'[ffn-taylor] skip doc={doc_index} reason={reason}: {exc}', flush=True)
            finally:
                context['active'] = False
                context['stage'] = {}
                outputs = loss = image_tensor = ids_qa = labels = None
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if stop_requested['value']:
                save_partial('interrupted')
                print(f'[ffn-taylor] interrupted; resume state -> {partial_path}', flush=True)
                return 75
    finally:
        for handle in hook_handles:
            handle.remove()
        embedding_handle.remove()
    if successful < args.n_samples:
        save_partial()
        raise RuntimeError(f'task {args.task} exhausted {len(doc_order)} docs with only {successful}/{args.n_samples} successful samples')
    scores = _normalise(sums, counts, 'token_all')
    scores_vision = _normalise(sums, counts, 'token_vision')
    scores_text = _normalise(sums, counts, 'token_text')
    scores_structured = _normalise(sums, counts, 'structured_all')
    scores_structured_vision = _normalise(sums, counts, 'structured_vision')
    scores_structured_text = _normalise(sums, counts, 'structured_text')
    intermediate_size = int(next(iter(scores.values())).numel())
    result = {'version': VERSION, 'metric': 'ffn_taylor_token_abs_and_structured_channel', 'fingerprint': fingerprint, 'fingerprint_payload': fingerprint_payload, 'task': args.task, 'num_samples': successful, 'attempted': attempted, 'skipped': dict(skipped), 'successful_doc_indices': successful_doc_indices, 'layers': len(scores), 'intermediate_size': intermediate_size, 'pretrained': str(pretrained), 'model_config_sha256': config_hash, 'ablation_config': str(ablation_config), 'ablation_sha256': mask_hash, 'mask_summary': str(mask_summary), 'scores': scores, 'scores_vision': scores_vision, 'scores_text': scores_text, 'scores_structured': scores_structured, 'scores_structured_vision': scores_structured_vision, 'scores_structured_text': scores_structured_text, 'counts': counts, 'elapsed_seconds': time.time() - started}
    _atomic_torch_save(result, output_path)
    if partial_path.exists():
        partial_path.unlink()
    save_progress('complete')
    print(f'[ffn-taylor] DONE task={args.task} successful={successful}/{attempted} shape={len(scores)}x{intermediate_size} -> {output_path}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
