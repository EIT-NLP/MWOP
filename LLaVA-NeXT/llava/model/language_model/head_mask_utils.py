import json
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import torch

def head_mask_decode_enabled() -> bool:
    raw = os.environ.get('HEAD_MASK_DECODE', os.environ.get('REGION_MASK_DECODE', '0')).strip().lower()
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    raise ValueError(f'HEAD_MASK_DECODE must be one of 0/1/false/true/no/yes/off/on, got {raw!r}')

@dataclass(frozen=True)
class HeadMaskSummary:
    head_count: int
    layer_count: int
    applied_module_count: int

def _normalize_heads(heads: Iterable[Sequence[int]]) -> Dict[int, Tuple[int, ...]]:
    by_layer: Dict[int, set] = {}
    for item in heads:
        if len(item) != 2:
            raise ValueError(f'Each head entry must be [layer, head], got {item!r}')
        layer_idx, head_idx = (int(item[0]), int(item[1]))
        by_layer.setdefault(layer_idx, set()).add(head_idx)
    return {layer_idx: tuple(sorted(head_set)) for layer_idx, head_set in sorted(by_layer.items())}

def load_zero_head_config(config_path: str) -> Dict[int, Tuple[int, ...]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    ablation_type = config.get('ablation_type', 'zero')
    if ablation_type != 'zero':
        raise ValueError(f"head_mask_utils only supports ablation_type='zero', got {ablation_type!r}. Use a dedicated plugin for other ablation types.")
    heads = config.get('heads', [])
    return _normalize_heads(heads)

def zero_configured_head_outputs(attn_output: torch.Tensor, zero_head_indices: Optional[Sequence[int]], mask_enabled: bool=True) -> torch.Tensor:
    if not mask_enabled or zero_head_indices is None:
        return attn_output
    if isinstance(zero_head_indices, torch.Tensor):
        if zero_head_indices.numel() == 0:
            return attn_output
    elif len(zero_head_indices) == 0:
        return attn_output
    if attn_output.dim() != 4:
        raise ValueError(f'Expected attention output shape [batch, heads, seq, head_dim], got {tuple(attn_output.shape)}')
    head_indices = torch.as_tensor(zero_head_indices, dtype=torch.long, device=attn_output.device)
    masked = attn_output.clone()
    masked.index_fill_(1, head_indices, 0)
    return masked
_LAYER_PATTERN = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.self_attn$')

def apply_zero_heads(model: torch.nn.Module, heads: Iterable[Sequence[int]]) -> HeadMaskSummary:
    heads_by_layer = _normalize_heads(heads)
    applied_module_count = 0
    for name, module in model.named_modules():
        match = _LAYER_PATTERN.search(name)
        if not match:
            continue
        layer_idx = int(getattr(module, 'layer_idx', match.group(1)))
        if layer_idx not in heads_by_layer:
            continue
        num_heads = getattr(module, 'num_heads', None)
        if num_heads is not None and any((head_idx < 0 or head_idx >= int(num_heads) for head_idx in heads_by_layer[layer_idx])):
            raise ValueError(f'Head indices {heads_by_layer[layer_idx]} out of range for layer {layer_idx} with {num_heads} heads')
        existing = tuple(getattr(module, '_ov_zero_head_indices', ()) or ())
        merged = tuple(sorted(set(existing) | set(heads_by_layer[layer_idx])))
        setattr(module, '_ov_zero_head_indices', merged)
        applied_module_count += 1
    if applied_module_count != len(heads_by_layer):
        missing = sorted((layer_idx for layer_idx in heads_by_layer if not any((int(getattr(module, 'layer_idx', -1)) == layer_idx and set(heads_by_layer[layer_idx]).issubset(set(getattr(module, '_ov_zero_head_indices', ()) or ())) for module in model.modules()))))
        raise ValueError(f'Failed to apply zero head mask to layers: {missing}')
    return HeadMaskSummary(head_count=sum((len(heads) for heads in heads_by_layer.values())), layer_count=len(heads_by_layer), applied_module_count=applied_module_count)

def apply_zero_head_mask_config(model: torch.nn.Module, config_path: str) -> HeadMaskSummary:
    heads_by_layer = load_zero_head_config(config_path)
    flat = [[li, hi] for li, hs in heads_by_layer.items() for hi in hs]
    return apply_zero_heads(model, flat)

def clear_zero_head_mask(model: torch.nn.Module) -> int:
    cleared = 0
    for module in model.modules():
        if hasattr(module, '_ov_zero_head_indices'):
            delattr(module, '_ov_zero_head_indices')
            cleared += 1
    return cleared
