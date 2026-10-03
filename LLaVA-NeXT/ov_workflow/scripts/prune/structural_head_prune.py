from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
_LAYER_RE = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.self_attn$')

def load_dead_heads(config_path: str) -> Dict[int, List[int]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ablation_type = cfg.get('ablation_type', 'zero')
    if ablation_type != 'zero':
        raise ValueError(f"structural_head_prune expects ablation_type='zero', got {ablation_type!r}.")
    by_layer: Dict[int, set] = {}
    for pair in cfg.get('heads', []):
        layer_idx, head_idx = (int(pair[0]), int(pair[1]))
        by_layer.setdefault(layer_idx, set()).add(head_idx)
    return {li: sorted(hs) for li, hs in sorted(by_layer.items())}

def _new_linear_like(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> nn.Linear:
    out_features, in_features = weight.shape
    lin = nn.Linear(in_features, out_features, bias=bias is not None)
    lin = lin.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin

def _block_index(blocks: Sequence[int], block_size: int, device) -> torch.Tensor:
    if len(blocks) == 0:
        raise ValueError('empty block list')
    return torch.cat([torch.arange(b * block_size, (b + 1) * block_size, device=device) for b in blocks])

def prune_attention_module(self_attn: nn.Module, dead_heads: Sequence[int], phase: int=1) -> dict:
    if phase not in (1, 2, 3):
        raise ValueError(f'phase must be 1, 2 or 3, got {phase}')
    num_heads = int(self_attn.num_heads)
    head_dim = int(self_attn.head_dim)
    num_kv = int(self_attn.num_key_value_heads)
    groups = int(self_attn.num_key_value_groups)
    if groups * num_kv != num_heads:
        raise ValueError(f'unexpected GQA layout: num_heads={num_heads} num_kv={num_kv} groups={groups}')
    dead = sorted({int(h) for h in dead_heads})
    for h in dead:
        if h < 0 or h >= num_heads:
            raise ValueError(f'dead head {h} out of range [0,{num_heads}) ')
    keep = [h for h in range(num_heads) if h not in set(dead)]
    if len(keep) == 0:
        raise ValueError('refusing to prune ALL heads in a layer')
    if len(dead) == 0:
        return {'orig_heads': num_heads, 'kept': num_heads, 'removed': 0, 'skipped': True}
    device = self_attn.q_proj.weight.device
    q_rows = _block_index(keep, head_dim, device)
    new_q = _new_linear_like(self_attn.q_proj.weight.index_select(0, q_rows), None if self_attn.q_proj.bias is None else self_attn.q_proj.bias.index_select(0, q_rows))
    o_cols = _block_index(keep, head_dim, device)
    new_o = _new_linear_like(self_attn.o_proj.weight.index_select(1, o_cols), None if self_attn.o_proj.bias is None else self_attn.o_proj.bias)
    kept = len(keep)
    kv_blocks = [h // groups for h in keep]
    self_attn.q_proj = new_q
    self_attn.o_proj = new_o
    self_attn.num_heads = kept
    self_attn.hidden_size = kept * head_dim
    if phase == 1:
        kv_rows = _block_index(kv_blocks, head_dim, device)
        self_attn.k_proj = _new_linear_like(self_attn.k_proj.weight.index_select(0, kv_rows), None if self_attn.k_proj.bias is None else self_attn.k_proj.bias.index_select(0, kv_rows))
        self_attn.v_proj = _new_linear_like(self_attn.v_proj.weight.index_select(0, kv_rows), None if self_attn.v_proj.bias is None else self_attn.v_proj.bias.index_select(0, kv_rows))
        self_attn.num_key_value_heads = kept
        self_attn.num_key_value_groups = 1
        if hasattr(self_attn, '_ov_pruned_kv_index'):
            delattr(self_attn, '_ov_pruned_kv_index')
    else:
        if hasattr(self_attn, '_ov_pruned_kv_index'):
            delattr(self_attn, '_ov_pruned_kv_index')
        if phase == 2:
            self_attn.num_key_value_heads = num_kv
            gather = kv_blocks
        else:
            surviving_kv = sorted({h // groups for h in keep})
            kv_rows = _block_index(surviving_kv, head_dim, device)
            self_attn.k_proj = _new_linear_like(self_attn.k_proj.weight.index_select(0, kv_rows), None if self_attn.k_proj.bias is None else self_attn.k_proj.bias.index_select(0, kv_rows))
            self_attn.v_proj = _new_linear_like(self_attn.v_proj.weight.index_select(0, kv_rows), None if self_attn.v_proj.bias is None else self_attn.v_proj.bias.index_select(0, kv_rows))
            self_attn.num_key_value_heads = len(surviving_kv)
            remap = {g: i for i, g in enumerate(surviving_kv)}
            gather = [remap[h // groups] for h in keep]
        self_attn.num_key_value_groups = 1
        idx = torch.as_tensor(gather, dtype=torch.long, device=device)
        self_attn.register_buffer('_ov_pruned_kv_index', idx, persistent=False)
    return {'orig_heads': num_heads, 'kept': kept, 'removed': len(dead), 'removed_heads': dead, 'phase': phase}

def find_attention_modules(model: nn.Module) -> Dict[int, nn.Module]:
    found: Dict[int, nn.Module] = {}
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if not m:
            continue
        if getattr(module, 'num_heads', None) is None:
            continue
        layer_idx = int(getattr(module, 'layer_idx', m.group(1)))
        found[layer_idx] = module
    return found

def prune_heads_from_mapping(model: nn.Module, dead_by_layer: Dict[int, Sequence[int]], phase: int=1) -> dict:
    found = find_attention_modules(model)
    report: Dict[int, dict] = {}
    for layer_idx, dead in sorted(dead_by_layer.items()):
        if layer_idx not in found:
            raise ValueError(f'self_attn for layer {layer_idx} not found in model')
        report[layer_idx] = prune_attention_module(found[layer_idx], dead, phase=phase)
    total_removed = sum((r['removed'] for r in report.values()))
    return {'layers_pruned': len(report), 'heads_removed': total_removed, 'phase': phase, 'per_layer': report}

def prune_heads_from_zero_config(model: nn.Module, config_path: str, phase: int=1) -> dict:
    return prune_heads_from_mapping(model, load_dead_heads(config_path), phase=phase)
