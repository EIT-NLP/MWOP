from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Sequence
import torch
import torch.nn as nn
_LAYER_RE = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.mlp$')

def load_dead_neurons(config_path: str) -> Dict[int, List[int]]:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ablation_type = cfg.get('ablation_type', 'ffn_prune')
    if ablation_type != 'ffn_prune':
        raise ValueError(f"structural_ffn_prune expects ablation_type='ffn_prune', got {ablation_type!r}.")
    neurons = cfg.get('neurons', {})
    by_layer: Dict[int, set] = {}
    if isinstance(neurons, dict):
        for layer, lst in neurons.items():
            by_layer.setdefault(int(layer), set()).update((int(n) for n in lst))
    else:
        for pair in neurons:
            layer_idx, neuron_idx = (int(pair[0]), int(pair[1]))
            by_layer.setdefault(layer_idx, set()).add(neuron_idx)
    return {li: sorted(ns) for li, ns in sorted(by_layer.items())}

def _new_linear_like(weight: torch.Tensor, bias: Optional[torch.Tensor]) -> nn.Linear:
    out_features, in_features = weight.shape
    lin = nn.Linear(in_features, out_features, bias=bias is not None)
    lin = lin.to(device=weight.device, dtype=weight.dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if bias is not None:
            lin.bias.copy_(bias)
    return lin

def prune_mlp_module(mlp: nn.Module, dead_neurons: Sequence[int]) -> dict:
    inter = int(mlp.gate_proj.weight.shape[0])
    dead = sorted({int(n) for n in dead_neurons})
    for n in dead:
        if n < 0 or n >= inter:
            raise ValueError(f'dead neuron {n} out of range [0,{inter})')
    if len(dead) == 0:
        return {'orig_inter': inter, 'kept': inter, 'removed': 0, 'skipped': True}
    dead_set = set(dead)
    keep = [n for n in range(inter) if n not in dead_set]
    if len(keep) == 0:
        raise ValueError('refusing to prune ALL neurons in an MLP layer')
    device = mlp.gate_proj.weight.device
    keep_idx = torch.as_tensor(keep, dtype=torch.long, device=device)
    mlp.gate_proj = _new_linear_like(mlp.gate_proj.weight.index_select(0, keep_idx), None if mlp.gate_proj.bias is None else mlp.gate_proj.bias.index_select(0, keep_idx))
    mlp.up_proj = _new_linear_like(mlp.up_proj.weight.index_select(0, keep_idx), None if mlp.up_proj.bias is None else mlp.up_proj.bias.index_select(0, keep_idx))
    mlp.down_proj = _new_linear_like(mlp.down_proj.weight.index_select(1, keep_idx), None if mlp.down_proj.bias is None else mlp.down_proj.bias)
    kept = len(keep)
    if hasattr(mlp, 'intermediate_size'):
        mlp.intermediate_size = kept
    return {'orig_inter': inter, 'kept': kept, 'removed': len(dead)}

def find_mlp_modules(model: nn.Module) -> Dict[int, nn.Module]:
    found: Dict[int, nn.Module] = {}
    for name, module in model.named_modules():
        m = _LAYER_RE.search(name)
        if not m:
            continue
        if getattr(module, 'gate_proj', None) is None or getattr(module, 'down_proj', None) is None:
            continue
        layer_idx = int(getattr(module, 'layer_idx', m.group(1)))
        found[layer_idx] = module
    return found

def prune_ffn_from_mapping(model: nn.Module, dead_by_layer: Dict[int, Sequence[int]]) -> dict:
    found = find_mlp_modules(model)
    report: Dict[int, dict] = {}
    for layer_idx, dead in sorted(dead_by_layer.items()):
        if layer_idx not in found:
            raise ValueError(f'mlp for layer {layer_idx} not found in model')
        report[layer_idx] = prune_mlp_module(found[layer_idx], dead)
    total_removed = sum((r['removed'] for r in report.values()))
    return {'layers_pruned': len(report), 'neurons_removed': total_removed, 'per_layer': report}

def prune_ffn_from_config(model: nn.Module, config_path: str) -> dict:
    return prune_ffn_from_mapping(model, load_dead_neurons(config_path))
