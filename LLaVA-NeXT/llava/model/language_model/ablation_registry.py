import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import torch
from .head_mask_utils import apply_zero_heads, HeadMaskSummary
from .attention_mask_plugin import apply_region_mask_config, RegionMaskSummary

def _resolve_heads(component: Dict[str, Any], base_dir: str) -> List[List[int]]:
    if 'heads_from' in component:
        ref = component['heads_from']
        if not isinstance(ref, str):
            raise ValueError(f'heads_from must be a path string, got {type(ref).__name__}')
        full = ref if os.path.isabs(ref) else os.path.join(base_dir, ref)
        if not os.path.exists(full):
            raise FileNotFoundError(f'heads_from file does not exist: {full}')
        with open(full, 'r', encoding='utf-8') as f:
            ref_cfg = json.load(f)
        ref_heads = ref_cfg.get('heads', [])
        if not isinstance(ref_heads, list):
            raise ValueError(f"heads_from file {full!r} has no 'heads' list")
        return list(ref_heads) + list(component.get('heads', []))
    return list(component.get('heads', []))

def _resolve_neurons(component: Dict[str, Any], base_dir: str) -> Dict[int, List[int]]:
    raw = component.get('neurons')
    if 'neurons_from' in component:
        ref = component['neurons_from']
        if not isinstance(ref, str):
            raise ValueError(f'neurons_from must be a path string, got {type(ref).__name__}')
        full = ref if os.path.isabs(ref) else os.path.join(base_dir, ref)
        if not os.path.exists(full):
            raise FileNotFoundError(f'neurons_from file does not exist: {full}')
        with open(full, 'r', encoding='utf-8') as f:
            raw = json.load(f).get('neurons')
    out: Dict[int, List[int]] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            out[int(k)] = [int(n) for n in v]
    elif isinstance(raw, list):
        for pair in raw:
            layer_idx, neuron = (int(pair[0]), int(pair[1]))
            out.setdefault(layer_idx, []).append(neuron)
    return out

@dataclass
class AblationSummary:
    zero: Optional[HeadMaskSummary] = None
    v2v: Optional[RegionMaskSummary] = None
    t2v: Optional[RegionMaskSummary] = None
    t2t: Optional[RegionMaskSummary] = None
    ffn_neurons: int = 0
    ffn_layers: int = 0
    ffn_token_scopes: List[str] = field(default_factory=list)
    ffn_scope_stats: List[Tuple[str, int, int]] = field(default_factory=list)
    ffn_compute_backend: str = 'dense'
    ffn_compute_layers: int = 0
    ffn_compute_vision_kept: int = 0
    ffn_compute_text_kept: int = 0
    ffn_compute_intermediate_total: int = 0
    components: List[str] = field(default_factory=list)

    def describe(self) -> str:
        bits = []
        if self.zero is not None:
            bits.append(f'zero: {self.zero.head_count} heads in {self.zero.layer_count} layers')
        if self.v2v is not None:
            bits.append(f'mask_v2v: {self.v2v.head_count} heads in {self.v2v.layer_count} layers')
        if self.t2v is not None:
            bits.append(f'mask_t2v: {self.t2v.head_count} heads in {self.t2v.layer_count} layers')
        if self.t2t is not None:
            bits.append(f'mask_t2t: {self.t2t.head_count} heads in {self.t2t.layer_count} layers')
        if self.ffn_neurons:
            if len(self.ffn_scope_stats) <= 1:
                bits.append(f"ffn_prune: {self.ffn_neurons} neurons in {self.ffn_layers} layers (token_scope={'+'.join(self.ffn_token_scopes) or 'all'})")
            else:
                detail = ' + '.join((f'{scope}={neurons} neurons in {layers} layers' for scope, neurons, layers in self.ffn_scope_stats))
                bits.append(f'ffn_prune: {detail}')
        if self.ffn_compute_backend != 'dense':
            bits.append(f'ffn_compute: {self.ffn_compute_backend} in {self.ffn_compute_layers} layers (vision_keep={self.ffn_compute_vision_kept}/{self.ffn_compute_intermediate_total}, text_keep={self.ffn_compute_text_kept}/{self.ffn_compute_intermediate_total})')
        if not bits:
            return 'no-op (empty config)'
        return ' | '.join(bits)

def _apply_one(model: torch.nn.Module, component: Dict[str, Any], summary: AblationSummary, base_dir: str) -> None:
    ablation_type = str(component.get('ablation_type', 'zero'))
    if ablation_type == 'ffn_prune':
        neurons = _resolve_neurons(component, base_dir)
        if not neurons:
            summary.components.append('ffn_prune:empty(skipped)')
            return
        from .ffn_mask_plugin import apply_ffn_neuron_mask
        token_scope = str(component.get('token_scope', 'all'))
        sub = apply_ffn_neuron_mask(model, neurons, token_scope=token_scope)
        summary.ffn_neurons += sub.neuron_count
        summary.ffn_layers += sub.layer_count
        summary.ffn_token_scopes.append(sub.token_scope)
        summary.ffn_scope_stats.append((sub.token_scope, sub.neuron_count, sub.layer_count))
        summary.components.append(f'ffn_prune:{sub.neuron_count}n/{sub.layer_count}l/{sub.token_scope}')
        return
    heads = _resolve_heads(component, base_dir)
    if not heads:
        summary.components.append(f'{ablation_type}:empty(skipped)')
        return
    if ablation_type == 'zero':
        sub = apply_zero_heads(model, heads)
        prev = summary.zero
        summary.zero = sub if prev is None else HeadMaskSummary(head_count=prev.head_count + sub.head_count, layer_count=prev.layer_count + sub.layer_count, applied_module_count=prev.applied_module_count + sub.applied_module_count)
        summary.components.append(f'zero:{sub.head_count}h/{sub.layer_count}l')
    elif ablation_type in ('mask_v2v', 'mask_t2v', 'mask_t2t'):
        kind = {'mask_v2v': 'v2v', 'mask_t2v': 't2v', 'mask_t2t': 't2t'}[ablation_type]
        sub = apply_region_mask_config(model, kind, heads)
        slot_map = {'v2v': 'v2v', 't2v': 't2v', 't2t': 't2t'}
        slot = slot_map[kind]
        prev = getattr(summary, slot)
        merged = sub if prev is None else RegionMaskSummary(kind=kind, head_count=prev.head_count + sub.head_count, layer_count=prev.layer_count + sub.layer_count, applied_module_count=prev.applied_module_count + sub.applied_module_count)
        setattr(summary, slot, merged)
        summary.components.append(f'{ablation_type}:{sub.head_count}h/{sub.layer_count}l')
    else:
        raise ValueError(f"Unknown ablation_type {ablation_type!r}. Supported: 'zero', 'mask_v2v', 'mask_t2v', 'mask_t2t', 'ffn_prune', 'composite'.")

def apply_ablation_config(model: torch.nn.Module, config_path: str) -> AblationSummary:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    ablation_type = str(cfg.get('ablation_type', 'zero'))
    summary = AblationSummary()
    base_dir = os.path.dirname(os.path.abspath(config_path))
    if ablation_type == 'composite':
        components = cfg.get('components', [])
        if not isinstance(components, list):
            raise ValueError("composite config must contain a 'components' list of {'ablation_type': ..., 'heads': [...] } entries")
        for comp in components:
            _apply_one(model, comp, summary, base_dir)
    else:
        _apply_one(model, cfg, summary, base_dir)
    if summary.ffn_neurons:
        from .ffn_mask_plugin import prepare_ffn_compute_pruning
        compute = prepare_ffn_compute_pruning(model)
        summary.ffn_compute_backend = compute.backend
        summary.ffn_compute_layers = compute.layer_count
        summary.ffn_compute_vision_kept = compute.vision_kept
        summary.ffn_compute_text_kept = compute.text_kept
        summary.ffn_compute_intermediate_total = compute.intermediate_total
        if compute.backend != 'dense':
            summary.components.append(f'ffn_compute:{compute.backend}/{compute.layer_count}l')
    return summary
