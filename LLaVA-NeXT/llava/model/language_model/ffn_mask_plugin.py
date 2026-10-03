from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
_MLP_PATTERN = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.mlp$')

def ffn_mask_decode_enabled() -> bool:
    raw = os.environ.get('FFN_MASK_DECODE', os.environ.get('REGION_MASK_DECODE', '0')).strip().lower()
    if raw in {'0', 'false', 'no', 'off'}:
        return False
    if raw in {'1', 'true', 'yes', 'on'}:
        return True
    raise ValueError(f'FFN_MASK_DECODE must be one of 0/1/false/true/no/yes/off/on, got {raw!r}')

def ffn_compute_backend() -> str:
    raw = os.environ.get('FFN_MASK_BACKEND', os.environ.get('FFN_COMPUTE_BACKEND', 'dense')).strip().lower()
    aliases = {'mask': 'dense', 'zero': 'dense', 'pruned': 'structured'}
    raw = aliases.get(raw, raw)
    if raw not in {'dense', 'structured'}:
        raise ValueError(f'FFN_MASK_BACKEND must be dense|structured, got {raw!r}')
    return raw

def ffn_structured_alignment() -> int:
    raw = os.environ.get('FFN_STRUCTURED_ALIGNMENT', '128').strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'FFN_STRUCTURED_ALIGNMENT must be a positive integer, got {raw!r}') from exc
    if value <= 0:
        raise ValueError(f'FFN_STRUCTURED_ALIGNMENT must be a positive integer, got {value}')
    return value

@dataclass
class FFNMaskSummary:
    neuron_count: int = 0
    layer_count: int = 0
    applied_module_count: int = 0
    token_scope: str = 'all'

    def describe(self) -> str:
        return f'ffn_mask: {self.neuron_count} neurons in {self.layer_count} layers (token_scope={self.token_scope})'

@dataclass
class FFNComputeSummary:
    backend: str = 'dense'
    layer_count: int = 0
    vision_kept: int = 0
    text_kept: int = 0
    intermediate_total: int = 0

    def describe(self) -> str:
        if self.backend == 'dense':
            return 'ffn_compute: dense'
        return f'ffn_compute: {self.backend} in {self.layer_count} layers (vision_keep={self.vision_kept}/{self.intermediate_total}, text_keep={self.text_kept}/{self.intermediate_total})'

def _merge_ranges(ranges: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    merged: List[Tuple[int, int]] = []
    for start, end in ranges:
        start, end = (int(start), int(end))
        if end <= start:
            continue
        if merged and merged[-1][1] == start:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return tuple(merged)

def _permute_parameter_(parameter: Optional[torch.nn.Parameter], dim: int, perm: torch.Tensor) -> None:
    if parameter is None:
        return
    if parameter.device.type == 'meta':
        raise RuntimeError('structured FFN backend cannot permute a meta parameter')
    index = perm.to(device=parameter.device)
    with torch.no_grad():
        parameter.data = parameter.data.index_select(dim, index).contiguous()

def _restore_ffn_compute_plan(module: torch.nn.Module) -> bool:
    plan = getattr(module, '_ov_ffn_compute_plan', None)
    if not isinstance(plan, dict):
        return False
    inverse = plan['inverse_permutation']
    _permute_parameter_(module.gate_proj.weight, 0, inverse)
    _permute_parameter_(module.up_proj.weight, 0, inverse)
    _permute_parameter_(module.down_proj.weight, 1, inverse)
    _permute_parameter_(getattr(module.gate_proj, 'bias', None), 0, inverse)
    _permute_parameter_(getattr(module.up_proj, 'bias', None), 0, inverse)
    masks = getattr(module, '_ov_ffn_keep_masks', None)
    if isinstance(masks, dict):
        module._ov_ffn_keep_masks = {scope: keep.index_select(0, inverse.to(device=keep.device)) for scope, keep in masks.items()}
    legacy = getattr(module, '_ov_ffn_keep_mask', None)
    if torch.is_tensor(legacy):
        module._ov_ffn_keep_mask = legacy.index_select(0, inverse.to(device=legacy.device))
    delattr(module, '_ov_ffn_compute_plan')
    return True

def restore_ffn_compute_pruning(model: torch.nn.Module) -> int:
    restored = 0
    for _, module in model.named_modules():
        restored += int(_restore_ffn_compute_plan(module))
    return restored

def prepare_ffn_compute_pruning(model: torch.nn.Module) -> FFNComputeSummary:
    backend = ffn_compute_backend()
    summary = FFNComputeSummary(backend=backend)
    if backend != 'structured':
        return summary
    for name, module in model.named_modules():
        match = _MLP_PATTERN.search(name)
        if match is None:
            continue
        masks = getattr(module, '_ov_ffn_keep_masks', None)
        if not isinstance(masks, dict) or not masks:
            continue
        if isinstance(getattr(module, '_ov_ffn_compute_plan', None), dict):
            plan = module._ov_ffn_compute_plan
            summary.layer_count += 1
            summary.vision_kept += int(plan['vision_kept'])
            summary.text_kept += int(plan['text_kept'])
            summary.intermediate_total += int(plan['intermediate_size'])
            continue
        inter = int(module.gate_proj.out_features)
        if int(module.up_proj.out_features) != inter or int(module.down_proj.in_features) != inter:
            raise RuntimeError(f'inconsistent Qwen2 MLP dimensions at {name}')
        prepared: Dict[str, torch.Tensor] = {}
        for scope, keep in masks.items():
            if scope not in {'all', 'vision', 'text'}:
                raise ValueError(f'unknown FFN token scope {scope!r} at {name}')
            current = keep.detach().to(device='cpu').flatten().ne(0)
            if int(current.numel()) != inter:
                raise ValueError(f'FFN keep-mask width {current.numel()} at {name} does not match intermediate size {inter}')
            prepared[scope] = current
        ones = torch.ones(inter, dtype=torch.bool)
        all_keep = prepared.get('all', ones)
        vision_keep = all_keep & prepared.get('vision', ones)
        text_keep = all_keep & prepared.get('text', ones)
        both = torch.nonzero(vision_keep & text_keep, as_tuple=False).flatten()
        vision_only = torch.nonzero(vision_keep & ~text_keep, as_tuple=False).flatten()
        text_only = torch.nonzero(~vision_keep & text_keep, as_tuple=False).flatten()
        neither = torch.nonzero(~vision_keep & ~text_keep, as_tuple=False).flatten()
        permutation = torch.cat((both, vision_only, text_only, neither)).to(torch.long)
        if int(permutation.numel()) != inter:
            raise RuntimeError(f'invalid structured FFN permutation at {name}')
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(inter, dtype=torch.long)
        _permute_parameter_(module.gate_proj.weight, 0, permutation)
        _permute_parameter_(module.up_proj.weight, 0, permutation)
        _permute_parameter_(module.down_proj.weight, 1, permutation)
        _permute_parameter_(getattr(module.gate_proj, 'bias', None), 0, permutation)
        _permute_parameter_(getattr(module.up_proj, 'bias', None), 0, permutation)
        module._ov_ffn_keep_masks = {scope: keep.index_select(0, permutation.to(device=keep.device)) for scope, keep in masks.items()}
        legacy = getattr(module, '_ov_ffn_keep_mask', None)
        if torch.is_tensor(legacy):
            module._ov_ffn_keep_mask = legacy.index_select(0, permutation.to(device=legacy.device))
        n_both = int(both.numel())
        n_vision_only = int(vision_only.numel())
        n_text_only = int(text_only.numel())
        vision_end = n_both + n_vision_only
        text_only_start = vision_end
        text_only_end = text_only_start + n_text_only
        permuted_vision_keep = vision_keep.index_select(0, permutation)
        permuted_text_keep = text_keep.index_select(0, permutation)
        alignment = ffn_structured_alignment()
        vision_compute_end = min(inter, (vision_end + alignment - 1) // alignment * alignment)
        with torch.no_grad():
            vision_down_weight = module.down_proj.weight[:, :vision_compute_end].detach().clone(memory_format=torch.contiguous_format)
            if vision_compute_end > vision_end:
                vision_down_weight[:, vision_end:] = 0
        module._ov_ffn_compute_plan = {'intermediate_size': inter, 'permutation': permutation, 'inverse_permutation': inverse, 'vision_ranges': _merge_ranges(((0, vision_end),)), 'text_ranges': _merge_ranges(((0, n_both), (text_only_start, text_only_end))), 'full_ranges': ((0, inter),), 'vision_kept': int(vision_keep.sum().item()), 'text_kept': int(text_keep.sum().item()), 'vision_end': vision_end, 'vision_compute_end': vision_compute_end, 'alignment': alignment, 'vision_down_weight': vision_down_weight, 'vision_keep_mask': permuted_vision_keep.to(device=module.down_proj.weight.device), 'text_keep_mask': permuted_text_keep.to(device=module.down_proj.weight.device), 'text_all_kept': bool(text_keep.all().item())}
        summary.layer_count += 1
        summary.vision_kept += int(vision_keep.sum().item())
        summary.text_kept += int(text_keep.sum().item())
        summary.intermediate_total += inter
    return summary

def _project_ffn_ranges(module: torch.nn.Module, x: torch.Tensor, ranges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    inter = int(module.gate_proj.out_features)
    ranges = _merge_ranges(ranges)
    if len(ranges) == 1 and ranges[0] == (0, inter):
        h = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
        return module.down_proj(h)
    out = None
    for start, end in ranges:
        gate_bias = module.gate_proj.bias
        up_bias = module.up_proj.bias
        gate = F.linear(x, module.gate_proj.weight[start:end, :], None if gate_bias is None else gate_bias[start:end])
        up = F.linear(x, module.up_proj.weight[start:end, :], None if up_bias is None else up_bias[start:end])
        hidden = module.act_fn(gate) * up
        contribution = F.linear(hidden, module.down_proj.weight[:, start:end], None)
        out = contribution if out is None else out + contribution
    if out is None:
        out = x.new_zeros((*x.shape[:-1], int(module.down_proj.out_features)))
    if module.down_proj.bias is not None:
        out = out + module.down_proj.bias
    return out

def _project_ffn_vision_prefix(module: torch.nn.Module, x: torch.Tensor, plan: Mapping[str, object]) -> torch.Tensor:
    end = int(plan['vision_end'])
    compute_end = int(plan['vision_compute_end'])
    inter = int(plan['intermediate_size'])
    if end == inter:
        hidden = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
        return module.down_proj(hidden)
    if end == 0:
        out = x.new_zeros((*x.shape[:-1], int(module.down_proj.out_features)))
        if module.down_proj.bias is not None:
            out = out + module.down_proj.bias
        return out
    gate_bias = module.gate_proj.bias
    up_bias = module.up_proj.bias
    gate = F.linear(x, module.gate_proj.weight[:compute_end, :], None if gate_bias is None else gate_bias[:compute_end])
    up = F.linear(x, module.up_proj.weight[:compute_end, :], None if up_bias is None else up_bias[:compute_end])
    hidden = module.act_fn(gate) * up
    return F.linear(hidden, plan['vision_down_weight'], module.down_proj.bias)

def _project_ffn_text_dense_masked(module: torch.nn.Module, x: torch.Tensor, plan: Mapping[str, object]) -> torch.Tensor:
    hidden = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
    keep = plan['text_keep_mask']
    if torch.is_tensor(keep) and (not bool(plan['text_all_kept'])):
        hidden = hidden * keep.to(device=hidden.device, dtype=hidden.dtype)
    return module.down_proj(hidden)

def run_ffn_compute_pruned(module: torch.nn.Module, x: torch.Tensor) -> Optional[torch.Tensor]:
    plan = getattr(module, '_ov_ffn_compute_plan', None)
    if not isinstance(plan, dict):
        return None
    if module.training:
        raise RuntimeError('FFN_MASK_BACKEND=structured is inference-only')
    if x.ndim != 3:
        raise ValueError(f'structured FFN expects [batch, tokens, hidden], got {tuple(x.shape)}')
    runtime = getattr(module, '_ov_ffn_region_runtime', None)
    q_len = int(x.shape[1])
    if isinstance(runtime, dict) and runtime.get('has_image'):
        required = ('vis_s', 'vis_e', 'prefill_kv_seq_len')
        if any((key not in runtime for key in required)):
            raise RuntimeError(f'incomplete visual span for structured FFN: missing {[key for key in required if key not in runtime]}')
        vis_s = int(runtime['vis_s'])
        vis_e = int(runtime['vis_e'])
        prefill_len = int(runtime['prefill_kv_seq_len'])
        if q_len != prefill_len:
            if ffn_mask_decode_enabled():
                return _project_ffn_text_dense_masked(module, x, plan)
            hidden = module.act_fn(module.gate_proj(x)) * module.up_proj(x)
            return module.down_proj(hidden)
        if not 0 <= vis_s < vis_e <= q_len:
            raise RuntimeError(f'invalid visual span [{vis_s}, {vis_e}) for FFN prefill length {q_len}')
        out = x.new_empty((*x.shape[:-1], int(module.down_proj.out_features)))
        prefix_len = vis_s
        suffix_len = q_len - vis_e
        text_len = prefix_len + suffix_len
        if text_len:
            if prefix_len and suffix_len:
                text_x = torch.cat((x[:, :vis_s, :], x[:, vis_e:, :]), dim=1)
            elif prefix_len:
                text_x = x[:, :vis_s, :]
            else:
                text_x = x[:, vis_e:, :]
            text_out = _project_ffn_text_dense_masked(module, text_x, plan)
            if prefix_len:
                out[:, :vis_s, :] = text_out[:, :prefix_len, :]
            if suffix_len:
                out[:, vis_e:, :] = text_out[:, prefix_len:, :]
        out[:, vis_s:vis_e, :] = _project_ffn_vision_prefix(module, x[:, vis_s:vis_e, :], plan)
        return out
    return _project_ffn_text_dense_masked(module, x, plan)

def apply_ffn_token_mask(h: torch.Tensor, keep: torch.Tensor, token_scope: str='all', region_runtime: Optional[dict]=None) -> torch.Tensor:
    if token_scope not in ('all', 'vision', 'text'):
        raise ValueError(f'token_scope must be all|vision|text, got {token_scope!r}')
    if h.ndim != 3:
        raise ValueError(f'FFN token mask expects [batch, tokens, intermediate], got {tuple(h.shape)}')
    keep = keep.to(dtype=h.dtype, device=h.device).flatten()
    if int(keep.numel()) != int(h.shape[-1]):
        raise ValueError(f'FFN keep-mask width {keep.numel()} does not match activation width {h.shape[-1]}')
    if token_scope == 'all':
        if isinstance(region_runtime, dict) and region_runtime.get('has_image') and ('prefill_kv_seq_len' in region_runtime) and (int(h.shape[1]) != int(region_runtime['prefill_kv_seq_len'])) and (not ffn_mask_decode_enabled()):
            return h
        return h * keep
    if not isinstance(region_runtime, dict):
        raise RuntimeError(f'token_scope={token_scope} requires the attention region runtime; put mask_v2v/mask_t2v/mask_t2t components before ffn_prune')
    if not region_runtime.get('has_image'):
        return h if token_scope == 'vision' else h * keep
    required = ('vis_s', 'vis_e', 'prefill_kv_seq_len')
    if any((key not in region_runtime for key in required)):
        raise RuntimeError(f'incomplete visual span for token_scope={token_scope}: missing {[key for key in required if key not in region_runtime]}')
    vis_s = int(region_runtime['vis_s'])
    vis_e = int(region_runtime['vis_e'])
    prefill_len = int(region_runtime['prefill_kv_seq_len'])
    q_len = int(h.shape[1])
    if q_len != prefill_len:
        if not ffn_mask_decode_enabled():
            return h
        return h if token_scope == 'vision' else h * keep
    if not 0 <= vis_s < vis_e <= q_len:
        raise RuntimeError(f'invalid visual span [{vis_s}, {vis_e}) for FFN prefill length {q_len}')
    out = h.clone()
    if token_scope == 'vision':
        out[:, vis_s:vis_e, :] = out[:, vis_s:vis_e, :] * keep
    else:
        if vis_s:
            out[:, :vis_s, :] = out[:, :vis_s, :] * keep
        if vis_e < q_len:
            out[:, vis_e:, :] = out[:, vis_e:, :] * keep
    return out

def apply_ffn_scope_masks(h: torch.Tensor, keep_masks: Mapping[str, torch.Tensor], region_runtime: Optional[dict]=None) -> torch.Tensor:
    if h.ndim != 3:
        raise ValueError(f'FFN scope masks expect [batch, tokens, intermediate], got {tuple(h.shape)}')
    unknown = set(keep_masks) - {'all', 'vision', 'text'}
    if unknown:
        raise ValueError(f'unknown FFN token scopes: {sorted(unknown)}')
    if not keep_masks:
        return h
    prepared: Dict[str, torch.Tensor] = {}
    for scope, keep in keep_masks.items():
        current = keep.to(dtype=h.dtype, device=h.device).flatten()
        if int(current.numel()) != int(h.shape[-1]):
            raise ValueError(f'FFN keep-mask width {current.numel()} for scope={scope} does not match activation width {h.shape[-1]}')
        prepared[scope] = current
    is_cached_decode = isinstance(region_runtime, dict) and region_runtime.get('has_image') and ('prefill_kv_seq_len' in region_runtime) and (int(h.shape[1]) != int(region_runtime['prefill_kv_seq_len']))
    if is_cached_decode and (not ffn_mask_decode_enabled()):
        return h
    out = h
    if 'all' in prepared:
        out = out * prepared['all']
    scoped = {scope for scope in ('vision', 'text') if scope in prepared}
    if not scoped:
        return out
    if not isinstance(region_runtime, dict):
        raise RuntimeError('token_scope=vision/text requires the attention region runtime; put mask_v2v/mask_t2v/mask_t2t components before ffn_prune')
    if not region_runtime.get('has_image'):
        return out if 'text' not in prepared else out * prepared['text']
    required = ('vis_s', 'vis_e', 'prefill_kv_seq_len')
    if any((key not in region_runtime for key in required)):
        raise RuntimeError(f'incomplete visual span for token_scope=vision/text: missing {[key for key in required if key not in region_runtime]}')
    vis_s = int(region_runtime['vis_s'])
    vis_e = int(region_runtime['vis_e'])
    prefill_len = int(region_runtime['prefill_kv_seq_len'])
    q_len = int(h.shape[1])
    if q_len != prefill_len:
        if not ffn_mask_decode_enabled():
            return out
        return out if 'text' not in prepared else out * prepared['text']
    if not 0 <= vis_s < vis_e <= q_len:
        raise RuntimeError(f'invalid visual span [{vis_s}, {vis_e}) for FFN prefill length {q_len}')
    out = out.clone()
    if 'vision' in prepared:
        out[:, vis_s:vis_e, :] = out[:, vis_s:vis_e, :] * prepared['vision']
    if 'text' in prepared:
        if vis_s:
            out[:, :vis_s, :] = out[:, :vis_s, :] * prepared['text']
        if vis_e < q_len:
            out[:, vis_e:, :] = out[:, vis_e:, :] * prepared['text']
    return out

def apply_ffn_neuron_mask(model: torch.nn.Module, neurons_by_layer: Dict[int, List[int]], token_scope: str='all') -> FFNMaskSummary:
    if token_scope not in ('all', 'vision', 'text'):
        raise ValueError(f'token_scope must be all|vision|text, got {token_scope!r}')
    restore_ffn_compute_pruning(model)
    from .attention_mask_plugin import _resolve_shared_runtime
    region_runtime = _resolve_shared_runtime(model)
    if token_scope != 'all':
        if region_runtime is None:
            from .attention_mask_plugin import apply_region_mask_config
            apply_region_mask_config(model, 'v2v', [])
            region_runtime = _resolve_shared_runtime(model)
            if region_runtime is None:
                raise RuntimeError('failed to initialize FFN-only visual-token runtime')
    n_neuron = 0
    n_layer = 0
    for name, module in model.named_modules():
        m = _MLP_PATTERN.search(name)
        if m is None:
            continue
        layer_idx = int(m.group(1))
        dead = neurons_by_layer.get(layer_idx, neurons_by_layer.get(str(layer_idx)))
        if not dead:
            continue
        inter = int(module.gate_proj.out_features)
        keep = torch.ones(inter, dtype=torch.float32)
        idx = torch.tensor([int(n) for n in dead if 0 <= int(n) < inter], dtype=torch.long)
        if idx.numel():
            keep[idx] = 0.0
        scope_masks = getattr(module, '_ov_ffn_keep_masks', None)
        if not isinstance(scope_masks, dict):
            scope_masks = {}
        else:
            scope_masks = dict(scope_masks)
        scope_masks[token_scope] = keep
        setattr(module, '_ov_ffn_keep_masks', scope_masks)
        setattr(module, '_ov_ffn_keep_mask', keep)
        setattr(module, '_ov_ffn_token_scope', token_scope)
        if region_runtime is not None:
            setattr(module, '_ov_ffn_region_runtime', region_runtime)
        elif hasattr(module, '_ov_ffn_region_runtime'):
            delattr(module, '_ov_ffn_region_runtime')
        n_neuron += int((keep == 0).sum().item())
        n_layer += 1
    return FFNMaskSummary(neuron_count=n_neuron, layer_count=n_layer, applied_module_count=n_layer, token_scope=token_scope)

def clear_ffn_masks(model: torch.nn.Module) -> int:
    restore_ffn_compute_pruning(model)
    cleared = 0
    for _, module in model.named_modules():
        if hasattr(module, '_ov_ffn_keep_mask') or hasattr(module, '_ov_ffn_keep_masks'):
            for attr in ('_ov_ffn_keep_mask', '_ov_ffn_keep_masks', '_ov_ffn_token_scope', '_ov_ffn_region_runtime'):
                if hasattr(module, attr):
                    delattr(module, attr)
            cleared += 1
    return cleared
