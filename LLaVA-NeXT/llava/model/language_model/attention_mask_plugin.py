import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple
import torch

def _read_decode_mask_mode() -> bool:
    raw = os.environ.get('REGION_MASK_DECODE', '0').strip().lower()
    if raw in ('1', 'true', 'yes', 'on'):
        return True
    if raw in ('0', 'false', 'no', 'off'):
        return False
    raise ValueError(f'REGION_MASK_DECODE must be one of 0/1/false/true/no/yes/off/on, got {raw!r}')
_REGION_MASK_DECODE_ENABLED = _read_decode_mask_mode()

def region_mask_decode_enabled() -> bool:
    return _REGION_MASK_DECODE_ENABLED

def _image_token_index() -> int:
    from llava.constants import IMAGE_TOKEN_INDEX
    return IMAGE_TOKEN_INDEX

@dataclass(frozen=True)
class RegionMaskSummary:
    kind: str
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
_LAYER_PATTERN = re.compile('(?:^|\\.)model\\.layers\\.(\\d+)\\.self_attn$')

def _ensure_runtime(model: torch.nn.Module) -> dict:
    rt = getattr(model, '_ov_region_runtime', None)
    if rt is None:
        rt = {}
        try:
            model._ov_region_runtime = rt
        except Exception:
            pass
    return rt

def _resolve_shared_runtime(model: torch.nn.Module):
    cached = getattr(model, '_ov_resolved_region_runtime', None)
    if cached is not None:
        return cached
    for _, mod in model.named_modules():
        if getattr(mod, 'num_heads', None) is None:
            continue
        lrt = getattr(mod, '_ov_region_runtime', None)
        if lrt is not None:
            try:
                model._ov_resolved_region_runtime = lrt
            except Exception:
                pass
            return lrt
    for _, mod in model.named_modules():
        lrt = getattr(mod, '_ov_region_runtime', None)
        if lrt is not None:
            try:
                model._ov_resolved_region_runtime = lrt
            except Exception:
                pass
            return lrt
    return getattr(model, '_ov_region_runtime', None)

def apply_region_mask_config(model: torch.nn.Module, kind: str, heads: Iterable[Sequence[int]]) -> RegionMaskSummary:
    if kind not in ('v2v', 't2v', 't2t'):
        raise ValueError(f"kind must be 'v2v', 't2v' or 't2t', got {kind!r}")
    attr_name = f'_ov_{kind}_head_indices'
    heads_by_layer = _normalize_heads(heads)
    rt = _ensure_runtime(model)
    applied_module_count = 0
    for name, module in model.named_modules():
        match = _LAYER_PATTERN.search(name)
        if not match:
            continue
        layer_idx = int(getattr(module, 'layer_idx', match.group(1)))
        if layer_idx not in heads_by_layer:
            continue
        num_heads = getattr(module, 'num_heads', None)
        if num_heads is not None and any((h < 0 or h >= int(num_heads) for h in heads_by_layer[layer_idx])):
            raise ValueError(f'Head indices {heads_by_layer[layer_idx]} out of range for layer {layer_idx} with {num_heads} heads')
        existing = tuple(getattr(module, attr_name, ()) or ())
        merged = tuple(sorted(set(existing) | set(heads_by_layer[layer_idx])))
        setattr(module, attr_name, merged)
        setattr(module, '_ov_region_runtime', rt)
        applied_module_count += 1
    if applied_module_count != len(heads_by_layer):
        missing = sorted((layer_idx for layer_idx in heads_by_layer if not any((int(getattr(m, 'layer_idx', -1)) == layer_idx and heads_by_layer[layer_idx][0] in (getattr(m, attr_name, ()) or ()) for m in model.modules()))))
        raise ValueError(f'Failed to apply {kind} mask to layers: {missing}')
    return RegionMaskSummary(kind=kind, head_count=sum((len(hs) for hs in heads_by_layer.values())), layer_count=len(heads_by_layer), applied_module_count=applied_module_count)

def update_region_runtime(model: torch.nn.Module, input_ids: Optional[torch.Tensor], *, past_key_values=None) -> None:
    rt = _resolve_shared_runtime(model)
    try:
        model._ov_region_runtime_active = rt is not None
    except Exception:
        pass
    if rt is None:
        return
    if input_ids is None:
        return
    if not isinstance(input_ids, torch.Tensor) or input_ids.numel() == 0:
        rt.clear()
        rt['has_image'] = False
        return
    img_idx = _image_token_index()
    if input_ids.dim() not in (1, 2):
        raise ValueError(f'Region attention masks require 1-D or 2-D input_ids; got shape {tuple(input_ids.shape)}')
    all_image_tokens = int((input_ids == img_idx).sum().item())
    if _past_key_values_length(past_key_values) > 0 and all_image_tokens == 0 and rt.get('has_image'):
        return
    if input_ids.dim() == 1:
        row = input_ids
    else:
        if input_ids.shape[0] != 1:
            raise ValueError(f'Region attention masks currently require batch_size == 1; got input_ids shape {tuple(input_ids.shape)}')
        row = input_ids[0]
    mask = row == img_idx
    n_img_ph = int(mask.sum().item())
    img_ph_pos = int(mask.nonzero(as_tuple=True)[0][0].item()) if n_img_ph >= 1 else 0
    rt.clear()
    rt.update({'has_image': n_img_ph > 0, 'orig_seq_len': int(input_ids.shape[-1]), 'n_img_placeholders': n_img_ph, 'img_placeholder_pos': img_ph_pos})

def _past_key_values_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    get_seq_length = getattr(past_key_values, 'get_seq_length', None)
    if callable(get_seq_length):
        try:
            return int(get_seq_length())
        except (TypeError, ValueError):
            return 0
    try:
        first_layer = past_key_values[0]
        if first_layer is None:
            return 0
        key_states = first_layer[0]
        return int(key_states.shape[-2])
    except (IndexError, KeyError, TypeError, AttributeError):
        return 0

def finalize_region_runtime(model: torch.nn.Module, expanded_seq_len: int) -> None:
    if not getattr(model, '_ov_region_runtime_active', False):
        return
    rt = _resolve_shared_runtime(model)
    if rt is None or not rt.get('has_image'):
        return
    if 'vis_s' in rt and 'vis_e' in rt:
        return
    if int(rt.get('n_img_placeholders', 0)) != 1:
        raise ValueError('Cannot finalize a region mask without exactly one image placeholder')
    n_img = int(expanded_seq_len) - int(rt['orig_seq_len']) + int(rt['n_img_placeholders'])
    vis_s = int(rt['img_placeholder_pos'])
    vis_e = vis_s + n_img
    if not 0 <= vis_s < vis_e <= int(expanded_seq_len):
        raise ValueError(f'Invalid visual token range after multimodal expansion: range=[{vis_s}, {vis_e}), expanded_seq_len={expanded_seq_len}')
    rt['vis_s'] = vis_s
    rt['vis_e'] = vis_e
    rt['prefill_kv_seq_len'] = int(expanded_seq_len)
    rt['span_source'] = 'length_inference'

def record_region_visual_span(model: torch.nn.Module, vis_s: int, vis_e: int, expanded_seq_len: int) -> None:
    if not getattr(model, '_ov_region_runtime_active', False):
        return
    rt = _resolve_shared_runtime(model)
    if rt is None or not rt.get('has_image'):
        return
    vis_s = int(vis_s)
    vis_e = int(vis_e)
    expanded_seq_len = int(expanded_seq_len)
    if not 0 <= vis_s < vis_e <= expanded_seq_len:
        raise ValueError(f'The image-token span was removed or invalid after sequence truncation: range=[{vis_s}, {vis_e}), expanded_seq_len={expanded_seq_len}')
    rt['vis_s'] = vis_s
    rt['vis_e'] = vis_e
    rt['prefill_kv_seq_len'] = expanded_seq_len
    rt['span_source'] = 'multimodal_expansion'


def record_region_visual_spans(model, spans, expanded_seq_len):
    """Record one contiguous visual region, rejecting interleaved text under modality masks.

    Adjacent image spans can share the existing contiguous attention/FFN backends.
    Disjoint spans require a per-token modality backend and must not be merged over text.
    Dense models have no active region runtime and accept their usual multi-image input.
    """
    if not getattr(model, '_ov_region_runtime_active', False):
        return
    runtime = _resolve_shared_runtime(model)
    if runtime is None or not runtime.get('has_image'):
        return
    length = int(expanded_seq_len)
    clipped = []
    for start, end in spans:
        start, end = int(start), int(end)
        if not 0 <= start < end:
            raise ValueError(f'Invalid visual span [{start}, {end})')
        if start >= length:
            continue
        end = min(end, length)
        if clipped and start < clipped[-1][1]:
            raise ValueError('Visual spans must be ordered and non-overlapping')
        if clipped and start == clipped[-1][1]:
            clipped[-1] = (clipped[-1][0], end)
        else:
            clipped.append((start, end))
    if len(clipped) > 1:
        raise ValueError('OV modality masks require a contiguous visual region; image-text-image inputs are unsupported while attention/FFN modality masks are active. Use a single contiguous visual region or dense inference.')
    if not clipped:
        raise ValueError('Sequence truncation removed every visual token under active modality masks')
    record_region_visual_span(model, *clipped[0], length)


def reset_region_runtime(model: torch.nn.Module) -> None:
    if not getattr(model, '_ov_region_runtime_active', False):
        return
    rt = _resolve_shared_runtime(model)
    if rt is None:
        return
    rt.clear()
    rt['has_image'] = False

def _compute_visual_range(rt: dict, kv_seq_len: int, past_len: int=0) -> Optional[Tuple[int, int]]:
    if not rt or not rt.get('has_image'):
        return None
    if 'vis_s' in rt and 'vis_e' in rt:
        vis_s, vis_e = (int(rt['vis_s']), int(rt['vis_e']))
        if 0 <= vis_s < vis_e <= kv_seq_len:
            return (vis_s, vis_e)
        return None
    if past_len > 0:
        return None
    n_img = kv_seq_len - int(rt['orig_seq_len']) + int(rt['n_img_placeholders'])
    vis_s = int(rt['img_placeholder_pos'])
    vis_e = vis_s + max(int(n_img), 0)
    if vis_s >= vis_e or vis_e > kv_seq_len:
        return None
    rt['vis_s'] = vis_s
    rt['vis_e'] = vis_e
    rt['prefill_kv_seq_len'] = int(kv_seq_len)
    rt['span_source'] = 'length_inference'
    return (vis_s, vis_e)

def _local_query_region_bounds(q_len: int, past_len: int, vis_s: int, vis_e: int) -> Tuple[int, int, int]:
    q_vis_s = max(0, min(q_len, vis_s - past_len))
    q_vis_e = max(0, min(q_len, vis_e - past_len))
    q_text_s = max(0, min(q_len, vis_e - past_len))
    return (q_vis_s, q_vis_e, q_text_s)

def _validate_nonempty_region_rows(*, vis_s: int, has_v2v: bool, has_t2v: bool, has_t2t: bool, t2v_heads: Optional[Sequence[int]], t2t_heads: Optional[Sequence[int]]) -> None:
    if vis_s != 0:
        return
    if has_v2v:
        raise ValueError('V2V would mask every causal key for a visual query because the visual block starts at position 0')
    if has_t2v and has_t2t and set(t2v_heads or ()).intersection(t2t_heads or ()):
        raise ValueError('T2V and T2T would mask every causal key for a text query because the visual block starts at position 0')

def apply_region_mask_to_attn_weights(attn_weights: torch.Tensor, v2v_heads: Optional[Sequence[int]], t2t_heads: Optional[Sequence[int]], runtime: Optional[dict], past_len: int, t2v_heads: Optional[Sequence[int]]=None) -> torch.Tensor:
    if not v2v_heads and (not t2v_heads) and (not t2t_heads):
        return attn_weights
    if past_len > 0 and (not _REGION_MASK_DECODE_ENABLED):
        return attn_weights
    if attn_weights.dim() != 4:
        raise ValueError(f'attn_weights must be 4-D [B, H, q_len, kv_seq_len], got {tuple(attn_weights.shape)}')
    q_len = attn_weights.shape[-2]
    kv_seq_len = attn_weights.shape[-1]
    rng = _compute_visual_range(runtime or {}, kv_seq_len, past_len)
    if rng is None:
        return attn_weights
    vis_s, vis_e = rng
    q_vis_s, q_vis_e, q_text_s = _local_query_region_bounds(q_len, past_len, vis_s, vis_e)
    has_v2v = bool(v2v_heads) and q_vis_s < q_vis_e
    has_t2v = bool(t2v_heads) and q_text_s < q_len
    has_t2t = bool(t2t_heads) and q_text_s < q_len
    if not (has_v2v or has_t2v or has_t2t):
        return attn_weights
    _validate_nonempty_region_rows(vis_s=vis_s, has_v2v=has_v2v, has_t2v=has_t2v, has_t2t=has_t2t, t2v_heads=t2v_heads, t2t_heads=t2t_heads)
    neg_inf = torch.finfo(attn_weights.dtype).min
    if has_v2v:
        for h in v2v_heads:
            attn_weights[:, h, q_vis_s:q_vis_e, vis_s:vis_e] = neg_inf
    if has_t2v:
        for h in t2v_heads:
            attn_weights[:, h, q_text_s:, vis_s:vis_e] = neg_inf
    if has_t2t:
        for h in t2t_heads:
            attn_weights[:, h, q_text_s:, vis_e:] = neg_inf
    return attn_weights

def build_region_attention_mask(attention_mask: Optional[torch.Tensor], *, bsz: int, num_heads: int, q_len: int, kv_seq_len: int, past_len: int, dtype: torch.dtype, device: torch.device, v2v_heads: Optional[Sequence[int]], t2t_heads: Optional[Sequence[int]], runtime: Optional[dict], t2v_heads: Optional[Sequence[int]]=None) -> Optional[torch.Tensor]:
    if not v2v_heads and (not t2v_heads) and (not t2t_heads):
        return attention_mask
    if past_len > 0 and (not _REGION_MASK_DECODE_ENABLED):
        return attention_mask
    rng = _compute_visual_range(runtime or {}, kv_seq_len, past_len)
    if rng is None:
        return attention_mask
    vis_s, vis_e = rng
    q_vis_s, q_vis_e, q_text_s = _local_query_region_bounds(q_len, past_len, vis_s, vis_e)
    has_v2v = bool(v2v_heads) and q_vis_s < q_vis_e
    has_t2v = bool(t2v_heads) and q_text_s < q_len
    has_t2t = bool(t2t_heads) and q_text_s < q_len
    if not (has_v2v or has_t2v or has_t2t):
        return attention_mask
    _validate_nonempty_region_rows(vis_s=vis_s, has_v2v=has_v2v, has_t2v=has_t2v, has_t2t=has_t2t, t2v_heads=t2v_heads, t2t_heads=t2t_heads)
    neg_inf = torch.finfo(dtype).min
    if attention_mask is None or attention_mask.dim() != 4:
        new_mask = torch.zeros(bsz, num_heads, q_len, kv_seq_len, dtype=dtype, device=device)
        q_pos = torch.arange(q_len, device=device).unsqueeze(1) + past_len
        k_pos = torch.arange(kv_seq_len, device=device).unsqueeze(0)
        base = torch.zeros(q_len, kv_seq_len, dtype=dtype, device=device)
        base.masked_fill_(k_pos > q_pos, neg_inf)
        new_mask[:] = base
    else:
        new_mask = attention_mask.expand(bsz, num_heads, q_len, kv_seq_len).clone().to(dtype)
    if has_v2v:
        for h in v2v_heads:
            new_mask[:, h, q_vis_s:q_vis_e, vis_s:vis_e] = neg_inf
    if has_t2v:
        for h in t2v_heads:
            new_mask[:, h, q_text_s:, vis_s:vis_e] = neg_inf
    if has_t2t:
        for h in t2t_heads:
            new_mask[:, h, q_text_s:, vis_e:] = neg_inf
    return new_mask

def clear_region_masks(model: torch.nn.Module) -> int:
    cleared = 0
    for module in model.modules():
        for attr in ('_ov_v2v_head_indices', '_ov_t2v_head_indices', '_ov_t2t_head_indices', '_ov_region_runtime', '_ov_resolved_region_runtime', '_ov_region_runtime_active'):
            if hasattr(module, attr):
                delattr(module, attr)
                cleared += 1
    if hasattr(model, '_ov_region_runtime'):
        delattr(model, '_ov_region_runtime')
    return cleared
