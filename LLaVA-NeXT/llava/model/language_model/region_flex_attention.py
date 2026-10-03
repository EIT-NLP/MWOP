import os
import warnings
import torch
_FLEX = None
_WARNED = False
_FLEX_BLOCK_SIZE = 128

def region_flex_enabled() -> bool:
    return os.environ.get('REGION_MASK_BACKEND', 'dense').strip().lower() == 'flex'

def region_flex_min_tokens() -> int:
    raw = os.environ.get('REGION_FLEX_MIN_TOKENS', '2048').strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f'REGION_FLEX_MIN_TOKENS must be a non-negative integer, got {raw!r}') from exc
    if value < 0:
        raise ValueError(f'REGION_FLEX_MIN_TOKENS must be non-negative, got {value}')
    return value

def _warn_once(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        warnings.warn(f'[region-flex] {msg} -> falling back to dense SDPA.')
        _WARNED = True

def _load_flex():
    global _FLEX
    if _FLEX is not None:
        return _FLEX
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
    if os.environ.get('REGION_FLEX_COMPILE', '1').strip() != '0':
        fa = torch.compile(flex_attention, dynamic=True)
    else:
        fa = flex_attention
    _FLEX = (fa, BlockMask, create_block_mask)
    return _FLEX

def _dense_blocks_to_ordered(dense_mask):
    dense_i32 = dense_mask.to(dtype=torch.int32)
    num_blocks = dense_i32.sum(dim=-1, dtype=torch.int32).contiguous()
    indices = torch.argsort(dense_i32, dim=-1, descending=True, stable=True).to(dtype=torch.int32).contiguous()
    return (num_blocks, indices)

def _build_analytic_block_mask(BlockMask, mask_mod, *, num_heads, q_len, kv_seq_len, past_len, vis_s, vis_e, v2v, t2v, t2t, device, block_size=_FLEX_BLOCK_SIZE):
    q_blocks = (q_len + block_size - 1) // block_size
    kv_blocks = (kv_seq_len + block_size - 1) // block_size
    index_dtype = torch.int64
    q_local_s = torch.arange(q_blocks, device=device, dtype=index_dtype) * block_size
    q_abs_s = q_local_s + past_len
    q_abs_e = torch.minimum(q_abs_s + (block_size - 1), torch.as_tensor(past_len + q_len - 1, device=device, dtype=index_dtype))
    k_s = torch.arange(kv_blocks, device=device, dtype=index_dtype) * block_size
    k_e = torch.minimum(k_s + (block_size - 1), torch.as_tensor(kv_seq_len - 1, device=device, dtype=index_dtype))
    region_s = torch.tensor((0, vis_s, vis_e), device=device, dtype=index_dtype)
    region_e = torch.tensor((vis_s - 1, vis_e - 1, kv_seq_len - 1), device=device, dtype=index_dtype)
    q_lo = torch.maximum(q_abs_s.unsqueeze(0), region_s.unsqueeze(1))
    q_hi = torch.minimum(q_abs_e.unsqueeze(0), region_e.unsqueeze(1))
    k_lo = torch.maximum(k_s.unsqueeze(0), region_s.unsqueeze(1))
    k_hi = torch.minimum(k_e.unsqueeze(0), region_e.unsqueeze(1))
    q_present = q_lo <= q_hi
    k_present = k_lo <= k_hi
    any_allowed = torch.zeros((num_heads, q_blocks, kv_blocks), dtype=torch.bool, device=device)
    any_blocked_cell = torch.zeros_like(any_allowed)
    never_blocked = torch.zeros(num_heads, dtype=torch.bool, device=device)
    for q_region in range(3):
        for k_region in range(3):
            blocked_heads = never_blocked
            if q_region == 1 and k_region == 1:
                blocked_heads = v2v
            elif q_region == 2 and k_region == 1:
                blocked_heads = t2v
            elif q_region == 2 and k_region == 2:
                blocked_heads = t2t
            region_cells = q_present[q_region].unsqueeze(1) & k_present[k_region].unsqueeze(0)
            causal_pair = k_lo[k_region].unsqueeze(0) <= q_hi[q_region].unsqueeze(1)
            allowed_pair = (region_cells & causal_pair).unsqueeze(0)
            any_allowed |= allowed_pair & ~blocked_heads[:, None, None]
            any_blocked_cell |= region_cells.unsqueeze(0) & blocked_heads[:, None, None]
    q_block_full = q_local_s + block_size <= q_len
    k_block_full = k_s + block_size <= kv_seq_len
    all_causal = k_e.unsqueeze(0) <= q_abs_s.unsqueeze(1)
    full_base = q_block_full[:, None] & k_block_full[None, :] & all_causal
    full_allowed = full_base.unsqueeze(0).expand(num_heads, -1, -1) & ~any_blocked_cell
    partial_allowed = any_allowed & ~full_allowed
    partial_num, partial_idx = _dense_blocks_to_ordered(partial_allowed.unsqueeze(0))
    full_num, full_idx = _dense_blocks_to_ordered(full_allowed.unsqueeze(0))
    return BlockMask.from_kv_blocks(partial_num, partial_idx, full_num, full_idx, BLOCK_SIZE=(block_size, block_size), mask_mod=mask_mod, seq_lengths=(q_len, kv_seq_len))

def _has_region_heads(module) -> bool:
    return bool(getattr(module, '_ov_v2v_head_indices', None) or getattr(module, '_ov_t2v_head_indices', None) or getattr(module, '_ov_t2t_head_indices', None))

def maybe_flex_region_attention(module, query_states, key_states, value_states, *, bsz: int, num_heads: int, q_len: int, kv_seq_len: int):
    if not region_flex_enabled():
        return None
    if not _has_region_heads(module):
        return None
    if q_len <= 1:
        return None
    if q_len < region_flex_min_tokens():
        return None
    if getattr(module, 'training', False) and float(getattr(module, 'attention_dropout', 0.0)) > 0.0:
        return None
    try:
        from .attention_mask_plugin import _compute_visual_range, region_mask_decode_enabled
        past_len = kv_seq_len - q_len
        if past_len > 0 and (not region_mask_decode_enabled()):
            return None
        rng = _compute_visual_range(getattr(module, '_ov_region_runtime', None) or {}, kv_seq_len, past_len)
        if rng is None:
            return None
        vis_s, vis_e = (int(rng[0]), int(rng[1]))
        device = query_states.device
        flex_attention_c, BlockMask, create_block_mask = _load_flex()

        def _bool_set(idxs):
            t = torch.zeros(num_heads, dtype=torch.bool, device=device)
            if idxs:
                t[torch.tensor(list(idxs), dtype=torch.long, device=device)] = True
            return t
        v2v = _bool_set(getattr(module, '_ov_v2v_head_indices', None))
        t2v = _bool_set(getattr(module, '_ov_t2v_head_indices', None))
        t2t = _bool_set(getattr(module, '_ov_t2t_head_indices', None))
        past_len_t = torch.tensor(past_len, dtype=torch.int64, device=device)
        vis_s_t = torch.tensor(vis_s, dtype=torch.int64, device=device)
        vis_e_t = torch.tensor(vis_e, dtype=torch.int64, device=device)

        def mask_mod(b, h, q_idx, kv_idx):
            q_abs = q_idx + past_len_t
            causal = kv_idx <= q_abs
            q_vis = (q_abs >= vis_s_t) & (q_abs < vis_e_t)
            q_txt = q_abs >= vis_e_t
            k_vis = (kv_idx >= vis_s_t) & (kv_idx < vis_e_t)
            k_txt = kv_idx >= vis_e_t
            blocked = q_vis & k_vis & v2v[h] | q_txt & k_vis & t2v[h] | q_txt & k_txt & t2t[h]
            return causal & ~blocked
        block_builder = os.environ.get('REGION_FLEX_BLOCK_BUILDER', 'analytic').strip().lower()
        if block_builder == 'analytic':
            block_mask = _build_analytic_block_mask(BlockMask, mask_mod, num_heads=num_heads, q_len=q_len, kv_seq_len=kv_seq_len, past_len=past_len, vis_s=vis_s, vis_e=vis_e, v2v=v2v, t2v=t2v, t2t=t2t, device=device)
        elif block_builder == 'official':
            compile_enabled = os.environ.get('REGION_FLEX_COMPILE', '1').strip() != '0'
            block_mask = create_block_mask(mask_mod, B=None, H=num_heads, Q_LEN=q_len, KV_LEN=kv_seq_len, device=device, _compile=compile_enabled)
        else:
            raise ValueError(f"REGION_FLEX_BLOCK_BUILDER must be 'analytic' or 'official', got {block_builder!r}")
        out = flex_attention_c(query_states, key_states, value_states, block_mask=block_mask, enable_gqa=False)
        return out
    except Exception as e:
        _warn_once(f'flex region attention failed ({type(e).__name__}: {e})')
        return None

def region_allow_matrix(*, num_heads, q_len, kv_seq_len, past_len, vis_s, vis_e, v2v_heads, t2v_heads, t2t_heads, device='cpu'):
    q_abs = torch.arange(q_len, device=device).view(1, q_len, 1) + past_len
    k_pos = torch.arange(kv_seq_len, device=device).view(1, 1, kv_seq_len)
    causal = k_pos <= q_abs
    q_vis = (q_abs >= vis_s) & (q_abs < vis_e)
    q_txt = q_abs >= vis_e
    k_vis = (k_pos >= vis_s) & (k_pos < vis_e)
    k_txt = k_pos >= vis_e

    def _bset(idxs):
        t = torch.zeros(num_heads, dtype=torch.bool, device=device)
        if idxs:
            t[torch.tensor(list(idxs), dtype=torch.long, device=device)] = True
        return t.view(num_heads, 1, 1)
    v2v = _bset(v2v_heads)
    t2v = _bset(t2v_heads)
    t2t = _bset(t2t_heads)
    blocked = q_vis & k_vis & v2v | q_txt & k_vis & t2v | q_txt & k_txt & t2t
    return causal & ~blocked
