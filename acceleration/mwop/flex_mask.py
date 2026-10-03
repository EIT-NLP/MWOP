import torch
_FLEX_BLOCK_SIZE = 128

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
