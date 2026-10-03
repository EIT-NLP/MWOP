from __future__ import annotations
import torch

def tilelang_sdpa_attention(query_states: torch.Tensor, key_states: torch.Tensor, value_states: torch.Tensor, attention_mask, *, scaling=None, is_causal: bool=False):
    from tilelang_fa2_4d_mask import tilelang_flash_attention
    if attention_mask is None:
        return tilelang_flash_attention(query_states, key_states, value_states, is_causal=True, scale=scaling)
    thresh = torch.finfo(attention_mask.dtype).min / 2.0
    keep = attention_mask > thresh
    if keep.dtype != torch.bool:
        keep = keep.bool()
    return tilelang_flash_attention(query_states, key_states, value_states, mask=keep, is_causal=False, scale=scaling)
