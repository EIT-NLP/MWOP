from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn

def _rmsnorm_reference(hidden_states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return weight * hidden_states.to(input_dtype)

def _rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    first = hidden_states[..., :hidden_states.shape[-1] // 2]
    second = hidden_states[..., hidden_states.shape[-1] // 2:]
    return torch.cat((-second, first), dim=-1)

def _rope_qk_reference(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(2)
    sin = sin.unsqueeze(2)
    query_embed = query * cos + _rotate_half(query) * sin
    key_embed = key * cos + _rotate_half(key) * sin
    return (query_embed, key_embed)

@torch.compile(fullgraph=True)
def compiled_rmsnorm(hidden_states: torch.Tensor, weight: torch.Tensor, eps: Optional[float]=1e-06) -> torch.Tensor:
    return _rmsnorm_reference(hidden_states, weight, eps)

@torch.compile(fullgraph=True)
def compiled_rope_qk(query: torch.Tensor, key: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return _rope_qk_reference(query, key, cos, sin)
__all__ = ['compiled_rmsnorm', 'compiled_rope_qk']
