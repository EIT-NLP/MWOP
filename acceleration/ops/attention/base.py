import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
from flash_attn import flash_attn_with_kvcache
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _flash_attention_forward, flash_attn_supports_top_left_mask
from .triton_kernel.prefill import DensePrefill
from .triton_kernel.decode import DenseDecode
_use_top_left_mask = flash_attn_supports_top_left_mask()

class _PruningAttentionKernel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    @abstractmethod
    def base_prefill(cls, **kwargs):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def base_decode(cls, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _ref_forward(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def precision_diff(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_benchmark(self, **kwargs):
        raise NotImplementedError

class DenseAttentionKernel(_PruningAttentionKernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def base_prefill(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, enable_triton: Optional[bool]=False, **kwargs):
        if enable_triton:
            attn_output = DensePrefill.kernel(q=q, k=k, v=v, pad_offset=None, impl='auto', **kwargs)
        else:
            attn_output = _flash_attention_forward(q, k, v, attention_mask, query_length=q.shape[1], is_causal=True, dropout=0.0, softmax_scale=q.shape[-1] ** (-0.5), use_top_left_mask=_use_top_left_mask, attn_implementation='flash_attention_2')
        return attn_output

    @classmethod
    def base_decode(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_offset: Optional[torch.Tensor]=None, enable_xqa: Optional[bool]=True, **kwargs):
        if enable_xqa and q.shape[2] > k.shape[2]:
            attn_output = DenseDecode.kernel(q=q, k=k, v=v, pad_offset=pad_offset, impl='auto', **kwargs)
        else:
            attn_output = flash_attn_with_kvcache(q, k, v, causal=True, cache_leftpad=pad_offset.to(torch.int32))
        return attn_output

    @classmethod
    def forward(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, route_mask: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, pad_offset: Optional[torch.Tensor]=None, prefill_impl: Optional[str]='', decode_impl: Optional[str]='', enable_autotune: Optional[bool]=False, enable_xqa: Optional[bool]=True, **kwargs):
        assert k.dim() == 4 and q.dim() == 4
        if q.shape[1] > 1:
            out = cls.base_prefill(q, k, v, attention_mask)
            if route_mask is not None:
                while route_mask.dim() < out.dim():
                    route_mask = route_mask.unsqueeze(-1)
                out = out * route_mask
        else:
            if pad_offset is None:
                pad_offset = torch.zeros(q.shape[0], device=q.device)
            out = cls.base_decode(q, k, v, pad_offset, enable_xqa=enable_xqa, enable_autotune=enable_autotune)
            if route_mask is not None:
                while route_mask.dim() < out.dim():
                    route_mask = route_mask.unsqueeze(-1)
                out = out * route_mask
        return out

    def _ref_forward(self, **kwargs):
        pass

    def precision_diff(self, **kwargs):
        pass

    def get_benchmark(self, **kwargs):
        pass
