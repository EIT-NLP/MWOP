import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from einops import rearrange
ACT2FUNC = dict(relu=nn.functional.relu, silu=nn.functional.silu, gelu=nn.functional.gelu, sigmoid=nn.functional.sigmoid)

class _PruningMLPKernel(nn.Module):

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

class DenseMLPKernel(_PruningMLPKernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def base_prefill(cls, x: torch.Tensor, w_up: torch.Tensor, w_gate: Optional[torch.Tensor]=None, b_up: Optional[torch.Tensor]=None, b_gate: Optional[torch.Tensor]=None, route_mask: Optional[torch.Tensor]=None, activation: Optional[str]='identity', **kwargs):
        if w_gate is None:
            res = F.linear(x, w_up, b_up)
            res = ACT2FUNC.get(activation, lambda x: x)(res)
            if route_mask is not None:
                res = res * route_mask[..., None].to(res.dtype)
        else:
            tmp = F.linear(x, w_gate, b_gate)
            tmp = ACT2FUNC.get(activation, lambda x: x)(tmp)
            res = F.linear(x, w_up, b_up) * tmp
            if route_mask is not None:
                res = res * route_mask[..., None].to(res.dtype)
        return res

    @classmethod
    def base_decode(cls, **kwargs):
        return cls.base_prefill(**kwargs)

    @classmethod
    def forward(cls, x: torch.Tensor, w_up: torch.Tensor, w_gate: Optional[torch.Tensor]=None, b_up: Optional[torch.Tensor]=None, b_gate: Optional[torch.Tensor]=None, route_mask: Optional[torch.Tensor]=None, activation: Optional[str]='identity', prefill_impl: Optional[str]='', **kwargs):
        assert x.dim() == 3
        assert w_up.shape[1] == x.shape[-1]
        if w_gate is not None:
            assert w_up.shape == w_gate.shape
        return cls.base_prefill(x=x, w_up=w_up, w_gate=w_gate, b_up=b_up, b_gate=b_gate, route_mask=route_mask, activation=activation, prefill_impl=prefill_impl, **kwargs)

    def _ref_forward(self, **kwargs):
        pass

    def precision_diff(self, **kwargs):
        pass

    def get_benchmark(self, **kwargs):
        pass

class LowRankMLPKernel(_PruningMLPKernel):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def base_prefill(cls, x: torch.Tensor, w_up_r1: torch.Tensor, w_up_r2: torch.Tensor, w_up: Optional[torch.Tensor]=None, w_gate: Optional[torch.Tensor]=None, w_down: Optional[torch.Tensor]=None, b_up: Optional[torch.Tensor]=None, b_gate: Optional[torch.Tensor]=None, b_down: Optional[torch.Tensor]=None, w_gate_r1: Optional[torch.Tensor]=None, w_gate_r2: Optional[torch.Tensor]=None, w_down_r1: Optional[torch.Tensor]=None, w_down_r2: Optional[torch.Tensor]=None, activation: Optional[str]='identity', **kwargs):
        if w_gate_r1 is None:
            low_rank = F.linear(x, w_up_r1, kwargs.get('b_up_r1', None))
            low_rank = F.linear(low_rank, w_up_r2, kwargs.get('b_up_r2', None))
        else:
            assert w_gate_r1 is not None and w_down_r1 is not None
            low_rank_up = F.linear(x, w_up_r1, kwargs.get('b_up_r1', None))
            low_rank_up = F.linear(low_rank_up, w_up_r2, kwargs.get('b_up_r2', None))
            low_rank_gate = F.linear(x, w_gate_r1, kwargs.get('b_gate_r1', None))
            low_rank_gate = F.linear(low_rank_gate, w_gate_r2, kwargs.get('b_gate_r2', None))
            low_rank = ACT2FUNC.get(activation, lambda x: x)(low_rank_gate) * low_rank_up
            low_rank = F.linear(low_rank, w_down_r1, kwargs.get('b_down_r1', None))
            low_rank = F.linear(low_rank, w_down_r2, kwargs.get('b_down_r2', None))
            if w_up.numel() > 0:
                retained_up = F.linear(x, w_up, b_up)
                retained_gate = F.linear(x, w_gate, b_gate)
                retained = ACT2FUNC.get(activation, lambda x: x)(retained_gate) * retained_up
                retained = F.linear(retained, w_down, b_down)
                low_rank = low_rank + retained
        return low_rank

    @classmethod
    def base_decode(cls, **kwargs):
        return cls.base_prefill(**kwargs)

    @classmethod
    def forward(cls, x: torch.Tensor, w_up_r1: torch.Tensor, w_up_r2: torch.Tensor, **kwargs):
        assert x.dim() == 3
        return cls.base_prefill(x=x, w_up_r1=w_up_r1, w_up_r2=w_up_r2, **kwargs)

    def _ref_forward(self, **kwargs):
        pass

    def precision_diff(self, **kwargs):
        pass

    def get_benchmark(self, **kwargs):
        pass
