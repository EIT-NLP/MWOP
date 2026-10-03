import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple, Dict, List, Any
from ops.utils import get_autotune_config, get_autotune_cache, check_shared_memory_attn
EVT_COMPUTE = tl.constexpr(0)
EVT_MEMORY = tl.constexpr(1)
EVT_SYNC = tl.constexpr(2)

def dense_prefill_impl(q: tl.tensor, k: tl.tensor, v: tl.tensor, pad_offset: tl.tensor, out: tl.tensor, B: tl.int64, LQ: tl.int64, LK: tl.int64, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, qk_scale: tl.float32, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    query_id, query_head_id, batch_id = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qk_scale *= 1.44269504
    key_head_id = query_head_id // G
    query_batch_offset = batch_id * LQ * HQ * D
    query_seq_offset = query_id * BLOCK_M + tl.arange(0, BLOCK_M)
    query_head_offset = query_head_id * D
    key_batch_offset = batch_id * LK * HK * D
    key_head_offset = key_head_id * D
    pad_offset_kv = tl.load(pad_offset + batch_id)
    key_length = LK - pad_offset_kv
    query_range_mask = query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ
    query_data = tl.load(q + (query_batch_offset + query_seq_offset[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]), mask=query_range_mask[:, None], other=0.0)
    max_pos_q = tl.max(query_seq_offset)
    num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)
    for tile_kv in tl.range(0, num_kv_iter):
        key_range_mask = tile_kv * BLOCK_N + tl.arange(0, BLOCK_N) < key_length
        key_data = tl.load(k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]), mask=key_range_mask[:, None], other=0.0)
        qk = tl.dot(query_data, key_data.T) * qk_scale
        causal_mask = query_seq_offset[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
        qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))
        value_data = tl.load(v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]), mask=key_range_mask[:, None], other=0.0)
        score_max_new = tl.maximum(score_max, tl.max(qk, 1))
        score_scale = tl.exp2(score_max - score_max_new)
        qk = tl.exp2(qk - score_max_new[:, None])
        score_sum = score_sum * score_scale + tl.sum(qk, 1)
        acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
        score_max = score_max_new
    acc /= score_sum[:, None]
    tl.store(out + (query_batch_offset + query_seq_offset[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]), acc.to(out.dtype.element_ty), mask=query_range_mask[:, None])

def query_sort_prefill_impl(q: tl.tensor, k: tl.tensor, v: tl.tensor, route_mask: tl.tensor, route_indices: tl.tensor, pad_offset: tl.tensor, out: tl.tensor, B: tl.int64, LQ: tl.int64, LK: tl.int64, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, qk_scale: tl.float32, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, IS_OFFLINE: tl.constexpr):
    query_id, query_head_id, batch_id = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    BLOCK_NUM = tl.num_programs(0)
    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    qk_scale *= 1.44269504
    key_head_id = query_head_id // G
    if IS_OFFLINE:
        skip_flag = tl.load(route_mask + batch_id * BLOCK_NUM + query_id)
    else:
        query_mask = tl.load(route_mask + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M), mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ, other=0)
        skip_flag = tl.reduce_or(query_mask, axis=-1)
    if skip_flag > 0:
        if IS_OFFLINE:
            query_indices = tl.load(route_indices + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M), mask=query_id * BLOCK_M + tl.arange(0, BLOCK_M) < LQ, other=-1)
            query_range_mask = query_indices >= 0
            query_indices = tl.where(query_range_mask, query_indices, 0)
        else:
            query_range_mask = query_mask > 0
            query_indices = tl.load(route_indices + batch_id * LQ + query_id * BLOCK_M + tl.arange(0, BLOCK_M), mask=query_range_mask, other=-1)
        query_batch_offset = batch_id * LQ * HQ * D
        query_head_offset = query_head_id * D
        key_batch_offset = batch_id * LK * HK * D
        key_head_offset = key_head_id * D
        pad_offset_kv = tl.load(pad_offset + batch_id)
        key_length = LK - pad_offset_kv
        query_data = tl.load(q + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]), mask=query_range_mask[:, None], other=0.0)
        max_pos_q = tl.max(query_indices)
        num_kv_iter = tl.cdiv(max_pos_q - pad_offset_kv + 1, BLOCK_N)
        for tile_kv in tl.range(0, num_kv_iter):
            key_range_mask = tile_kv * BLOCK_N + tl.arange(0, BLOCK_N) < key_length
            key_data = tl.load(k + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]), mask=key_range_mask[:, None], other=0.0)
            qk = tl.dot(query_data, key_data.T) * qk_scale
            causal_mask = query_indices[:, None] >= (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[None, :]
            qk = tl.where(causal_mask & (query_range_mask[:, None] & key_range_mask[None, :]), qk, -float('inf'))
            value_data = tl.load(v + (key_batch_offset + (tile_kv * BLOCK_N + pad_offset_kv + tl.arange(0, BLOCK_N))[:, None] * HK * D + key_head_offset + tl.arange(0, D)[None, :]), mask=key_range_mask[:, None], other=0.0)
            score_max_new = tl.maximum(score_max, tl.max(qk, 1))
            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        acc /= score_sum[:, None]
        tl.store(out + (query_batch_offset + query_indices[:, None] * HQ * D + query_head_offset + tl.arange(0, D)[None, :]), acc.to(out.dtype.element_ty), mask=query_range_mask[:, None])

class DensePrefill:
    support_kernel = ['dense']

    @classmethod
    def _dense_kernel(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_offset: Optional[torch.Tensor]=None, BLOCK_M: Optional[int]=64, BLOCK_N: Optional[int]=32, num_stages: Optional[int]=3, num_warps: Optional[int]=4, do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'], **kwargs):
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        out = torch.zeros_like(q)
        grid = lambda meta: (triton.cdiv(LQ, meta['BLOCK_M']), HQ, B)
        config = get_autotune_config(params=['BLOCK_M', 'BLOCK_N', 'num_stages'], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=num_stages, **kwargs)
        kernel = get_autotune_cache(dense_prefill_impl, enable_autotune=True, config=config, keys=['B', 'LQ', 'LK'], do_not_specialize=['qk_scale'])
        kernel[grid](q, k, v, pad_offset, out, B, LQ, LK, HQ, HK, D, G, D ** (-0.5))
        return out

    @classmethod
    def kernel(cls, impl: str, **kwargs):
        q = kwargs.get('q')
        dtype = q.dtype
        device = q.device
        B, LQ, HQ, D = q.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        num_sm = torch.cuda.get_device_properties('cuda').multi_processor_count
        cc = torch.cuda.get_device_capability('cuda')[0]
        num_stages = 2
        num_warps = 4
        BLOCK_M = triton.next_power_of_2(LQ)
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LQ)))
        assert D <= 128, f'sm80 style flash_attn head_dim {D} must be <= 128'
        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_N > 32:
                BLOCK_N >>= 1
            elif BLOCK_M > 32:
                BLOCK_M >>= 1
            else:
                num_stages -= 1
        while HQ * B * triton.cdiv(LQ, BLOCK_M) < num_sm:
            if BLOCK_M > 16:
                BLOCK_M >>= 1
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
        return getattr(cls, f'_dense_kernel')(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=num_stages, num_warps=num_warps, pad_offset=pad_offset, BLOCK_M_list=[16, 32, 64], BLOCK_N_list=[32, 64], num_stages_list=[2, 3], **kwargs)

class QuerySparsePrefill:
    support_kernel = ['sort_offline', 'sort_online']

    @classmethod
    def _sort_kernel(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, route_mask: torch.Tensor, pad_offset: Optional[torch.Tensor]=None, BLOCK_M: Optional[int]=64, BLOCK_N: Optional[int]=32, num_stages: Optional[int]=3, num_warps: Optional[int]=4, do_not_specialize: Optional[List]=['LQ', 'LK', 'qk_scale'], **kwargs):
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        m_sort, m_sort_indices = torch.sort(route_mask, descending=True, stable=False)
        if kwargs.get('is_offline', False):
            m_sort_indices = m_sort_indices.masked_fill(m_sort.logical_not(), -1)
            m_sort = torch.nn.functional.pad(m_sort, (0, BLOCK_M - LQ % BLOCK_M), value=0).reshape(B, -1, BLOCK_M)
            m_sort = m_sort.any(dim=-1)
        out = torch.zeros_like(q)
        grid = lambda meta: (triton.cdiv(LQ, meta['BLOCK_M']), HQ, B)
        config = get_autotune_config(params=['BLOCK_N', 'num_stages'], BLOCK_N=BLOCK_N, num_stages=num_stages, **kwargs)
        kernel = get_autotune_cache(query_sort_prefill_impl, enable_autotune=True, config=config, keys=['B', 'LQ', 'LK'], do_not_specialize=['qk_scale'])
        kernel[grid](q, k, v, m_sort, m_sort_indices, pad_offset, out, B, LQ, LK, HQ, HK, D, G, D ** (-0.5), BLOCK_M=BLOCK_M, IS_OFFLINE=kwargs.get('is_offline', False))
        return out

    @classmethod
    def kernel(cls, impl: str, **kwargs):
        q = kwargs.get('q')
        dtype = q.dtype
        device = q.device
        B, LQ, HQ, D = q.shape
        route_mask = kwargs.pop('route_mask', None)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        num_sm = torch.cuda.get_device_properties('cuda').multi_processor_count
        cc = torch.cuda.get_device_capability('cuda')[0]
        num_stages = 2
        num_warps = 4
        if impl == 'auto':
            impl = 'sort_offline'
        estimated_sparsity = kwargs.pop('estimated_sparsity', 1)
        if route_mask is None:
            estimated_sparsity = 1
            route_mask = torch.ones((B, LQ), dtype=torch.bool, device=device)
        if estimated_sparsity == 0:
            estimated_sparsity = 1
        BLOCK_M = triton.next_power_of_2(int(LQ * estimated_sparsity))
        if BLOCK_M >= int(LQ * estimated_sparsity):
            BLOCK_M = BLOCK_M >> 1
        BLOCK_M = min(128, max(16, BLOCK_M))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LQ)))
        assert D <= 128, f'sm80 style flash_attn head_dim {D} must be <= 128'
        while not check_shared_memory_attn(BLOCK_M, BLOCK_N, D, num_stages, dtype.itemsize):
            if BLOCK_N > 32:
                BLOCK_N >>= 1
            elif BLOCK_M > 32:
                BLOCK_M >>= 1
            elif num_stages > 2:
                num_stages -= 1
            else:
                break
        while HQ * B * triton.cdiv(LQ, BLOCK_M) < num_sm:
            if BLOCK_M > 16:
                BLOCK_M >>= 1
            else:
                break
        BLOCK_M = kwargs.pop('BLOCK_M', BLOCK_M)
        BLOCK_N = kwargs.pop('BLOCK_N', BLOCK_N)
        if impl not in cls.support_kernel:
            raise ValueError(f'{impl} is not supported')
        is_offline = impl == 'sort_offline'
        impl = impl.split('_')[0]
        return getattr(cls, f'_{impl}_kernel')(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=num_stages, num_warps=num_warps, route_mask=route_mask, pad_offset=pad_offset, is_offline=is_offline, BLOCK_M_list=[16, 32, 64], BLOCK_N_list=[32, 64], num_stages_list=[2, 3], **kwargs)
