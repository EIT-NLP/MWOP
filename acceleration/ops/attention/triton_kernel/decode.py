import os
import itertools
import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple, Dict, List, Any
from ops.utils import get_autotune_config, get_autotune_cache

def dense_decode_impl(q: tl.tensor, k: tl.tensor, v: tl.tensor, pad_offset: tl.tensor, out: tl.tensor, metadata: tl.tensor, LK: tl.int64, LBK: tl.int64, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, NBK: tl.int64, qk_scale: tl.float32, kv_split: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    split_id, key_head_id, batch_id = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    qk_scale *= 1.44269504
    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    start_q, end_q = (batch_id, batch_id + 1)
    seq_offset_q = start_q
    pad_offset_k = tl.load(pad_offset + batch_id)
    start_k = batch_id * LK + split_id * LBK + pad_offset_k
    end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
    key_length = end_k - start_k
    query_range_mask = tl.arange(0, BLOCK_M) < G
    query_data = tl.load(q + (seq_offset_q * HQ * D + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :], mask=query_range_mask[:, None], other=0.0)
    split_n_range = tl.cdiv(key_length, BLOCK_N)
    for tile_n in tl.range(0, split_n_range):
        key_range_mask = tile_n * BLOCK_N + tl.arange(0, BLOCK_N) < key_length
        key_data = tl.load(k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :], mask=key_range_mask[:, None], other=0.0)
        qk = tl.dot(query_data, key_data.T) * qk_scale
        qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))
        value_data = tl.load(v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :], mask=key_range_mask[:, None], other=0.0)
        score_max_new = tl.maximum(score_max, tl.max(qk, 1))
        score_scale = tl.exp2(score_max - score_max_new)
        qk = tl.exp2(qk - score_max_new[:, None])
        score_sum = score_sum * score_scale + tl.sum(qk, 1)
        acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
        score_max = score_max_new
    if not kv_split:
        acc /= score_sum[:, None]
    tl.store(out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :], acc.to(out.dtype.element_ty), mask=query_range_mask[:, None])
    if kv_split:
        tl.store(metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id, score_max.to(metadata.dtype.element_ty), mask=query_range_mask)
        tl.store(metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id, score_sum.to(metadata.dtype.element_ty), mask=query_range_mask)

def query_sort_decode_impl(q: tl.tensor, k: tl.tensor, v: tl.tensor, route_mask: tl.tensor, pad_offset: tl.tensor, out: tl.tensor, metadata: tl.tensor, LK: tl.int64, LBK: tl.int64, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, NBK: tl.int64, qk_scale: tl.float32, kv_split: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    split_id, key_head_id, batch_id = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    qk_scale *= 1.44269504
    score_max = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    score_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    is_active = tl.load(route_mask + batch_id)
    if is_active == 1:
        start_q, end_q = (batch_id, batch_id + 1)
        seq_offset_q = start_q
        pad_offset_k = tl.load(pad_offset + batch_id)
        start_k = batch_id * LK + split_id * LBK + pad_offset_k
        end_k = tl.minimum(start_k + LBK, (batch_id + 1) * LK)
        key_length = end_k - start_k
        query_range_mask = tl.arange(0, BLOCK_M) < G
        query_data = tl.load(q + (seq_offset_q * HQ * D + (key_head_id * G + tl.arange(0, BLOCK_M)) * D)[:, None] + tl.arange(0, D)[None, :], mask=query_range_mask[:, None], other=0.0)
        split_n_range = tl.cdiv(key_length, BLOCK_N)
        for tile_n in tl.range(0, split_n_range):
            key_range_mask = tile_n * BLOCK_N + tl.arange(0, BLOCK_N) < key_length
            key_data = tl.load(k + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :], mask=key_range_mask[:, None], other=0.0)
            qk = tl.dot(query_data, key_data.T) * qk_scale
            qk = tl.where(query_range_mask[:, None] & key_range_mask[None, :], qk, -float('inf'))
            value_data = tl.load(v + ((start_k + tile_n * BLOCK_N + tl.arange(0, BLOCK_N)) * HK * D + key_head_id * D)[:, None] + tl.arange(0, D)[None, :], mask=key_range_mask[:, None], other=0.0)
            score_max_new = tl.maximum(score_max, tl.max(qk, 1))
            score_scale = tl.exp2(score_max - score_max_new)
            qk = tl.exp2(qk - score_max_new[:, None])
            score_sum = score_sum * score_scale + tl.sum(qk, 1)
            acc = acc * score_scale[:, None] + tl.dot(qk.to(q.dtype.element_ty), value_data)
            score_max = score_max_new
        if not kv_split:
            acc /= score_sum[:, None]
        tl.store(out + (seq_offset_q * HQ * NBK * D + (key_head_id * G + tl.arange(0, BLOCK_M))[:, None] * NBK * D + split_id * D) + tl.arange(0, D)[None, :], acc.to(out.dtype.element_ty), mask=query_range_mask[:, None])
        if kv_split:
            tl.store(metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + split_id, score_max.to(metadata.dtype.element_ty), mask=query_range_mask)
            tl.store(metadata + seq_offset_q * HQ * 2 * NBK + (key_head_id * G + tl.arange(0, BLOCK_M)) * 2 * NBK + NBK + split_id, score_sum.to(metadata.dtype.element_ty), mask=query_range_mask)

@triton.jit(do_not_specialize=['NBK'])
def merge_impl(out_tmp: tl.tensor, metadata: tl.tensor, route_mask: tl.tensor, out: tl.tensor, HQ: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, G: tl.constexpr, NBK: tl.int64, NBK2: tl.constexpr, sparse_type: tl.constexpr):
    query_head_id, batch_id = (tl.program_id(0), tl.program_id(1))
    route_offset = -1
    if sparse_type == 'query':
        route_offset = batch_id
    NBK_mask = tl.arange(0, NBK2) < NBK
    is_active = 1
    if sparse_type != 'none':
        is_active = tl.load(route_mask + route_offset)
    split_range = tl.arange(0, NBK2)
    if is_active == 1:
        acc = tl.load(out_tmp + (batch_id * HQ + query_head_id) * NBK * D + (split_range * D)[:, None] + tl.arange(0, D)[None, :], mask=NBK_mask[:, None], other=0.0)
        score_max = tl.load(metadata + (batch_id * HQ + query_head_id) * 2 * NBK + split_range, mask=NBK_mask, other=-float('inf'))
        score_sum = tl.load(metadata + (batch_id * HQ + query_head_id) * 2 * NBK + NBK + split_range, mask=NBK_mask, other=0.0)
        global_max = tl.max(score_max)
        global_scale = tl.exp2(score_max - global_max)
        score_sum *= global_scale
        global_sum = tl.sum(score_sum)
        acc = tl.sum(acc * global_scale[:, None], axis=0)
        acc /= global_sum
        tl.store(out + (batch_id * HQ + query_head_id) * D + tl.arange(0, D), acc.to(out.dtype.element_ty))

class DenseDecode:
    support_kernel = ['split', 'fuse']

    @classmethod
    def _dense_kernel(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pad_offset: Optional[torch.Tensor]=None, BLOCK_M: Optional[int]=16, BLOCK_N: Optional[int]=32, split_size: Optional[int]=0, num_split: Optional[int]=1, num_stages: Optional[int]=2, num_warps: Optional[int]=4, do_not_specialize: Optional[List]=['LK', 'LBK', 'NBK', 'qk_scale'], **kwargs):
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4
        _, LK, HK, _ = k.shape
        G = HQ // HK
        out = torch.zeros_like(q)
        grid = lambda meta: (num_split, HK, B)
        config = get_autotune_config(params=['BLOCK_M', 'BLOCK_N', 'num_stages'], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=num_stages, **kwargs)
        kernel = get_autotune_cache(dense_decode_impl, enable_autotune=True, config=config, keys=['LK'], do_not_specialize=['NBK', 'qk_scale'])
        if split_size == 0:
            kernel[grid](q, k, v, pad_offset, out, None, LK, split_size, HQ, HK, D, G, num_split, D ** (-0.5), kv_split=False)
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
            kernel[grid](q, k, v, pad_offset, out_local, metadata, LK, split_size, HQ, HK, D, G, num_split, D ** (-0.5), kv_split=True)
            grid = lambda meta: (HQ, B)
            merge_impl[grid](out_local, metadata, None, out, HQ, HK, D, G, num_split, triton.next_power_of_2(num_split), sparse_type='none', num_stages=2, num_warps=4)
        return out

    @classmethod
    def kernel(cls, impl: str, **kwargs):
        num_sm = torch.cuda.get_device_properties('cuda').multi_processor_count
        cc = torch.cuda.get_device_capability('cuda')[0]
        if impl == 'auto':
            impl = 'split'
        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        BLOCK_M = max(16, triton.next_power_of_2(G))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        num_stages = 2
        num_warps = 4
        num_split = 1
        while num_split * B * HK < num_sm * 2 and int(LK / num_split) > max(BLOCK_N, 64):
            num_split += 1
        if impl not in cls.support_kernel:
            raise ValueError(f'{impl} is not supported')
        split_size = triton.cdiv(LK, num_split)
        return getattr(cls, f'_dense_kernel')(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, split_size=split_size if impl == 'split' and num_split > 1 else 0, num_split=num_split if impl == 'split' else 1, num_stages=num_stages, num_warps=num_warps, pad_offset=pad_offset, BLOCK_N_list=[16, 32, 64], num_stages_list=[2, 3], **kwargs)

class QuerySparseDecode:
    support_kernel = ['split', 'fuse']

    @classmethod
    def _sort_kernel(cls, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, route_mask: torch.Tensor, pad_offset: Optional[torch.Tensor]=None, BLOCK_M: Optional[int]=16, BLOCK_N: Optional[int]=32, split_size: Optional[int]=0, num_split: Optional[int]=1, num_stages: Optional[int]=2, num_warps: Optional[int]=4, do_not_specialize: Optional[List]=['LK', 'LBK', 'NBK', 'qk_scale'], **kwargs):
        assert q.shape[1] == 1 and q.shape[:2] == route_mask.shape
        B, LQ, HQ, D = q.shape
        assert k.dim() == 4
        _, LK, HK, _ = k.shape
        G = HQ // HK
        out = torch.zeros_like(q)
        grid = lambda meta: (num_split, HK, B)
        config = get_autotune_config(params=['BLOCK_M', 'BLOCK_N', 'num_stages'], BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, num_stages=num_stages, **kwargs)
        kernel = get_autotune_cache(query_sort_decode_impl, enable_autotune=True, config=config, keys=['LK'], do_not_specialize=['NBK', 'qk_scale'])
        if split_size == 0:
            kernel[grid](q, k, v, route_mask, pad_offset, out, None, LK, split_size, HQ, HK, D, G, num_split, D ** (-0.5), kv_split=False)
        else:
            out_local = torch.zeros((B, LQ, HQ, num_split, D), dtype=torch.float32, device=q.device)
            metadata = torch.zeros((B, LQ, HQ, 2, num_split), dtype=torch.float32, device=q.device)
            kernel[grid](q, k, v, route_mask, pad_offset, out_local, metadata, LK, split_size, HQ, HK, D, G, num_split, D ** (-0.5), kv_split=True)
            grid = lambda meta: (HQ, B)
            merge_impl[grid](out_local, metadata, route_mask, out, HQ, HK, D, G, num_split, triton.next_power_of_2(num_split), sparse_type='query', num_stages=2, num_warps=4)
        return out

    @classmethod
    def kernel(cls, impl: str, **kwargs):
        num_sm = torch.cuda.get_device_properties('cuda').multi_processor_count
        cc = torch.cuda.get_device_capability('cuda')[0]
        if impl == 'auto':
            impl = 'split'
        q = kwargs.get('q')
        k = kwargs.get('k')
        dtype = q.dtype
        device = q.device
        B, LQ, HQ, D = q.shape
        _, LK, HK, _ = k.shape
        G = HQ // HK
        route_mask = kwargs.pop('route_mask', None)
        if route_mask is None:
            route_mask = torch.ones((B, LQ), dtype=torch.int32, device=device)
        pad_offset = kwargs.pop('pad_offset', None)
        if pad_offset is None:
            pad_offset = torch.zeros((B,), dtype=torch.int32, device=device)
        BLOCK_M = max(16, triton.next_power_of_2(G))
        BLOCK_N = min(64, max(16, triton.next_power_of_2(LK)))
        num_stages = 2
        num_warps = 4
        num_split = 1
        while num_split * B * HK < num_sm * 2 and int(LK / num_split) > max(BLOCK_N, 64):
            num_split += 1
        if impl not in cls.support_kernel:
            raise ValueError(f'{impl} is not supported')
        split_size = triton.cdiv(LK, num_split)
        return getattr(cls, f'_sort_kernel')(BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, split_size=split_size if impl == 'split' and num_split > 1 else 0, num_split=num_split if impl == 'split' else 1, num_stages=num_stages, num_warps=num_warps, route_mask=route_mask, pad_offset=pad_offset, BLOCK_N_list=[16, 32, 64], num_stages_list=[2, 3], **kwargs)
