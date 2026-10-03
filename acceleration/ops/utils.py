import os
import itertools
import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from typing import Optional, List, Tuple, Dict, Any, Callable
from einops import rearrange
os.environ['TRITON_PRINT_AUTOTUNING'] = '0'
_autotune_cache = {}

def get_shared_memory_size():
    major, minor = torch.cuda.get_device_capability('cuda')
    if major == 8 and minor == 0:
        return 163 * 1024
    elif major in (9, 10):
        return 227 * 1024
    else:
        return 99 * 1024

def generate_autotune_config(tuning_dict: Dict):
    keys = list(tuning_dict.keys())
    value_lists = [tuning_dict[key] for key in keys]
    combinations = []
    for value_combination in itertools.product(*value_lists):
        combination_dict = dict(zip(keys, value_combination))
        num_stages = combination_dict.pop('num_stages', 4)
        num_warps = combination_dict.pop('num_warps', 4)
        combinations.append(triton.Config(combination_dict, num_stages=num_stages, num_warps=num_warps))
    return combinations

class AutotuneMixin:

    @staticmethod
    def generate_autotune_config(tuning_dict: Dict):
        keys = list(tuning_dict.keys())
        value_lists = [tuning_dict[key] for key in keys]
        combinations = []
        for value_combination in itertools.product(*value_lists):
            combination_dict = dict(zip(keys, value_combination))
            num_stages = combination_dict.pop('num_stages', 3)
            num_warps = combination_dict.pop('num_warps', 4)
            combinations.append(triton.Config(combination_dict, num_stages=num_stages, num_warps=num_warps))
        return combinations

    @classmethod
    def conditional_jit(cls, enable_autotune: bool, config: Optional[List[triton.Config]]=[], keys: Optional[List[str]]=[], do_not_specialize: Optional[List[str]]=[], restored_kwargs: Optional[List[str]]=[], is_gluon: Optional[bool]=False, **kwargs):

        def decorator(func):
            if is_gluon:
                if enable_autotune:
                    return triton.autotune(configs=config, key=keys, restore_value=restored_kwargs)(gluon.jit(func, do_not_specialize=do_not_specialize))
                else:
                    return gluon.jit(func, do_not_specialize=do_not_specialize)
            elif enable_autotune:
                return triton.autotune(configs=config, key=keys, restore_value=restored_kwargs)(triton.jit(func, do_not_specialize=do_not_specialize))
            else:
                return triton.jit(func, do_not_specialize=do_not_specialize)
        return decorator

def get_autotune_cache(func: Callable, enable_autotune, config, keys, do_not_specialize: Optional[List[str]]=[], restored_kwargs: Optional[List[str]]=[], is_gluon: Optional[bool]=False):
    func_name = func.__name__
    cache_key = (func_name, enable_autotune, tuple(config), tuple(keys), tuple(do_not_specialize), is_gluon)
    if cache_key not in _autotune_cache:
        _autotune_cache[cache_key] = AutotuneMixin.conditional_jit(enable_autotune=enable_autotune, config=list(config), keys=keys, do_not_specialize=do_not_specialize, restored_kwargs=restored_kwargs, is_gluon=is_gluon)(func)
    return _autotune_cache[cache_key]

def get_autotune_config(params: List[str], **kwargs) -> List[triton.Config]:
    enable_autotune = kwargs.get('enable_autotune', False)
    if enable_autotune:
        tuning_dict = {param: kwargs.get(f'{param}_list', [kwargs.get(param)]) for param in params}
    else:
        tuning_dict = {param: [kwargs.get(param)] for param in params}
    return AutotuneMixin.generate_autotune_config(tuning_dict)

def check_shared_memory_gemm(BM: int, BN: int, BK: int, stage: int, dtype_in_byte: int):
    shared_memory_size = get_shared_memory_size()
    return (BM * BK * stage + BN * BK * stage) * dtype_in_byte <= shared_memory_size / 2

def check_shared_memory_attn(BM: int, BN: int, BK: int, stage: int, dtype_in_byte: int):
    shared_memory_size = get_shared_memory_size()
    return (BM * BK * stage + BN * BK * 2 * stage) * dtype_in_byte <= shared_memory_size

def config_optimize(m_list: List[int], n_list: List[int], num_stages: List[int], k_list: Optional[List[int]]=[32], is_attention: Optional[bool]=False, dtype_in_byte: Optional[int]=2, M: Optional[int]=1, N: Optional[int]=1, B: Optional[int]=1, H: Optional[int]=1):
    shared_memory_size = get_shared_memory_size()
    sm_num = torch.cuda.get_device_properties('cuda').multi_processor_count
    sm_loop_num = 0
    remains = 0
    best_setting = None
    for k in k_list:
        for n in n_list:
            for stage in num_stages:
                for m in m_list:
                    if is_attention:
                        smem = (m * k * num_stages + n * k * 2 * num_stages) * dtype_in_byte
                    else:
                        smem = (m * k * num_stages + n * k * num_stages) * dtype_in_byte
                    if smem > shared_memory_size / 2:
                        continue
                    if is_attention:
                        current_sm = B * H * triton.cdiv(M, m)
                    else:
                        current_sm = triton.cdiv(M, m) * triton.cdiv(N, n)
                    sm_loop_num = current_sm // sm_num
                    remains = current_sm % sm_num
                    if best_setting is None:
                        best_setting = (sm_loop_num, remains, m, n, k, stage)
                    else:
                        if sm_loop_num == 0 and best_setting[0] > 1:
                            continue
                        if best_setting[0] == 0 and sm_loop_num > 0:
                            best_setting = (sm_loop_num, remains, m, n, k, stage)
                        elif best_setting[1] < remains and best_setting[0] > 0:
                            best_setting = (sm_loop_num, remains, m, n, k, stage)
    return dict(BLOCK_M=best_setting[2], BLOCK_N=best_setting[3], BLOCK_K=best_setting[4], num_stages=best_setting[5])

def triton_rope_qk_align_fwd(q: tl.tensor, k: tl.tensor, cos: tl.tensor, sin: tl.tensor, out_q_cos: tl.tensor, out_q_sin: tl.tensor, out_k_cos: tl.tensor, out_k_sin: tl.tensor, B: tl.int64, L: tl.int64, M: tl.int64, HQ: tl.constexpr, HK: tl.constexpr, G: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr):
    mid = tl.program_id(0)
    head_id_q = tl.program_id(1)
    head_id_k = head_id_q // G
    offset_m = (mid * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
    batch_id = offset_m // L
    pos_id = offset_m % L
    offset_emb = pos_id * N
    offset_q = offset_m * HQ * N + head_id_q * N
    offset_k = offset_m * HK * N + head_id_k * N
    emb_mask = (batch_id < B) & (pos_id < L)
    cos_left = tl.load(cos + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    cos_right = tl.load(cos + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
    sin_left = tl.load(sin + offset_emb[:, None] + tl.arange(0, N // 2)[None, :], mask=emb_mask[:, None], other=0)
    sin_right = tl.load(sin + offset_emb[:, None] + tl.arange(N // 2, N)[None, :], mask=emb_mask[:, None], other=0)
    q_data_left = tl.load(q + offset_q[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < M)[:, None], other=0)
    q_data_right = tl.load(q + offset_q[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < M)[:, None], other=0)
    k_data_left = tl.load(k + offset_k[:, None] + tl.arange(0, N // 2)[None, :], mask=(offset_m < M)[:, None], other=0)
    k_data_right = tl.load(k + offset_k[:, None] + tl.arange(N // 2, N)[None, :], mask=(offset_m < M)[:, None], other=0)
    q_left_cos = q_data_left * cos_left
    q_left_sin = -q_data_right * sin_left
    q_right_cos = q_data_right * cos_right
    q_right_sin = q_data_left * sin_right
    k_left_cos = k_data_left * cos_left
    k_left_sin = -k_data_right * sin_left
    k_right_cos = k_data_right * cos_right
    k_right_sin = k_data_left * sin_right
    store_mask = (offset_m < M)[:, None]
    tl.store(out_q_cos + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_q_cos + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_cos.to(out_q_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_q_sin + offset_q[:, None] + tl.arange(0, N // 2)[None, :], q_left_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_q_sin + offset_q[:, None] + tl.arange(N // 2, N)[None, :], q_right_sin.to(out_q_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_k_cos + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_cos.to(out_k_cos.dtype.element_ty), mask=store_mask)
    tl.store(out_k_cos + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_cos.to(out_k_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(0, N // 2)[None, :], k_left_sin.to(out_k_sin.dtype.element_ty), mask=store_mask)
    tl.store(out_k_sin + offset_k[:, None] + tl.arange(N // 2, N)[None, :], k_right_sin.to(out_k_sin.dtype.element_ty), mask=store_mask)

def triton_rope_qk_align(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    out_q_cos, out_q_sin = (torch.empty_like(q), torch.empty_like(q))
    out_k_cos, out_k_sin = (torch.empty_like(k), torch.empty_like(k))
    assert q.dim() == k.dim()
    assert sin.dim() == cos.dim()
    if sin.dim() == 3:
        assert sin.shape[0] == 1
    else:
        assert sin.dim() == 2
    B, L, HQ, N = q.shape
    HK = k.shape[-2]
    G = HQ // HK
    M = q.shape[0] * q.shape[1]
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), HQ)
    config = get_autotune_config(params=['BLOCK_M'], enable_autotune=True, BLOCK_M=1, BLOCK_M_list=[1, 2, 4, 8])
    kernel = get_autotune_cache(triton_rope_qk_align_fwd, enable_autotune=True, config=config, keys=['HQ', 'HK', 'N'], do_not_specialize=['B', 'L', 'M'])
    kernel[grid](q, k, cos, sin, out_q_cos, out_q_sin, out_k_cos, out_k_sin, B, L, M, HQ, HK, G, N)
    return (out_q_cos + out_q_sin, out_k_cos + out_k_sin)

@triton.autotune(configs=generate_autotune_config(dict(BLOCK_M=[1, 2, 4, 8], num_stages=[3, 4])), key=['N', 'BLOCK_N'])
@triton.jit(do_not_specialize=['eps', 'M'])
def triton_rmsnorm_fwd(x: tl.tensor, out: tl.tensor, w: tl.tensor, eps: tl.float32, M: tl.int64, N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    mid = tl.program_id(0)
    x_ptr = tl.make_block_ptr(x, shape=(M, N), strides=(N, 1), offsets=(mid * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    accum = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        accum += x_data * x_data
    x_mean = tl.sum(accum, axis=1, keep_dims=True) / N + eps
    w_ptr = tl.make_block_ptr(w, shape=(N,), strides=(1,), offsets=(0,), block_shape=(BLOCK_N,), order=(0,))
    out_ptr = tl.make_block_ptr(out, shape=(M, N), strides=(N, 1), offsets=(mid * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    x_rsqrt = tl.math.rsqrt(x_mean)
    for i in tl.range(0, tl.cdiv(N, BLOCK_N)):
        x_data = tl.load(tl.advance(x_ptr, (0, i * BLOCK_N)), boundary_check=(0, 1), padding_option='zero')
        w_data = tl.load(tl.advance(w_ptr, (i * BLOCK_N,)), boundary_check=(0,), padding_option='zero')
        x_norm = x_data * x_rsqrt * w_data[None, :]
        tl.store(tl.advance(out_ptr, (0, i * BLOCK_N)), x_norm.to(x.dtype.element_ty), boundary_check=(0, 1))

def triton_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    original_shape = x.shape
    if x.dim() > 2:
        x_flatten = x.flatten(0, -2)
    else:
        x_flatten = x
    M, N = x_flatten.shape
    assert N == w.shape[0]
    out = torch.empty_like(x_flatten)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),)
    config = get_autotune_config(params=['BLOCK_M'], enable_autotune=True, BLOCK_M=1, BLOCK_M_list=[1, 2, 4, 8])
    kernel = get_autotune_cache(triton_rmsnorm_fwd, enable_autotune=True, config=config, keys=['N', 'BLOCK_N'], do_not_specialize=['M', 'eps'])
    kernel[grid](x, out, w, eps, M, N, BLOCK_N=min(triton.next_power_of_2(N), 1024))
    return out.reshape(original_shape)
