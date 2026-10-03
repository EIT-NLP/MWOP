import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple, Dict, List, Any

class BaseThreshold(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def generate_mask(self, **kwargs):
        return None

    def get_threshold_kwargs(self, **kwargs):
        return {'threshold': -1}

class BlasstThreshold(BaseThreshold):

    def __init__(self, threshold: Optional[float]=-1, **kwargs):
        super().__init__()
        self.threshold = threshold

    def generate_mask(self, kv_cache_len: int, sparsity: Optional[float]=0, device: Optional[torch.device]='cuda:0', **kwargs):
        block_num = (kv_cache_len + 16 - 1) // 16
        return torch.rand((block_num,), device=device) > sparsity

    def get_threshold_kwargs(self, **kwargs):
        return {'threshold': self.threshold}

class SeerThreshold(BaseThreshold):

    def __init__(self, **kwargs):
        super().__init__()

    def generate_mask(self, batch_size: int, query_len: int, kv_cache_len: int, num_key_heads: int, sparsity: Optional[float]=0, device: Optional[torch.device]='cuda:0'):
        block_num_q = (query_len + 16 - 1) // 16
        block_num_k = (kv_cache_len + 16 - 1) // 16
        return torch.rand((batch_size, num_key_heads, block_num_q, block_num_k), device=device) > sparsity
