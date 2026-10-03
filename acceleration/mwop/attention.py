import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import BlockMask, flex_attention
from .flex_mask import _build_analytic_block_mask
from .triton_prefill import attention as triton_attention
_compiled_flex = torch.compile(flex_attention, dynamic=False)

def allow_mask(heads, length, span, flags, device):
    s, e = span
    q = torch.arange(length, device=device)[:, None]
    k = torch.arange(length, device=device)[None, :]
    blocked = ((q >= s) & (q < e) & (k >= s) & (k < e))[None] & flags[0, :, None, None]
    blocked |= ((q >= e) & (k >= s) & (k < e))[None] & flags[1, :, None, None]
    blocked |= ((q >= e) & (k >= e))[None] & flags[2, :, None, None]
    return (k <= q)[None] & ~blocked

def reference(q, k, v, flags, span, fp32=False):
    mask = allow_mask(q.shape[2], q.shape[1], span, flags, q.device).unsqueeze(0)
    dtype = torch.float32 if fp32 else q.dtype
    with sdpa_kernel(SDPBackend.MATH):
        out = F.scaled_dot_product_attention(q.transpose(1, 2).to(dtype), k.transpose(1, 2).to(dtype), v.transpose(1, 2).to(dtype), attn_mask=mask, enable_gqa=True)
    return out.transpose(1, 2).to(q.dtype).contiguous()

class RegionAttention(torch.nn.Module):

    def __init__(self, flags, span, length, backend='triton', tile=(128, 32, 4, 2)):
        super().__init__()
        self.register_buffer('flags', flags)
        self.span, self.length, self.backend = (tuple(span), length, backend)
        self.tile = dict(tile) if isinstance(tile, dict) else tuple(tile)
        self.block = None

    def prepare_flex(self):
        s, e = self.span
        flags = self.flags

        def allowed(b, h, q, k):
            qv, kv = ((q >= s) & (q < e), (k >= s) & (k < e))
            return (k <= q) & ~(qv & kv & flags[0, h] | (q >= e) & kv & flags[1, h] | (q >= e) & (k >= e) & flags[2, h])
        self.block = _build_analytic_block_mask(BlockMask, allowed, num_heads=28, q_len=self.length, kv_seq_len=self.length, past_len=0, vis_s=s, vis_e=e, v2v=flags[0], t2v=flags[1], t2t=flags[2], device=flags.device)

    def forward(self, q, k, v, **kwargs):
        if q.shape[1] != self.length or k.shape[1] != self.length:
            raise ValueError('MWOP benchmark supports fixed-length fresh prefill only')
        if self.backend == 'reference':
            return reference(q, k, v, self.flags, self.span)
        if self.backend == 'triton':
            return triton_attention(q, k, v, self.flags, self.span, self.tile)
        if self.backend == 'flex':
            if self.block is None:
                raise RuntimeError('prepare_flex must run before warmup / CUDA Graph capture')
            return _compiled_flex(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=self.block, enable_gqa=True).transpose(1, 2).contiguous()
        raise ValueError(self.backend)
