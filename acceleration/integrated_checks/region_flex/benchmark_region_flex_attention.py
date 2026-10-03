from __future__ import annotations
import argparse
import importlib.util
import os
import statistics
from pathlib import Path
import torch

def timed_ms(fn, warmup: int, repeats: int):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values)

class DummyAttention:
    training = False
    attention_dropout = 0.0

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=2048)
    parser.add_argument('--heads', type=int, default=28)
    parser.add_argument('--head-dim', type=int, default=128)
    parser.add_argument('--vision-fraction', type=float, default=0.9)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required')
    os.environ['REGION_MASK_BACKEND'] = 'flex'
    os.environ.setdefault('REGION_FLEX_COMPILE', '1')
    os.environ['REGION_FLEX_MIN_TOKENS'] = '0'
    os.environ['REGION_MASK_DECODE'] = '0'
    language_model = Path(__file__).resolve().parents[3] / 'LLaVA-NeXT' / 'llava' / 'model' / 'language_model'

    def load_plugin(name):
        spec = importlib.util.spec_from_file_location(name, language_model / f'{name}.py')
        plugin = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(plugin)
        return plugin

    build_region_attention_mask = load_plugin('attention_mask_plugin').build_region_attention_mask
    maybe_flex_region_attention = load_plugin('region_flex_attention').maybe_flex_region_attention
    torch.manual_seed(123)
    bsz = 1
    h = args.heads
    t = args.tokens
    d = args.head_dim
    dtype = torch.bfloat16
    vis_len = max(1, round(t * args.vision_fraction))
    vis_s = (t - vis_len) // 2
    vis_e = vis_s + vis_len
    module = DummyAttention()
    module._ov_v2v_head_indices = tuple(range(round(h * 0.4)))
    module._ov_t2v_head_indices = tuple(range(round(h * 0.6)))
    module._ov_t2t_head_indices = tuple(range(round(h * 0.1)))
    module._ov_region_runtime = {'has_image': True, 'orig_seq_len': t, 'n_img_placeholders': vis_len, 'img_placeholder_pos': vis_s}
    q = torch.randn(bsz, h, t, d, device=args.device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    def make_dense_mask():
        return build_region_attention_mask(None, bsz=bsz, num_heads=h, q_len=t, kv_seq_len=t, past_len=0, dtype=dtype, device=q.device, v2v_heads=module._ov_v2v_head_indices, t2v_heads=module._ov_t2v_head_indices, t2t_heads=module._ov_t2t_head_indices, runtime=module._ov_region_runtime)

    def dense_call():
        mask = make_dense_mask()
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)

    def flex_call():
        out = maybe_flex_region_attention(module, q, k, v, bsz=bsz, num_heads=h, q_len=t, kv_seq_len=t)
        if out is None:
            raise RuntimeError('FlexAttention unexpectedly fell back')
        return out
    with torch.inference_mode():
        expected = dense_call()
        actual = flex_call()
        diff = (expected.float() - actual.float()).abs()
        del expected, actual
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dense_ms = timed_ms(dense_call, args.warmup, args.repeats)
        dense_peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        flex_ms = timed_ms(flex_call, args.warmup, args.repeats)
        flex_peak = torch.cuda.max_memory_allocated()
    print(f'shape: B={bsz} H={h} T={t} D={d} dtype={dtype} vision=[{vis_s},{vis_e})')
    print(f'numerics: max_abs={diff.max().item():.6g} mean_abs={diff.mean().item():.6g}')
    print(f'dense_total_ms_median={dense_ms:.3f}')
    print(f'flex_total_ms_median={flex_ms:.3f}')
    print(f'speedup={dense_ms / flex_ms:.3f}x')
    print(f'dense_peak_allocated_gib={dense_peak / 2 ** 30:.3f}')
    print(f'flex_peak_allocated_gib={flex_peak / 2 ** 30:.3f}')
if __name__ == '__main__':
    main()
