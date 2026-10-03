from __future__ import annotations
import argparse
import copy
import importlib.util
import os
import statistics
import sys
from pathlib import Path
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[3]
PLUGIN_PATH = ROOT / 'LLaVA-NeXT' / 'llava' / 'model' / 'language_model' / 'ffn_mask_plugin.py'
SPEC = importlib.util.spec_from_file_location('ffn_mask_plugin_bench', PLUGIN_PATH)
assert SPEC is not None and SPEC.loader is not None
PLUGIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLUGIN
SPEC.loader.exec_module(PLUGIN)

class BenchMLP(nn.Module):

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

class BenchLayer(nn.Module):

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.mlp = BenchMLP(hidden, intermediate)

class BenchCore(nn.Module):

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.layers = nn.ModuleList([BenchLayer(hidden, intermediate)])

class BenchModel(nn.Module):

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.model = BenchCore(hidden, intermediate)

def dense_masked(mlp, x, masks, runtime):
    hidden = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    hidden = PLUGIN.apply_ffn_scope_masks(hidden, masks, runtime)
    return mlp.down_proj(hidden)

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
    return (statistics.median(values), values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden', type=int, default=3584)
    parser.add_argument('--intermediate', type=int, default=18944)
    parser.add_argument('--tokens', type=int, default=1024)
    parser.add_argument('--vision-fraction', type=float, default=0.9)
    parser.add_argument('--vision-keep', type=float, default=0.6)
    parser.add_argument('--text-keep', type=float, default=0.9)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('CUDA is required for this benchmark')
    os.environ['FFN_MASK_BACKEND'] = 'structured'
    os.environ['FFN_MASK_DECODE'] = '0'
    torch.manual_seed(123)
    dtype = torch.bfloat16
    dense_model = BenchModel(args.hidden, args.intermediate).to(device=args.device, dtype=dtype).eval()
    structured_model = copy.deepcopy(dense_model).eval()
    dense_mlp = dense_model.model.layers[0].mlp
    structured_mlp = structured_model.model.layers[0].mlp
    order = torch.randperm(args.intermediate)
    n_vision = round(args.intermediate * args.vision_keep)
    n_text = round(args.intermediate * args.text_keep)
    vision = torch.zeros(args.intermediate)
    text = torch.zeros(args.intermediate)
    vision[order[:n_vision]] = 1
    text_order = torch.roll(order, shifts=args.intermediate // 7)
    text[text_order[:n_text]] = 1
    masks = {'vision': vision, 'text': text}
    vis_len = max(1, round(args.tokens * args.vision_fraction))
    vis_s = (args.tokens - vis_len) // 2
    runtime = {'has_image': True, 'vis_s': vis_s, 'vis_e': vis_s + vis_len, 'prefill_kv_seq_len': args.tokens}
    structured_mlp._ov_ffn_keep_masks = {key: value.clone() for key, value in masks.items()}
    structured_mlp._ov_ffn_region_runtime = runtime
    summary = PLUGIN.prepare_ffn_compute_pruning(structured_model)
    x = torch.randn(1, args.tokens, args.hidden, device=args.device, dtype=dtype)
    with torch.inference_mode():
        expected = dense_masked(dense_mlp, x, masks, runtime)
        actual = PLUGIN.run_ffn_compute_pruned(structured_mlp, x)
        diff = (expected.float() - actual.float()).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        dense_ms, _ = timed_ms(lambda: dense_masked(dense_mlp, x, masks, runtime), args.warmup, args.repeats)
        dense_peak = torch.cuda.max_memory_allocated()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        structured_ms, _ = timed_ms(lambda: PLUGIN.run_ffn_compute_pruned(structured_mlp, x), args.warmup, args.repeats)
        structured_peak = torch.cuda.max_memory_allocated()
    print(summary.describe())
    print(f'shape: tokens={args.tokens} hidden={args.hidden} intermediate={args.intermediate} dtype={dtype}')
    print(f'numerics: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}')
    print(f'dense_ms_median={dense_ms:.3f}')
    print(f'structured_ms_median={structured_ms:.3f}')
    print(f'speedup={dense_ms / structured_ms:.3f}x')
    print(f'dense_peak_allocated_gib={dense_peak / 2 ** 30:.3f}')
    print(f'structured_peak_allocated_gib={structured_peak / 2 ** 30:.3f}')
if __name__ == '__main__':
    main()
