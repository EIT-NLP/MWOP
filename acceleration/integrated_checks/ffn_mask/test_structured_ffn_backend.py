from __future__ import annotations
import copy
import importlib.util
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
ROOT = Path(__file__).resolve().parents[3]
PLUGIN_PATH = ROOT / 'LLaVA-NeXT' / 'llava' / 'model' / 'language_model' / 'ffn_mask_plugin.py'
SPEC = importlib.util.spec_from_file_location('ffn_mask_plugin_under_test', PLUGIN_PATH)
assert SPEC is not None and SPEC.loader is not None
PLUGIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PLUGIN
SPEC.loader.exec_module(PLUGIN)
apply_ffn_scope_masks = PLUGIN.apply_ffn_scope_masks
prepare_ffn_compute_pruning = PLUGIN.prepare_ffn_compute_pruning
restore_ffn_compute_pruning = PLUGIN.restore_ffn_compute_pruning
run_ffn_compute_pruned = PLUGIN.run_ffn_compute_pruned

class TinyMLP(nn.Module):

    def __init__(self, hidden: int=8, intermediate: int=12):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)
        self.act_fn = nn.SiLU()

class TinyLayer(nn.Module):

    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

class TinyCore(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer()])

class TinyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = TinyCore()

def dense_masked(mlp: TinyMLP, x: torch.Tensor, masks, runtime) -> torch.Tensor:
    hidden = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    hidden = apply_ffn_scope_masks(hidden, masks, runtime)
    return mlp.down_proj(hidden)

def dense_full(mlp: TinyMLP, x: torch.Tensor) -> torch.Tensor:
    hidden = mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)
    return mlp.down_proj(hidden)

def run_case(vision_keep: torch.Tensor, text_keep: torch.Tensor) -> None:
    torch.manual_seed(7)
    original = TinyModel().eval()
    model = copy.deepcopy(original).eval()
    mlp = model.model.layers[0].mlp
    baseline_mlp = original.model.layers[0].mlp
    runtime = {'has_image': True, 'vis_s': 2, 'vis_e': 7, 'prefill_kv_seq_len': 9}
    masks = {'vision': vision_keep.float(), 'text': text_keep.float()}
    mlp._ov_ffn_keep_masks = {key: value.clone() for key, value in masks.items()}
    mlp._ov_ffn_region_runtime = runtime
    x = torch.randn(2, 9, 8)
    expected_prefill = dense_masked(baseline_mlp, x, masks, runtime)
    summary = prepare_ffn_compute_pruning(model)
    assert summary.backend == 'structured'
    assert summary.layer_count == 1
    actual_prefill = run_ffn_compute_pruned(mlp, x)
    torch.testing.assert_close(actual_prefill, expected_prefill, rtol=1e-05, atol=1e-06)
    decode_x = torch.randn(2, 1, 8)
    expected_decode = dense_full(baseline_mlp, decode_x)
    actual_decode = run_ffn_compute_pruned(mlp, decode_x)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-05, atol=1e-06)
    assert restore_ffn_compute_pruning(model) == 1
    restored = dense_full(mlp, x)
    expected_full = dense_full(baseline_mlp, x)
    torch.testing.assert_close(restored, expected_full, rtol=1e-05, atol=1e-06)

def run_all_and_decode_case() -> None:
    torch.manual_seed(11)
    original = TinyModel().eval()
    model = copy.deepcopy(original).eval()
    mlp = model.model.layers[0].mlp
    baseline_mlp = original.model.layers[0].mlp
    runtime = {'has_image': True, 'vis_s': 1, 'vis_e': 5, 'prefill_kv_seq_len': 7}
    masks = {'all': torch.tensor([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]).float(), 'text': torch.tensor([1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1]).float()}
    mlp._ov_ffn_keep_masks = {key: value.clone() for key, value in masks.items()}
    mlp._ov_ffn_region_runtime = runtime
    prepare_ffn_compute_pruning(model)
    os.environ['FFN_MASK_DECODE'] = '1'
    decode_x = torch.randn(2, 1, 8)
    expected_decode = dense_masked(baseline_mlp, decode_x, masks, runtime)
    actual_decode = run_ffn_compute_pruned(mlp, decode_x)
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-05, atol=1e-06)
    text_runtime = {'has_image': False}
    mlp._ov_ffn_region_runtime = text_runtime
    text_x = torch.randn(2, 4, 8)
    expected_text = dense_masked(baseline_mlp, text_x, masks, text_runtime)
    actual_text = run_ffn_compute_pruned(mlp, text_x)
    torch.testing.assert_close(actual_text, expected_text, rtol=1e-05, atol=1e-06)
    os.environ['FFN_MASK_DECODE'] = '0'

def main() -> None:
    os.environ['FFN_MASK_BACKEND'] = 'structured'
    os.environ['FFN_MASK_DECODE'] = '0'
    run_case(vision_keep=torch.tensor([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]), text_keep=torch.tensor([1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0]))
    run_case(vision_keep=torch.zeros(12), text_keep=torch.tensor([1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1]))
    run_all_and_decode_case()
    print('PASS: structured FFN matches dense masks for prefill, decode, and restore')
if __name__ == '__main__':
    main()
