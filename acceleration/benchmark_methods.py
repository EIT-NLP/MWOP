import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from benchmark_mwop import measure, kv_tensors, write_json
from mwop.core_bench import error_stats
from mwop.model import load_model
from mwop.ffn import VisionMLP
from mwop.methods import METHODS, MethodController
from mwop.method_theory import compute
from ops import __KV_CACHE__
from ops.compiled_ops import compiled_rmsnorm

def reference_attention(q, k, v, prefix, visual, kind, window=256):
    n = k.shape[1]
    p, e = (prefix, prefix + visual)
    positions = list(range(p)) + list(range(e, n)) if kind == 'shortv' else list(range(n))
    keys = torch.arange(n, device=q.device)[None]
    outs = []
    with sdpa_kernel(SDPBackend.MATH):
        for start in range(0, len(positions), 128):
            queries = torch.tensor(positions[start:start + 128], device=q.device)[:, None]
            mask = keys <= queries
            if kind == 'redundancy_lens':
                mask &= ~((queries >= p) & (queries < e) & (keys >= p) & (keys < e) & (keys < queries - window))
            y = F.scaled_dot_product_attention(q[:, start:start + 128].transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float(), attn_mask=mask, enable_gqa=True)
            outs.append(y.transpose(1, 2).to(q.dtype))
    return torch.cat(outs, dim=1)

def bounded_error(actual, expected):
    r = error_stats(actual, expected, check=False)
    assert r['relative_rms'] < 0.01 and r['max_row_relative_l2'] < 0.02, r
    return r

def compare(a, b, lengths):
    r = dict(logits=error_stats(a.logits, b.logits, atol=0.01, rtol=0.01), kv=[])
    for i, ((ak, av), (bk, bv)) in enumerate(zip(kv_tensors(a.past_key_values), kv_tensors(b.past_key_values))):
        assert ak.shape[1] == bk.shape[1] == lengths[i]
        r['kv'].append(dict(layer=i, length=lengths[i], k=error_stats(ak, bk, atol=0.01, rtol=0.01), v=error_stats(av, bv, atol=0.01, rtol=0.01)))
    assert len(r['kv']) == 28 and torch.equal(a.logits.argmax(-1), b.logits.argmax(-1))
    return r

@torch.inference_mode()
def token_reference(model, ctl, x):
    full_pe = model.model.rotary_emb(x, torch.arange(x.shape[1], device=x.device)[None])
    cache = __KV_CACHE__['base'](model.config)
    active = list(range(x.shape[1]))
    h = x
    ctl.active = False
    try:
        for i, layer in enumerate(model.model.layers):
            desired = ctl.positions[i]
            if desired != active:
                lookup = {position: j for j, position in enumerate(active)}
                indices = torch.tensor([lookup[position] for position in desired], device=x.device)
                h = h.index_select(1, indices)
            idx = torch.tensor(desired, device=x.device)
            h = layer(h, position_embeddings=tuple((z.index_select(1, idx) for z in full_pe)), position_ids=idx[None], cache_position=idx, past_key_values=cache, pad_offset=torch.zeros(1, device=x.device, dtype=torch.long))
            active = desired
        h = compiled_rmsnorm(h, model.model.norm.weight, model.model.norm.variance_epsilon)
        return SimpleNamespace(logits=model.lm_head(h[:, -1:]), past_key_values=cache)
    finally:
        ctl.active = True

@torch.inference_mode()
def semantic_check(model, ctl, x, mask, out):
    checks = {}
    call = lambda: model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    if ctl.method in ('zoo', 'pdrop'):
        actual = call()
        expected = token_reference(model, ctl, x)
        checks['independent_token_loop'] = compare(actual, expected, ctl.lengths)
        del actual, expected
    else:
        rows = []
        handles = []
        for i, layer in enumerate(model.model.layers):
            if ctl.changed(i, 'attention_op'):

                def core_hook(module, inputs, kwargs, result, i=i):
                    expected = reference_attention(kwargs['q'], kwargs['k'], kwargs['v'], ctl.prefix, ctl.visual, ctl.method, ctl.config['rl_preceding_visual_window'])
                    rows.append(dict(layer=i, component='attention_op', **bounded_error(result, expected)))
                handles.append(layer.self_attn.attention_impl.register_forward_hook(core_hook, with_kwargs=True))
            if ctl.method == 'shortv' and ctl.changed(i, 'attention_module'):

                def module_hook(module, inputs, kwargs, result, i=i):
                    expected = module.original(inputs[0], position_embeddings=kwargs['position_embeddings'], past_key_values=None)
                    p, e = ctl.span
                    expected[:, p:e] = inputs[0][:, p:e]
                    assert torch.equal(result[:, p:e], inputs[0][:, p:e])
                    rows.append(dict(layer=i, component='attention_module', **bounded_error(result, expected)))
                handles.append(layer.self_attn.register_forward_hook(module_hook, with_kwargs=True))
            if ctl.changed(i, 'ffn_module'):

                def ffn_hook(module, inputs, result, i=i):
                    if ctl.method == 'shortv':
                        expected = module.original(inputs[0])
                        p, e = ctl.span
                        expected[:, p:e] = inputs[0][:, p:e]
                        assert torch.equal(result[:, p:e], inputs[0][:, p:e])
                    else:
                        saved = module.mode
                        module.mode = 'reference'
                        try:
                            expected = VisionMLP.forward(module, inputs[0])
                        finally:
                            module.mode = saved
                    rows.append(dict(layer=i, component='ffn_module', **bounded_error(result, expected)))
                handles.append(layer.mlp.register_forward_hook(ffn_hook))
        try:
            call()
        finally:
            for h in handles:
                h.remove()
        checks['local_modules'] = rows
    write_json(out / 'semantic_checks.json', checks)
    print('SEMANTICS_PASSED', ctl.method, flush=True)

@torch.inference_mode()
def graph_check(model, ctl, x, mask, out):
    call = lambda: model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = call()
        token = captured.logits[:, -1].argmax(-1)
    saved = x.clone()
    x.mul_(0.9).add_(0.01)
    expected = call()
    graph.replay()
    result = compare(captured, expected, ctl.lengths)
    assert torch.equal(token, expected.logits[:, -1].argmax(-1))
    x.copy_(saved)
    graph.replay()
    write_json(out / 'graph_check.json', result)
    del graph, captured, expected, token, saved
    gc.collect()
    print('GRAPH_CHECK_PASSED', ctl.method, flush=True)

def historical_measurement(reference, layer, component):
    if component == 'attention_op':
        row = reference['core']['rows'][layer]
        raw = row['raw_ms']['dense']
        return dict(ms=row['dense_ms'], mean_ms=statistics.mean(raw), raw_ms=raw, source='historical_dense', raw_granularity='five helper-window means; no per-batch raw recorded in original op data')
    row = reference['modules'][layer]
    measurement = dict(row['attn_dense' if component == 'attention_module' else 'ffn_dense'])
    measurement['source'] = 'historical_dense'
    return measurement

@torch.inference_mode()
def modules(model, ctl, x, mask, args, out, reference):
    attn, ffn, core, handles = ({}, {}, {}, [])
    for i, layer in enumerate(model.model.layers):
        if ctl.changed(i, 'attention_module'):

            def save_a(module, inputs, kwargs, i=i):
                attn[i] = (inputs[0].clone(), tuple((t.clone() for t in kwargs['position_embeddings'])))
            handles.append(layer.self_attn.register_forward_pre_hook(save_a, with_kwargs=True))
        if ctl.changed(i, 'ffn_module'):

            def save_f(module, inputs, i=i):
                ffn[i] = inputs[0].clone()
            handles.append(layer.mlp.register_forward_pre_hook(save_f))
        if ctl.changed(i, 'attention_op'):

            def save_c(module, inputs, kwargs, i=i):
                core[i] = tuple((kwargs[k].clone() for k in ('q', 'k', 'v')))
            handles.append(layer.self_attn.attention_impl.register_forward_pre_hook(save_c, with_kwargs=True))
    try:
        model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
    rows = []
    for i, layer in enumerate(model.model.layers):
        row = dict(layer=i, length=ctl.lengths[i], visual=ctl.counts[i])
        for component in ('attention_op', 'attention_module', 'ffn_module'):
            if not ctl.changed(i, component):
                row[component] = historical_measurement(reference, i, component)
                continue
            if component == 'attention_op':
                q, k, v = core.pop(i)
                fn = lambda: layer.self_attn.attention_impl(q=q, k=k, v=v)
            elif component == 'attention_module':
                ax, pe = attn.pop(i)
                fn = lambda: layer.self_attn(ax, position_embeddings=pe, past_key_values=None)
            else:
                fx = ffn.pop(i)
                fn = lambda: layer.mlp(fx)
            measurement = measure(fn, args)
            measurement['source'] = 'new_method_measurement'
            row[component] = measurement
            if component == 'attention_op':
                del q, k, v
            elif component == 'attention_module':
                del ax, pe
            else:
                del fx
        rows.append(row)
        write_json(out / 'modules.json', rows)
        print('MODULE', ctl.method, i, *(round(row[k]['ms'], 6) for k in ('attention_op', 'attention_module', 'ffn_module')), flush=True)
    gc.collect()
    return rows
