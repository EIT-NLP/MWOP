import argparse
import gc
import hashlib
import importlib.metadata
import json
import subprocess
import statistics
import mwop.timing as timing
from pathlib import Path
import torch
from mwop.timing import do_bench_cudagraph
from mwop.config import load_plans
from mwop.core_bench import error_stats
from mwop.model import load_model
from mwop.token_schedule import install, override_plans

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def measure(fn, args):
    fn()
    torch.cuda.synchronize()
    raw, windows = ([], [])
    for _ in range(args.samples):
        raw.append(do_bench_cudagraph(fn, rep=args.rep))
        windows.append(dict(timing.LAST_SAMPLE))
    return dict(ms=statistics.median(raw), mean_ms=statistics.mean(raw), std_ms=statistics.stdev(raw) if len(raw) > 1 else 0, raw_ms=raw, timing_windows=windows)

def kv_tensors(cache):
    return [(x.keys, x.values) for x in cache.layers]

@torch.inference_mode()
def correctness(model, x, mask, output, backend):
    from mwop.attention import reference

    def compare(actual, expected):
        return dict(logits=error_stats(actual.logits, expected.logits, check=False), kv=[dict(layer=i, k=error_stats(ak, ek, check=False), v=error_stats(av, ev, check=False)) for i, ((ak, av), (ek, ev)) in enumerate(zip(kv_tensors(actual.past_key_values), kv_tensors(expected.past_key_values)))])

    def within(errors, calibration):
        if errors['logits']['relative_rms'] >= max(0.03, 3 * calibration['logits']['relative_rms']):
            return False
        return all((row[key]['relative_rms'] < max(0.05, 3 * calibration['kv'][i][key]['relative_rms']) for i, row in enumerate(errors['kv']) for key in ('k', 'v')))
    saved_flags = [layer.self_attn.mwop_region.flags.clone() for layer in model.model.layers]
    model.configure('dense')
    dense = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    try:
        for layer in model.model.layers:
            layer.self_attn.mwop_region.flags.zero_()
        model.configure('attention_only', backend='reference')
        dense_ref = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        baseline = compare(dense, dense_ref)
    finally:
        for layer, flags in zip(model.model.layers, saved_flags):
            layer.self_attn.mwop_region.flags.copy_(flags)
    write_json(output / 'dense_numerical_baseline.json', baseline)
    print('DENSE_NUMERICS', baseline['logits'], flush=True)
    del dense, dense_ref, saved_flags
    checks = []
    for mode in ('attention_only', 'ffn_only', 'both'):
        model.configure(mode, backend='reference', reference_ffn=True)
        expected = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        model.configure(mode, backend=backend)
        local, handles = ([], [])

        def core_hook(i):

            def check(module, inputs, kwargs, result):
                ref = reference(kwargs['q'], kwargs['k'], kwargs['v'], module.flags, module.span)
                local.append(dict(layer=i, **error_stats(result, ref, check=False)))
            return check
        if mode in ('attention_only', 'both'):
            for i, layer in enumerate(model.model.layers):
                handles.append(layer.self_attn.mwop_region.register_forward_hook(core_hook(i), with_kwargs=True))
        try:
            actual = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
        finally:
            for handle in handles:
                handle.remove()
        row = dict(mode=mode, **compare(actual, expected), local_core=local, first_token_match=bool(torch.equal(actual.logits.argmax(-1), expected.logits.argmax(-1))))
        checks.append(row)
        write_json(output / 'model_correctness.json', checks)
        for err in local:
            assert err['relative_rms'] < 0.01 and err['max_row_relative_l2'] < 0.02, (mode, err)
        passed = within(row, baseline)
        if not passed and mode in ('attention_only', 'both') and (backend == 'triton'):
            model.configure(mode, backend='flex', reference_ffn=True)
            calibrated = model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
            row['same_mask_flex_baseline'] = compare(calibrated, expected)
            row['triton_flex'] = compare(actual, calibrated)
            passed = within(row, row['same_mask_flex_baseline']) and within(row['triton_flex'], baseline)
            del calibrated
        row['passed'] = passed
        write_json(output / 'model_correctness.json', checks)
        print('MODEL_CHECK', mode, row['logits'], 'same_mask_calibration', 'same_mask_flex_baseline' in row, flush=True)
        del expected, actual
        model.configure(mode, backend=backend)
        assert passed, (mode, row['logits'], 'See model_correctness.json for calibration and local checks')
    return checks

@torch.inference_mode()
def modules(model, x, mask, args, output):
    attn_inputs, ffn_inputs, handles = ({}, {}, [])
    model.configure('dense')
    for i, layer in enumerate(model.model.layers):

        def save_attn(module, inputs, kwargs, i=i):
            attn_inputs[i] = (inputs[0].clone(), kwargs['position_embeddings'])

        def save_ffn(module, inputs, i=i):
            ffn_inputs[i] = inputs[0].clone()
        handles.append(layer.self_attn.register_forward_pre_hook(save_attn, with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(save_ffn))
    model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    for handle in handles:
        handle.remove()
    rows = []
    for i, layer in enumerate(model.model.layers):
        ax, pe = attn_inputs[i]
        fx = ffn_inputs[i]
        a, f = (layer.self_attn, layer.mlp)
        acall = lambda: a(ax, position_embeddings=pe, past_key_values=None)
        fcall = lambda: f(fx)
        a.attention_impl, f.mode = (a.mwop_dense, 'dense')
        ad = measure(acall, args)
        fd = measure(fcall, args)
        a.mwop_region.backend = 'reference'
        a.attention_impl, f.mode = (a.mwop_region, 'reference')
        ar, fr = (acall(), fcall())
        a.mwop_region.backend = args.backend
        if args.backend == 'flex' and a.mwop_region.block is None:
            a.mwop_region.prepare_flex()
        f.mode = 'packed'
        astats = error_stats(acall(), ar, check=False)
        fstats = error_stats(fcall(), fr, check=False)
        for name, stats in [('attention', astats), ('ffn', fstats)]:
            assert stats['relative_rms'] < 0.01 and stats['max_row_relative_l2'] < 0.02, (i, name, stats)
        ap, fp = (measure(acall, args), measure(fcall, args))
        row = dict(layer=i, length=ax.shape[1], visual=a.mwop_region.span[1] - a.mwop_region.span[0], vision_kept=f.kept, vision_width=f.width, attn_dense=ad, attn_mwop=ap, attn_speedup=ad['ms'] / ap['ms'], ffn_dense=fd, ffn_mwop=fp, ffn_speedup=fd['ms'] / fp['ms'], attn_error=astats, ffn_error=fstats)
        rows.append(row)
        write_json(output / 'modules.json', rows)
        import csv
        with (output / 'layerwise.csv').open('w') as file:
            writer = csv.writer(file)
            writer.writerow(['layer', 'vision_kept', 'vision_width', 'attn_dense_ms', 'attn_mwop_ms', 'attn_speedup', 'ffn_dense_ms', 'ffn_mwop_ms', 'ffn_speedup'])
            for r in rows:
                writer.writerow([r['layer'], r['vision_kept'], r['vision_width'], r['attn_dense']['ms'], r['attn_mwop']['ms'], r['attn_speedup'], r['ffn_dense']['ms'], r['ffn_mwop']['ms'], r['ffn_speedup']])
        print('MODULE', i, 'attn', row['attn_speedup'], 'ffn', row['ffn_speedup'], flush=True)
        del ar, fr
    return rows

@torch.inference_mode()
def ttft(model, x, mask, args, output):
    results = {}
    for mode in ('dense', 'attention_only', 'ffn_only', 'both'):
        model.configure(mode, args.backend)

        def forward():
            return model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)

        def timed_forward():
            return forward().logits[:, -1].argmax(-1)
        for _ in range(3):
            forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                captured = forward()
                captured_token = captured.logits[:, -1].argmax(-1)
        torch.cuda.current_stream().wait_stream(stream)
        original = x.clone()
        x.copy_(original * 0.9 + 0.01)
        expected = forward()
        graph.replay()
        graph_check = dict(logits=error_stats(captured.logits, expected.logits, atol=0.01, rtol=0.01))
        for layer_id, ((ak, av), (ek, ev)) in enumerate(zip(kv_tensors(captured.past_key_values), kv_tensors(expected.past_key_values))):
            error_stats(ak, ek, atol=0.01, rtol=0.01)
            error_stats(av, ev, atol=0.01, rtol=0.01)
            assert ak.shape[1] == model._mwop_layer_lengths[layer_id]
        assert torch.equal(captured_token, expected.logits[:, -1].argmax(-1))
        x.copy_(original)
        graph.replay()
        del graph, captured, captured_token, expected, original
        gc.collect()
        measurement = measure(timed_forward, args)
        measurement['graph_check'] = graph_check
        results[mode] = measurement
        measurement['speedup'] = results['dense']['ms'] / measurement['ms']
        write_json(output / 'ttft.json', results)
        print('TTFT', mode, measurement['ms'], 'speedup', measurement['speedup'], flush=True)
    return results

@torch.inference_mode()
def main(args):
    torch.manual_seed(17)
    plans, hashes = load_plans(args.config_dir)
    plans = override_plans(plans, args.scenario)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tiles = json.loads(Path(args.tiles).read_text()) if args.tiles else None
    length, span = (args.prefix + args.visual + args.suffix, (args.prefix, args.prefix + args.visual))
    manifest = dict(args=vars(args), config_hashes=hashes, length=length, span=span, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), torch=torch.__version__, cuda=torch.version.cuda, packages={x: importlib.metadata.version(x) for x in ('torch', 'triton', 'transformers', 'flash-attn', 'einops', 'nvtx', 'safetensors', 'torchvision', 'pillow')}, git_head=json.loads(Path('provenance.json').read_text())['upstream_commit'], purpose='performance_benchmark', dtype='bfloat16', model_family='LLaVA-OneVision-Qwen2-7B', position_convention='OV native one-dimensional rotary positions for synthetic inputs; no image grid supplied', module_scope='includes default norm/residual/projections; no KV cache allocation', ttft_scope='decoder prefill + fresh KV + final norm + last-token LM head + argmax; CUDA Graph; excludes vision frontend', timing='Triton do_bench_cudagraph mean per independent window; median across windows; warm L2', source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path('mwop').glob('*.py')) + [Path('benchmark_mwop.py'), Path('wrapper/static/dense.py'), Path('wrapper/base.py'), Path('ops/compiled_ops.py'), Path('ops/attention/base.py'), Path('ops/__init__.py'), Path('wrapper/static/__init__.py'), Path('provenance.json')]}, input_kind='synthetic embedding-table inputs, including visual positions; seed 17', model_files={f.name: dict(size_bytes=f.stat().st_size, **{'sha256': hashlib.sha256(f.read_bytes()).hexdigest()} if f.suffix == '.json' else {}) for f in Path(args.model_path).iterdir() if f.suffix in ('.json', '.safetensors')}, gpu_state=subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,pstate,power.limit,clocks.sm,clocks.mem', '--format=csv'], text=True).strip())
    write_json(output / 'manifest.json', manifest)
    model = load_model(args.model_path, plans, length, span, tiles=tiles, alignment=args.alignment)
    ids = torch.randint(0, model.config.vocab_size, (1, length), device='cuda')
    x = model.model.embed_tokens(ids).detach()
    mask = torch.ones((1, length), device='cuda', dtype=torch.bool)
    print('MODEL_READY', length, sum((p.numel() for p in model.parameters())), flush=True)
    shapes = install(model, args.scenario, args.prefix, args.visual, args.suffix)
    write_json(output / 'token_schedule.json', shapes)
    if args.task in ('check', 'ttft', 'all'):
        correctness(model, x, mask, output, args.backend)
    if args.task in ('modules', 'all'):
        modules(model, x, mask, args, output)
    if args.task in ('ttft', 'all'):
        ttft(model, x, mask, args, output)
    write_json(output / 'status.json', dict(status='complete', task=args.task, backend=args.backend))
    print('BENCHMARK_COMPLETE', flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', choices=('base', 'pdrop', 'zoo', 'upper'), default='base')
    p.add_argument('--model-path', required=True)
    p.add_argument('--config-dir', default=str(Path(__file__).resolve().parent / 'pruning_config/llava_onevision'))
    p.add_argument('--task', choices=('check', 'modules', 'ttft', 'all'), default='all')
    p.add_argument('--backend', choices=('triton', 'flex'), default='triton')
    p.add_argument('--tiles')
    p.add_argument('--alignment', type=int, default=64)
    p.add_argument('--prefix', type=int, default=18)
    p.add_argument('--visual', type=int, default=3215)
    p.add_argument('--suffix', type=int, default=19)
    p.add_argument('--rep', type=int, default=100)
    p.add_argument('--samples', type=int, default=5)
    p.add_argument('--output', default='mwop_results/prefill')
    args = p.parse_args()
    try:
        main(args)
    except Exception as exc:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / 'error.json', dict(status='error', error=repr(exc)))
        raise
