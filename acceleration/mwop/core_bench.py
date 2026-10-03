import argparse, functools, hashlib, itertools, json, random, statistics
from pathlib import Path
import torch, triton
from .timing import do_bench_cudagraph
from .config import load_plans, flags_tensor
from .token_schedule import layout, override_plans
from .attention import RegionAttention, reference
from .triton_prefill import attention
CONFIGS = list(itertools.product((32, 64, 128), (32, 64, 128), (4,), (2, 3)))
KERNELS = ('range_offband', 'range_interleave', 'range_masked')

def error_stats(actual, expected, atol=0.02, rtol=0.02, check=True):
    assert torch.isfinite(actual).all(), 'nonfinite output'
    diff = actual.float() - expected.float()
    stats = dict(max_abs=diff.abs().max().item(), rms=diff.square().mean().sqrt().item(), relative_rms=(diff.square().mean() / expected.float().square().mean().clamp_min(1e-20)).sqrt().item(), max_row_relative_l2=(diff.norm(dim=-1) / expected.float().norm(dim=-1).clamp_min(1e-06)).max().item())
    if check:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol, msg=str(stats))
    return stats

def dump(p, x):
    p.write_text(json.dumps(x, indent=2) + '\n')

def time_once(fn, rep):
    fn()
    torch.cuda.synchronize()
    return do_bench_cudagraph(fn, rep=rep)

@torch.inference_mode()
def run(a):
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    plans, hashes = load_plans(a.config_dir)
    plans = override_plans(plans, a.scenario)
    shapes = layout(a.scenario, a.prefix, a.visual, a.suffix)
    length = a.prefix + a.visual + a.suffix
    span = (a.prefix, a.prefix + a.visual)
    torch.manual_seed(17 if a.task == 'tune' else 29)
    manifest = dict(args=vars(a), config_hashes=hashes, length=length, span=span, configs=CONFIGS, kernels=list(KERNELS), torch=torch.__version__, cuda=torch.version.cuda, triton=triton.__version__, gpu=torch.cuda.get_device_name(), capability=torch.cuda.get_device_capability(), source_hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('mwop').glob('*.py')}, scope='Sequence-first attention op only, native GQA 28Q/4KV heads; FA2 dense baseline', causal='Only lower-triangular KV tiles traversed; diagonal/tail tiles use exact masks')
    dump(out / 'manifest.json', manifest)
    if a.task == 'smoke':
        checks = []
        for l, sp, kind in [(107, (18, 83), 0), (129, (0, 65), 27), (67, (0, 67), 'all'), (73, (18, 18), 'none'), (97, (0, 0), 'all')]:
            q = torch.randn(1, l, 28, 128, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(1, l, 4, 128, device='cuda', dtype=q.dtype)
            v = torch.randn_like(k)
            f = flags_tensor(plans[kind], 'cuda') if isinstance(kind, int) else torch.full((3, 28), kind == 'all', device='cuda', dtype=torch.bool)
            expected = reference(q, k, v, f, sp, fp32=True)
            for kernel in KERNELS:
                for cfg in [(32, 32, 4, 2), (64, 64, 4, 2), (128, 32, 4, 3)]:
                    plan = dict(kernel=kernel, config=cfg)
                    checks.append(dict(length=l, span=sp, kind=kind, plan=plan, error=error_stats(attention(q, k, v, f, sp, plan), expected)))
            print('SMOKE', l, sp, kind, flush=True)
        dump(out / 'core_correctness.json', checks)
    else:
        from ops.attention.base import DenseAttentionKernel
        q = torch.randn(1, length, 28, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(1, length, 4, 128, device='cuda', dtype=q.dtype)
        v = torch.randn_like(k)
        fa = DenseAttentionKernel()
        dense = functools.partial(fa, q=q, k=k, v=v)
        rng = random.Random(1701)
        rows = []
        selected = {}
        if a.task == 'measure':
            if not a.tiles:
                raise ValueError('--tiles is required for independent fixed-plan measurement')
            selected = json.loads(Path(a.tiles).read_text())
            assert set(selected) == {str(i) for i in range(28)}
            dump(out / 'tiles.json', selected)
            manifest['selection_sha256'] = hashlib.sha256(Path(a.tiles).read_bytes()).hexdigest()
            dump(out / 'manifest.json', manifest)
        for layer in range(28):
            length = shapes[layer]['length']
            span = tuple(shapes[layer]['span'])
            q = torch.randn(1, length, 28, 128, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(1, length, 4, 128, device='cuda', dtype=q.dtype)
            v = torch.randn_like(k)
            dense = functools.partial(fa, q=q, k=k, v=v)
            f = flags_tensor(plans[layer], 'cuda')
            expected = reference(q, k, v, f, span)
            if a.task == 'tune':
                candidates = list(itertools.product(KERNELS, CONFIGS))
                rng.shuffle(candidates)
                times = []
                for kernel, cfg in candidates:
                    plan = dict(kernel=kernel, config=cfg)
                    fn = functools.partial(attention, q, k, v, f, span, plan)
                    try:
                        err = error_stats(fn(), expected)
                        times.append(dict(**plan, ms=time_once(fn, a.rep), error=err))
                    except triton.OutOfResources as exc:
                        times.append(dict(**plan, rejected='OutOfResources', reason=str(exc)))
                best = min((r for r in times if 'ms' in r), key=lambda r: r['ms'])
                selected[str(layer)] = {k: best[k] for k in ('kernel', 'config')}
                rows.append(dict(layer=layer, candidates=times, best=best))
                dump(out / 'tuning.json', rows)
                dump(out / 'tiles.json', selected)
                print('SELECTED', layer, selected[str(layer)], best['ms'], flush=True)
            else:
                plan = selected[str(layer)]
                assert plan['kernel'] in KERNELS and tuple(plan['config']) in CONFIGS
                fn = functools.partial(attention, q, k, v, f, span, plan)
                err = error_stats(fn(), expected)
                flex = RegionAttention(f, span, length, backend='flex')
                flex.prepare_flex()
                flexfn = functools.partial(flex, q, k, v)
                ferr = error_stats(flexfn(), expected)
                fs = dict(dense=dense, triton=fn, flex=flexfn)
                raw = {k: [] for k in fs}
                for _ in range(a.samples):
                    names = list(fs)
                    rng.shuffle(names)
                    for n in names:
                        raw[n].append(time_once(fs[n], a.rep))
                ms = {n: statistics.median(vs) for n, vs in raw.items()}
                row = dict(layer=layer, length=length, span=list(span), dense_ms=ms['dense'], triton_ms=ms['triton'], flex_ms=ms['flex'], triton_speedup=ms['dense'] / ms['triton'], flex_speedup=ms['dense'] / ms['flex'], best_tile=plan['config'], best_kernel=plan['kernel'], raw_ms=raw, triton_error=err, flex_error=ferr)
                rows.append(row)
                dump(out / 'core.json', dict(rows=rows, config_hashes=hashes, length=length, span=span, args=vars(a), gpu=torch.cuda.get_device_name(), torch=torch.__version__, triton=triton.__version__, timing='Fixed preselected plans, independent seed and separate measurement process; randomized paired rounds; median of do_bench_cudagraph helper means'))
                print('CORE', layer, 'dense', ms['dense'], 'triton', ms['triton'], 'speedup', row['triton_speedup'], flush=True)
    dump(out / 'status.json', dict(status='complete', task=a.task))
    print('CORE_TASK_COMPLETE', a.task, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--scenario', choices=['base', 'pdrop', 'zoo', 'upper'], default='base')
    p.add_argument('--task', choices=['smoke', 'tune', 'measure'], required=True)
    p.add_argument('--config-dir', default='../../pruning_config')
    p.add_argument('--output', required=True)
    p.add_argument('--tiles')
    p.add_argument('--prefix', type=int, default=18)
    p.add_argument('--visual', type=int, default=3215)
    p.add_argument('--suffix', type=int, default=19)
    p.add_argument('--rep', type=int, default=100)
    p.add_argument('--samples', type=int, default=5)
    run(p.parse_args())
