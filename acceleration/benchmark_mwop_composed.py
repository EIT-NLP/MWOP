import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import torch
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
from benchmark_methods import semantic_check, graph_check, modules
from benchmark_methods import bounded_error
from benchmark_mwop import measure, write_json
from benchmark_combined_check import correctness
from mwop.config import load_plans
from mwop.model import load_model
from mwop.methods import MethodController
from mwop.method_theory import compute

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

@torch.inference_mode()
def mwop_local_checks(model, x, mask, out):
    rows, handles = ([], [])
    for i, layer in enumerate(model.model.layers):

        def attention_hook(module, inputs, kwargs, actual, i=i):
            saved = module.mwop_region.backend
            try:
                module.mwop_region.backend = 'reference'
                expected = module.forward(inputs[0], position_embeddings=kwargs['position_embeddings'], past_key_values=None)
            finally:
                module.mwop_region.backend = saved
            rows.append(dict(layer=i, component='attention_module', **bounded_error(actual, expected)))

        def ffn_hook(module, inputs, actual, i=i):
            saved = module.mode
            try:
                module.mode = 'reference'
                expected = module.forward(inputs[0])
            finally:
                module.mode = saved
            rows.append(dict(layer=i, component='ffn_module', **bounded_error(actual, expected)))
        handles.append(layer.self_attn.register_forward_hook(attention_hook, with_kwargs=True))
        handles.append(layer.mlp.register_forward_hook(ffn_hook))
    try:
        model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    finally:
        for h in handles:
            h.remove()
        write_json(out / 'local_module_correctness.json', rows)
    assert len(rows) == 56

def theory(config, method, model, plans):
    token_method = {'mwop_p': 'pdrop', 'mwop_z': 'zoo'}[method]
    result = compute(config, token_method, model.config)
    result['method'] = method
    p, s = (config['prefix'], config['suffix'])
    d, inner, hd, heads = (3584, 18944, 128, 28)
    for i, row in enumerate(result['layers']):
        v = row['visual_tokens']
        n = p + v + s
        removed = sum(plans[i]['flags'][0]) * v * (v + 1) // 2 + sum(plans[i]['flags'][1]) * s * v + sum(plans[i]['flags'][2]) * s * (s + 1) // 2
        pairs = heads * n * (n + 1) // 2 - removed
        f = model.model.layers[i].mlp
        width = inner if f.kept == inner else f.width
        row.update(vision_ffn_width=width, vision_ffn_kept=f.kept, attention_op_TFLOP=4 * hd * pairs / 1000000000000.0, ffn_module_TFLOP=6 * d * ((p + s) * inner + v * width) / 1000000000000.0)
        row['attention_module_TFLOP'] = row['qkvo_TFLOP'] + row['attention_op_TFLOP']
    for key in ('attention_op_TFLOP', 'attention_module_TFLOP', 'ffn_module_TFLOP', 'qkvo_TFLOP'):
        result['totals'][key] = sum((r[key] for r in result['layers']))
    t = result['totals']
    t['model_TFLOP'] = t['attention_module_TFLOP'] + t['ffn_module_TFLOP'] + t['last_token_lm_head_TFLOP']
    result['accounting'] += f" Current {config['model']} MWOP masks plus token pruning; actual packed visual FFN widths with configured alignment."
    return result

@torch.inference_mode()
def main(args):
    root = Path(__file__).resolve().parent
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(Path(args.config).read_text())
    p, v, s = (cfg[k] for k in ('prefix', 'visual', 'suffix'))
    n = p + v + s
    expected = {'llava_onevision': dict(shape=(3252, 37, 3215), zoo=1607, pdrop=[1747, 950, 516])}[cfg['model']]
    assert (n, p + s, v) == expected['shape']
    assert cfg['zoo_keep_visual'] == expected['zoo'] and cfg['pdrop_visual'] == expected['pdrop']
    torch.manual_seed(17)
    tiles = json.loads(Path(args.tiles).read_text()) if args.tiles else None
    plans, pruning_hashes = load_plans(args.config_dir)
    model = load_model(args.model_path, plans, n, (p, p + v), tiles=tiles, alignment=args.alignment)
    torch.manual_seed(17)
    ids = torch.randint(0, model.config.vocab_size, (1, n), device='cuda')
    x = model.model.embed_tokens(ids).detach()
    mask = torch.ones((1, n), device='cuda', dtype=torch.bool)
    sources = sorted(set([*root.glob('benchmark*.py'), *root.glob('mwop/*.py'), *root.glob('ops/**/*.py'), *root.glob('wrapper/**/*.py')]))
    write_json(out / 'manifest.json', dict(start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), args=vars(args), method=args.method, worker=args.worker, config=cfg, config_sha256=sha(args.config), tiles_sha256=sha(args.tiles) if args.tiles else None, pruning_hashes=pruning_hashes, source_hashes={str(f.relative_to(root)): sha(f) for f in sources}, checkpoint_hashes={name: sha(Path(args.model_path) / name) for name in ('config.json', 'model.safetensors.index.json')}, input_shape=list(x.shape), input_ids_sha256=sha_bytes(ids.cpu().numpy().tobytes()), gpu=torch.cuda.get_device_name(), hostname=subprocess.check_output(['hostname'], text=True).strip(), job_id=os.environ.get('SLURM_JOB_ID'), torch=torch.__version__, cuda=torch.version.cuda, emulate_precision_casts=os.environ.get('TORCHINDUCTOR_EMULATE_PRECISION_CASTS', '0'), bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction, packages={name: importlib.metadata.version(name) for name in ('torch', 'triton', 'transformers', 'flash-attn', 'einops', 'nvtx', 'safetensors', 'torchvision', 'pillow')}, scope='Decoder prefill, fresh KV, final norm, final-token LM head and argmax; CUDA Graph. No vision frontend or online selectors.', input_kind='Real checkpoint, synthetic embedding-table input, seed17; native model rotary positions.', timing_source='Every component and E2E freshly measured. No historical timing reuse.', mwop_tiles=('Explicit fixed tile plan' if args.tiles else 'Original default fixed tile plan') + f'; FFN alignment={args.alignment}. No change to masks or math.'))
    print('MODEL_READY', args.method, n, flush=True)
    token_method = {'mwop_p': 'pdrop', 'mwop_z': 'zoo'}[args.method]
    ctl = MethodController(model, cfg)
    ctl.configure(token_method)
    for layer, length, count in zip(model.model.layers, ctl.lengths, ctl.counts):
        layer.self_attn.mwop_region.length = length
        layer.self_attn.mwop_region.span = (p, p + count)
        layer.self_attn.mwop_region.block = None
        layer.mlp.span = (p, p + count)
    model._mwop_layer_lengths = list(ctl.lengths)
    model.configure('both', 'triton')
    semantic_check(model, ctl, x, mask, out)
    correctness(model, x, mask, out, 'triton')
    model.configure('both', 'triton')
    mwop_local_checks(model, x, mask, out)
    write_json(out / 'schedule.json', dict(lengths=ctl.lengths, visual_counts=ctl.counts))
    write_json(out / 'theory.json', theory(cfg, args.method, model, plans))
    graph_check(model, ctl, x, mask, out)
    print('ALL_CHECKS_PASSED', args.method, flush=True)
    call = lambda: model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1).logits[:, -1].argmax(-1)

    def warmup():
        deadline = time.monotonic() + args.warmup_seconds
        while time.monotonic() < deadline:
            call()
            torch.cuda.synchronize()
    warmup()
    collection = SimpleNamespace(method=args.method, lengths=ctl.lengths, counts=ctl.counts, changed=lambda layer, component: True)
    modules(model, collection, x, mask, args, out, reference=None)
    warmup()
    write_json(out / 'ttft.json', measure(call, args))
    write_json(out / 'status.json', dict(status='complete', method=args.method, end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print('FRESH_MEASUREMENT_COMPLETE', args.method, flush=True)

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()
if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--method', required=True, choices=['mwop_p', 'mwop_z'])
    for name in ('model-path', 'config', 'config-dir', 'output', 'worker'):
        cli.add_argument('--' + name, required=True)
    cli.add_argument('--tiles')
    cli.add_argument('--alignment', type=int, required=True)
    cli.add_argument('--rep', type=int, default=100)
    cli.add_argument('--samples', type=int, default=5)
    cli.add_argument('--warmup-seconds', type=float, default=20)
    a = cli.parse_args()
    try:
        main(a)
    except Exception as exc:
        p = Path(a.output)
        p.mkdir(parents=True, exist_ok=True)
        write_json(p / 'error.json', dict(error=repr(exc)))
        raise
