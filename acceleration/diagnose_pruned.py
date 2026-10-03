import argparse, json
from pathlib import Path
import torch
from mwop.model import load_model
from mwop.config import load_plans
from mwop.core_bench import error_stats
from mwop.attention import reference

@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    for n in ('model-path', 'config-dir', 'tiles', 'output'):
        p.add_argument('--' + n, required=True)
    a = p.parse_args()
    out = Path(a.output)
    report = {}

    def save():
        out.write_text(json.dumps(report, indent=2) + '\n')
    torch.manual_seed(17)
    plans, _ = load_plans(a.config_dir)
    model = load_model(a.model_path, plans, 3252, (18, 3233), tiles=json.loads(Path(a.tiles).read_text()))
    ids = torch.randint(0, model.config.vocab_size, (1, 3252), device='cuda')
    x = model.model.embed_tokens(ids).detach()
    mask = torch.ones((1, 3252), device='cuda', dtype=torch.bool)

    def forward():
        return model(inputs_embeds=x, attention_mask=mask, use_cache=True, logits_to_keep=1)
    for mode in ('attention_only', 'both'):
        report[mode] = {'local_core': []}
        rows = report[mode]['local_core']
        handles = []
        model.configure(mode, backend='reference', reference_ffn=True)
        expected = forward()
        print(mode, 'REFERENCE_DONE', flush=True)
        model.configure(mode, backend='triton')

        def hook(i):

            def check(module, args, kwargs, result):
                q, k, v = [kwargs[n] for n in ('q', 'k', 'v')]
                ref = reference(q, k, v, module.flags, module.span)
                old = module.backend
                module.backend = 'flex'
                try:
                    if module.block is None:
                        module.prepare_flex()
                    flex = module.forward(q, k, v)
                finally:
                    module.backend = old
                row = dict(layer=i, triton_math=error_stats(result, ref, check=False), flex_math=error_stats(flex, ref, check=False), triton_flex=error_stats(result, flex, check=False))
                rows.append(row)
                save()
                print(mode, 'LOCAL', i, row, flush=True)
            return check
        for i, l in enumerate(model.model.layers):
            handles.append(l.self_attn.mwop_region.register_forward_hook(hook(i), with_kwargs=True))
        actual = forward()
        for h in handles:
            h.remove()
        report[mode]['triton_logits'] = error_stats(actual.logits, expected.logits, check=False)
        model.configure(mode, backend='flex')
        flex = forward()
        report[mode]['flex_logits'] = error_stats(flex.logits, expected.logits, check=False)
        report[mode]['triton_flex_logits'] = error_stats(actual.logits, flex.logits, check=False)
        report[mode]['flex_kv'] = []
        for arow, erow in zip(flex.past_key_values.layers, expected.past_key_values.layers):
            report[mode]['flex_kv'].append({n: error_stats(getattr(arow, n), getattr(erow, n), check=False) for n in ('keys', 'values')})
        save()
        print(mode, 'GLOBAL', {k: v for k, v in report[mode].items() if k.endswith('logits')}, flush=True)
        del actual, expected, flex
    print('DIAG_COMPLETE', flush=True)
if __name__ == '__main__':
    main()
