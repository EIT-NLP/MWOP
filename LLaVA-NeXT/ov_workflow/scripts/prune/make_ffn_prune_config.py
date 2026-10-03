from __future__ import annotations
import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Tuple
import torch
_NAME_RE = re.compile('(?:^|\\.)layers\\.(\\d+)\\.mlp\\.(gate|up|down)_proj\\.weight$')

def _iter_mlp_tensors(ckpt: str):
    index = os.path.join(ckpt, 'model.safetensors.index.json')
    single = os.path.join(ckpt, 'model.safetensors')
    shards = sorted(glob.glob(os.path.join(ckpt, '*.safetensors')))
    try:
        from safetensors import safe_open
        if os.path.isfile(index):
            with open(index, 'r', encoding='utf-8') as f:
                weight_map = json.load(f)['weight_map']
            by_shard: Dict[str, list] = defaultdict(list)
            for name, shard in weight_map.items():
                if _NAME_RE.search(name):
                    by_shard[shard].append(name)
            for shard, names in by_shard.items():
                with safe_open(os.path.join(ckpt, shard), framework='pt', device='cpu') as h:
                    for name in names:
                        m = _NAME_RE.search(name)
                        yield (int(m.group(1)), m.group(2), h.get_tensor(name))
            return
        if shards:
            for shard in shards:
                with safe_open(shard, framework='pt', device='cpu') as h:
                    for name in h.keys():
                        m = _NAME_RE.search(name)
                        if m:
                            yield (int(m.group(1)), m.group(2), h.get_tensor(name))
            return
    except ImportError:
        pass
    bins = sorted(glob.glob(os.path.join(ckpt, 'pytorch_model*.bin'))) or sorted(glob.glob(os.path.join(ckpt, '*.bin')))
    if not bins:
        raise FileNotFoundError(f'no .safetensors or .bin weights found in {ckpt}; point --ckpt at the merged HF model directory.')
    for b in bins:
        sd = torch.load(b, map_location='cpu', weights_only=True)
        for name, t in sd.items():
            m = _NAME_RE.search(name)
            if m:
                yield (int(m.group(1)), m.group(2), t)
        del sd

def _load_activation_scores(act_scores: str) -> Dict[int, torch.Tensor]:
    if not act_scores:
        raise ValueError('metric=activation requires --act_scores <file.pt> (or env FFN_ACT_SCORES); produce it with scripts/prune/collect_ffn_activation.py')
    if not os.path.isfile(act_scores):
        raise FileNotFoundError(f'activation scores file not found: {act_scores}')
    obj = torch.load(act_scores, map_location='cpu')
    raw = obj.get('scores', obj) if isinstance(obj, dict) else obj
    scores: Dict[int, torch.Tensor] = {}
    for k, v in raw.items():
        scores[int(k)] = torch.as_tensor(v).float().flatten()
    if not scores:
        raise RuntimeError(f'no per-layer scores found in {act_scores}')
    return scores

def _load_invalid_layers(act_scores: str) -> List[int]:
    if not act_scores or not os.path.isfile(act_scores):
        return []
    obj = torch.load(act_scores, map_location='cpu')
    if not isinstance(obj, dict):
        return []
    raw = obj.get('invalid_all_zero_layers', [])
    if not isinstance(raw, (list, tuple)):
        raise ValueError('invalid_all_zero_layers must be a list of layer indices')
    return sorted({int(layer) for layer in raw})

def _neuron_scores(ckpt: str, metric: str, seed: int=0, act_scores: str='') -> Dict[int, torch.Tensor]:
    if metric in ('activation', 'taylor'):
        return _load_activation_scores(act_scores)
    if metric == 'wanda':
        act = _load_activation_scores(act_scores)
        dn: Dict[int, torch.Tensor] = {}
        for layer, which, w in _iter_mlp_tensors(ckpt):
            if which == 'down':
                dn[layer] = w.float().norm(dim=0)
        out: Dict[int, torch.Tensor] = {}
        for layer, a in act.items():
            if layer not in dn:
                raise RuntimeError(f'wanda: layer {layer} has activation but no down_proj in ckpt')
            out[layer] = dn[layer] * a
        return out
    if metric == 'random':
        scores: Dict[int, torch.Tensor] = {}
        if act_scores:
            template = _load_activation_scores(act_scores)
            for layer, values in template.items():
                g = torch.Generator().manual_seed(int(seed) + int(layer))
                scores[layer] = torch.rand(int(values.numel()), generator=g)
            return scores
        for layer, which, w in _iter_mlp_tensors(ckpt):
            if which != 'gate':
                continue
            inter = int(w.shape[0])
            g = torch.Generator().manual_seed(int(seed) + int(layer))
            scores[layer] = torch.rand(inter, generator=g)
        if not scores:
            raise RuntimeError('found no mlp weights; check the checkpoint path/format.')
        return scores
    norms: Dict[int, Dict[str, torch.Tensor]] = defaultdict(dict)
    for layer, which, w in _iter_mlp_tensors(ckpt):
        w = w.float()
        if which in ('gate', 'up'):
            norms[layer][which] = w.norm(dim=1)
        else:
            norms[layer]['down'] = w.norm(dim=0)
    if not norms:
        raise RuntimeError('found no mlp weights; check the checkpoint path/format.')
    scores: Dict[int, torch.Tensor] = {}
    for layer, d in norms.items():
        if not {'gate', 'up', 'down'}.issubset(d):
            raise RuntimeError(f'layer {layer} missing one of gate/up/down ({sorted(d)})')
        if metric == 'down':
            s = d['down']
        elif metric == 'sum':
            s = d['gate'] + d['up'] + d['down']
        else:
            s = d['gate'] * d['up'] * d['down']
        scores[layer] = s
    return scores

def build_config(ckpt: str, ratio: float, metric: str, align: int, seed: int=0, act_scores: str='', select: str='low', alloc: str='layer', token_scope: str='all', exclude_layers: Iterable[int]=(), force_prune_invalid_layers: bool=False) -> dict:
    if not 0.0 <= float(ratio) <= 1.0:
        raise ValueError(f'ratio must be in [0,1], got {ratio}')
    if token_scope not in ('all', 'vision', 'text'):
        raise ValueError(f'token_scope must be all|vision|text, got {token_scope!r}')
    scores = _neuron_scores(ckpt, metric, seed, act_scores)
    layers_sorted = sorted(scores)
    inter = int(scores[layers_sorted[0]].numel())
    for layer in layers_sorted:
        if scores[layer].numel() != inter:
            raise RuntimeError(f'layer {layer} intermediate {scores[layer].numel()} != {inter}')
    largest = select == 'high'
    neurons: Dict[str, list] = {}
    explicit_excluded = {int(layer) for layer in exclude_layers}
    ranking_invalid: set[int] = set()
    if metric in ('activation', 'taylor', 'wanda') or (metric == 'random' and act_scores):
        ranking_invalid.update(_load_invalid_layers(act_scores))
    if force_prune_invalid_layers:
        if alloc != 'global' or select != 'low':
            raise ValueError('force_prune_invalid_layers requires alloc=global and select=low')
        if not ranking_invalid:
            raise ValueError('force_prune_invalid_layers requested but ranking has no invalid_all_zero_layers')
        overlap = explicit_excluded & ranking_invalid
        if overlap:
            raise ValueError(f'layers cannot be both excluded and force-pruned: {sorted(overlap)}')
        forced_pruned = set(ranking_invalid)
        excluded = set(explicit_excluded)
    else:
        forced_pruned = set()
        excluded = explicit_excluded | ranking_invalid
    unknown_excluded = sorted(excluded - set(layers_sorted))
    if unknown_excluded:
        raise ValueError(f'excluded layers not present in scores: {unknown_excluded}')
    unknown_forced = sorted(forced_pruned - set(layers_sorted))
    if unknown_forced:
        raise ValueError(f'force-pruned layers not present in scores: {unknown_forced}')
    candidate_layers = [layer for layer in layers_sorted if layer not in excluded]
    if not candidate_layers:
        raise ValueError('no candidate layers remain after exclusions')
    if alloc == 'global':
        ranked_candidate_layers = [layer for layer in candidate_layers if layer not in forced_pruned]
        flat = torch.cat([scores[l] for l in ranked_candidate_layers])
        candidate_total = inter * len(candidate_layers)
        n_remove_total = int(round(ratio * candidate_total))
        forced_total = inter * len(forced_pruned)
        if n_remove_total < forced_total:
            raise ValueError(f'ratio={ratio} targets only {n_remove_total} neurons, fewer than the {forced_total} neurons in force-pruned layers {sorted(forced_pruned)}')
        ranked_remove_total = n_remove_total - forced_total
        if ranked_remove_total > int(flat.numel()):
            raise ValueError('requested removal exceeds ranked candidate pool')
        for l in layers_sorted:
            neurons[str(l)] = []
        for l in forced_pruned:
            neurons[str(l)] = list(range(inter))
        if ranked_remove_total > 0:
            dead_flat = torch.topk(flat, ranked_remove_total, largest=largest).indices
            for pos, loc in zip((dead_flat // inter).tolist(), (dead_flat % inter).tolist()):
                neurons[str(ranked_candidate_layers[pos])].append(loc)
            for k in neurons:
                neurons[k] = sorted(neurons[k])
        counts = [len(neurons[str(l)]) for l in layers_sorted]
        return {'ablation_type': 'ffn_prune', 'metric': metric, 'select': select, 'alloc': 'global', 'seed': seed, 'token_scope': token_scope, 'act_scores': act_scores if metric in ('activation', 'taylor', 'wanda') else '', 'score_template': act_scores if metric == 'random' and act_scores else '', 'ratio': ratio, 'align': 1, 'per_layer': True, 'intermediate_size': inter, 'candidate_total': int(candidate_total), 'candidate_layers': candidate_layers, 'ranked_candidate_layers': ranked_candidate_layers, 'excluded_layers': sorted(excluded), 'ranking_invalid_layers': sorted(ranking_invalid), 'forced_pruned_layers': sorted(forced_pruned), 'forced_pruned_total': int(forced_total), 'removed_total': int(sum(counts)), 'removed_per_layer_min': int(min(counts)), 'removed_per_layer_max': int(max(counts)), 'removed_counts': {str(layers_sorted[i]): counts[i] for i in range(len(counts))}, 'layers': len(neurons), 'neurons': neurons}
    n_remove = int(round(ratio * inter))
    if align > 1:
        kept = inter - n_remove
        kept = (kept + align - 1) // align * align
        kept = min(kept, inter)
        n_remove = inter - kept
    for layer in layers_sorted:
        if layer in excluded:
            neurons[str(layer)] = []
            continue
        s = scores[layer]
        if n_remove > 0:
            dead = torch.topk(s, n_remove, largest=largest).indices.sort().values.tolist()
        else:
            dead = []
        neurons[str(layer)] = dead
    return {'ablation_type': 'ffn_prune', 'metric': metric, 'select': select, 'alloc': 'layer', 'token_scope': token_scope, 'seed': seed, 'act_scores': act_scores if metric in ('activation', 'taylor', 'wanda') else '', 'score_template': act_scores if metric == 'random' and act_scores else '', 'ratio': ratio, 'align': align, 'per_layer': True, 'intermediate_size': inter, 'removed_per_layer': n_remove, 'removed_total': int(n_remove * len(candidate_layers)), 'candidate_total': int(inter * len(candidate_layers)), 'candidate_layers': candidate_layers, 'excluded_layers': sorted(excluded), 'layers': len(neurons), 'neurons': neurons}

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True, help='merged HF model dir (safetensors or .bin)')
    ap.add_argument('--ratio', type=float, default=0.3, help='fraction of intermediate neurons to prune per layer')
    ap.add_argument('--metric', default='prod', choices=['prod', 'sum', 'down', 'random', 'activation', 'taylor', 'wanda'])
    ap.add_argument('--seed', type=int, default=0, help='random seed (only used by --metric random)')
    ap.add_argument('--act_scores', default=os.environ.get('FFN_ACT_SCORES', ''), help='path to collect_ffn_activation.py output (required for --metric activation; optional shape/invalid-layer template for --metric random; defaults to env FFN_ACT_SCORES so existing sweep scripts work unchanged)')
    ap.add_argument('--select', default=os.environ.get('FFN_SELECT', 'low'), choices=['low', 'high'], help="which end to PRUNE: 'low' (default, drop least-important) or 'high' (drop most-important -- control to test if the metric is anti-correlated; env FFN_SELECT fallback so sweep scripts can flip it)")
    ap.add_argument('--alloc', default=os.environ.get('FFN_ALLOC', 'layer'), choices=['layer', 'global'], help="'layer' (default): same count pruned per layer (uniform ratio). 'global': rank all neurons across all layers, prune the globally lowest fraction -> per-layer counts DIFFER (adaptive, like per-layer head pruning). env FFN_ALLOC fallback so sweep scripts can flip it.")
    ap.add_argument('--token_scope', default=os.environ.get('FFN_TOKEN_SCOPE', 'all'), choices=['all', 'vision', 'text'], help='token positions where selected neurons are zeroed; all preserves legacy behaviour')
    ap.add_argument('--exclude_layers', default=os.environ.get('FFN_EXCLUDE_LAYERS', ''), help='comma-separated layers excluded from pruning candidates; ranking-file invalid_all_zero_layers are also excluded automatically unless --force_prune_invalid_layers is set')
    ap.add_argument('--force_prune_invalid_layers', action='store_true', help='with global low-importance selection, count ranking-file invalid_all_zero_layers in the total ratio and prune all of them first')
    ap.add_argument('--align', type=int, default=1, help='round kept count up to a multiple of this (1 = exact ratio; layer-alloc only)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    excluded = [int(item) for item in args.exclude_layers.split(',') if item.strip()]
    cfg = build_config(args.ckpt, args.ratio, args.metric, args.align, args.seed, args.act_scores, args.select, args.alloc, args.token_scope, excluded, args.force_prune_invalid_layers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(cfg, f)
    if cfg.get('alloc') == 'global':
        print(f"[make_ffn_prune] metric={cfg['metric']} alloc=global scope={cfg['token_scope']} select={cfg['select']} ratio={cfg['ratio']} -> removed {cfg['removed_total']} neurons total across {cfg['layers']} layers; per-layer count varies {cfg['removed_per_layer_min']}..{cfg['removed_per_layer_max']} (candidates={cfg['candidate_total']}, excluded_layers={cfg['excluded_layers']}, forced_pruned_layers={cfg.get('forced_pruned_layers', [])})")
    else:
        print(f"[make_ffn_prune] metric={cfg['metric']} alloc=layer scope={cfg['token_scope']} select={cfg['select']} ratio={cfg['ratio']} intermediate={cfg['intermediate_size']} -> remove {cfg['removed_per_layer']}/layer x {len(cfg['candidate_layers'])} candidate layers = {cfg['removed_total']} neurons (excluded_layers={cfg['excluded_layers']})" + ('  (PRUNING HIGHEST-NORM = anti-correlation control)' if cfg['select'] == 'high' else ''))
    print(f'  wrote {args.out}')
if __name__ == '__main__':
    main()
