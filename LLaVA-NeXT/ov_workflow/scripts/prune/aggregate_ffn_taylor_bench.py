from __future__ import annotations
import argparse
import csv
import gzip
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import torch
FIELDS = ('scores', 'scores_vision', 'scores_text', 'scores_structured', 'scores_structured_vision', 'scores_structured_text')

def _atomic_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    flat = values.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    if flat.numel() == 0:
        return torch.empty_like(values, dtype=torch.float32, device='cpu')
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError('cannot rank non-finite values')
    sorted_values, order = torch.sort(flat, stable=True)
    n = int(flat.numel())
    if n == 1:
        return torch.zeros(shape, dtype=torch.float32)
    group_start = torch.ones(n, dtype=torch.bool)
    group_start[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = group_start.cumsum(0) - 1
    group_count = torch.bincount(group_id)
    group_end = group_count.cumsum(0) - 1
    group_begin = group_end - group_count + 1
    group_average = (group_begin.to(torch.float64) + group_end.to(torch.float64)) / 2.0
    sorted_ranks = group_average[group_id] / float(n - 1)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = sorted_ranks
    return ranks.reshape(shape).to(torch.float32)

def _as_matrix(obj: Mapping[str, Any], field: str, layers: int, intermediate: int) -> torch.Tensor:
    values = obj.get(field)
    if not isinstance(values, Mapping):
        raise KeyError(f'missing score mapping: {field}')
    matrix = torch.stack([torch.as_tensor(values[layer], dtype=torch.float32) for layer in range(layers)])
    if tuple(matrix.shape) != (layers, intermediate):
        raise ValueError(f'{field}: expected {(layers, intermediate)}, got {tuple(matrix.shape)}')
    if not bool(torch.isfinite(matrix).all().item()):
        raise ValueError(f'{field}: contains NaN/Inf')
    if bool((matrix < 0).any().item()):
        raise ValueError(f'{field}: contains negative Taylor scores')
    return matrix

def _to_layer_dict(matrix: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {layer: matrix[layer].contiguous() for layer in range(matrix.shape[0])}

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--files', nargs='+', required=True, help='completed per-task .pt files')
    parser.add_argument('--out_pt', required=True)
    parser.add_argument('--out_csv', default='', help='optional .csv or .csv.gz with all neuron ranks')
    parser.add_argument('--expected_samples', type=int, default=200)
    parser.add_argument('--expected_layers', type=int, default=28)
    parser.add_argument('--expected_intermediate', type=int, default=18944)
    return parser.parse_args()

def main() -> int:
    args = _parse_args()
    paths = [Path(value).expanduser().resolve() for value in args.files]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('missing per-task files: ' + ', '.join(missing))
    objects: List[Mapping[str, Any]] = []
    tasks: List[str] = []
    model_hashes = set()
    mask_hashes = set()
    for path in paths:
        obj = torch.load(path, map_location='cpu')
        task = str(obj.get('task', ''))
        if not task:
            raise ValueError(f'missing task metadata: {path}')
        if task in tasks:
            raise ValueError(f'duplicate task: {task}')
        if int(obj.get('num_samples', -1)) != args.expected_samples:
            raise ValueError(f"{task}: expected {args.expected_samples} successful samples, got {obj.get('num_samples')}")
        if int(obj.get('layers', -1)) != args.expected_layers:
            raise ValueError(f"{task}: expected {args.expected_layers} layers, got {obj.get('layers')}")
        if int(obj.get('intermediate_size', -1)) != args.expected_intermediate:
            raise ValueError(f"{task}: expected intermediate={args.expected_intermediate}, got {obj.get('intermediate_size')}")
        tasks.append(task)
        objects.append(obj)
        model_hashes.add(str(obj.get('model_config_sha256', '')))
        mask_hashes.add(str(obj.get('ablation_sha256', '')))
    if len(model_hashes) != 1:
        raise ValueError(f'per-task files use different model config hashes: {model_hashes}')
    if len(mask_hashes) != 1:
        raise ValueError(f'per-task files use different ablation hashes: {mask_hashes}')
    layers, intermediate = (args.expected_layers, args.expected_intermediate)
    layerwise: Dict[str, torch.Tensor] = {}
    global_rank: Dict[str, torch.Tensor] = {}
    raw_mean: Dict[str, torch.Tensor] = {}
    for field in FIELDS:
        matrices = [_as_matrix(obj, field, layers, intermediate) for obj in objects]
        raw_mean[field] = torch.stack(matrices).mean(dim=0)
        layer_sum = torch.zeros((layers, intermediate), dtype=torch.float64)
        global_sum = torch.zeros((layers, intermediate), dtype=torch.float64)
        for matrix in matrices:
            for layer in range(layers):
                layer_sum[layer].add_(percentile_rank(matrix[layer]).to(torch.float64))
            global_sum.add_(percentile_rank(matrix).to(torch.float64))
        layerwise[field] = (layer_sum / len(matrices)).to(torch.float32)
        global_rank[field] = (global_sum / len(matrices)).to(torch.float32)
        print(f'[aggregate] ranked {field}: {len(matrices)} tasks x {layers}x{intermediate}', flush=True)

    def mean_safe_rank(vision_field: str, text_field: str) -> Tuple[torch.Tensor, torch.Tensor]:
        safe_layer_sum = torch.zeros((layers, intermediate), dtype=torch.float64)
        safe_global_sum = torch.zeros((layers, intermediate), dtype=torch.float64)
        for obj in objects:
            vision = _as_matrix(obj, vision_field, layers, intermediate)
            text = _as_matrix(obj, text_field, layers, intermediate)
            for layer in range(layers):
                safe_layer_sum[layer].add_(torch.maximum(percentile_rank(vision[layer]), percentile_rank(text[layer])).to(torch.float64))
            safe_global_sum.add_(torch.maximum(percentile_rank(vision), percentile_rank(text)).to(torch.float64))
        return ((safe_layer_sum / len(objects)).to(torch.float32), (safe_global_sum / len(objects)).to(torch.float32))
    safe_layerwise, safe_global = mean_safe_rank('scores_vision', 'scores_text')
    safe_structured_layerwise, safe_structured_global = mean_safe_rank('scores_structured_vision', 'scores_structured_text')
    result = {'version': 2, 'metric': 'ffn_taylor_multibench_mean_percentile_rank', 'rank_direction': 'lower_is_less_important', 'tasks': tasks, 'num_tasks': len(tasks), 'samples_per_task': args.expected_samples, 'layers': layers, 'intermediate_size': intermediate, 'total_neurons': layers * intermediate, 'model_config_sha256': next(iter(model_hashes)), 'ablation_sha256': next(iter(mask_hashes)), 'source_files': [str(path) for path in paths], 'scores': _to_layer_dict(layerwise['scores']), 'scores_vision': _to_layer_dict(layerwise['scores_vision']), 'scores_text': _to_layer_dict(layerwise['scores_text']), 'scores_safe': _to_layer_dict(safe_layerwise), 'scores_structured': _to_layer_dict(layerwise['scores_structured']), 'scores_structured_vision': _to_layer_dict(layerwise['scores_structured_vision']), 'scores_structured_text': _to_layer_dict(layerwise['scores_structured_text']), 'scores_structured_safe': _to_layer_dict(safe_structured_layerwise), 'global_rank': {field: _to_layer_dict(value) for field, value in global_rank.items()}, 'scores_safe_global': _to_layer_dict(safe_global), 'scores_structured_safe_global': _to_layer_dict(safe_structured_global), 'raw_mean': {field: _to_layer_dict(value) for field, value in raw_mean.items()}}
    out_pt = Path(args.out_pt).expanduser().resolve()
    _atomic_torch_save(result, out_pt)
    print(f'[aggregate] wrote PT -> {out_pt}', flush=True)
    if args.out_csv:
        out_csv = Path(args.out_csv).expanduser().resolve()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_csv.with_name(out_csv.name + '.tmp')
        columns = ['layer', 'neuron', 'token_all_layer_rank', 'token_vision_layer_rank', 'token_text_layer_rank', 'token_safe_layer_rank', 'structured_all_layer_rank', 'structured_vision_layer_rank', 'structured_text_layer_rank', 'structured_safe_layer_rank', 'token_all_global_rank', 'token_vision_global_rank', 'token_text_global_rank', 'token_safe_global_rank', 'structured_all_global_rank', 'structured_vision_global_rank', 'structured_text_global_rank', 'structured_safe_global_rank']
        opener = gzip.open if out_csv.suffix == '.gz' else open
        open_kwargs = {'mode': 'wt', 'newline': '', 'encoding': 'utf-8'}
        with opener(tmp, **open_kwargs) as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for layer in range(layers):
                for neuron in range(intermediate):
                    writer.writerow([layer, neuron, float(layerwise['scores'][layer, neuron]), float(layerwise['scores_vision'][layer, neuron]), float(layerwise['scores_text'][layer, neuron]), float(safe_layerwise[layer, neuron]), float(layerwise['scores_structured'][layer, neuron]), float(layerwise['scores_structured_vision'][layer, neuron]), float(layerwise['scores_structured_text'][layer, neuron]), float(safe_structured_layerwise[layer, neuron]), float(global_rank['scores'][layer, neuron]), float(global_rank['scores_vision'][layer, neuron]), float(global_rank['scores_text'][layer, neuron]), float(safe_global[layer, neuron]), float(global_rank['scores_structured'][layer, neuron]), float(global_rank['scores_structured_vision'][layer, neuron]), float(global_rank['scores_structured_text'][layer, neuron]), float(safe_structured_global[layer, neuron])])
        os.replace(tmp, out_csv)
        print(f'[aggregate] wrote CSV -> {out_csv}', flush=True)
    final_vision = raw_mean['scores_vision'][-1]
    print(f'[aggregate] final-layer vision raw Taylor: nonzero={int(torch.count_nonzero(final_vision).item())}/{intermediate}; min={float(final_vision.min()):.4g} max={float(final_vision.max()):.4g}', flush=True)
    print(f'[aggregate] DONE tasks={len(tasks)} samples={len(tasks) * args.expected_samples} neurons={layers * intermediate}', flush=True)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
