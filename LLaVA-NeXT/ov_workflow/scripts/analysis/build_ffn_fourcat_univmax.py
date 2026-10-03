from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
import torch
LAYERS = 28
INTERMEDIATE = 18944
CATEGORIES: Mapping[str, Sequence[Tuple[str, ...]]] = {'General': (('gqa',), ('vqav2_val',), ('ok_vqa_val2014',)), 'OCR': (('textvqa_val',), ('docvqa_val',), ('ocrbench',)), 'Reasoning': (('scienceqa_img',), ('ai2d',), ('mmstar',)), 'Grounding': (('refcoco_bbox_rec_testA', 'refcoco_bbox_rec_testB'), ('refcoco+_bbox_rec_testA', 'refcoco+_bbox_rec_testB'), ('refcocog_bbox_rec_test',))}
FIELD_BY_METRIC = {'structured': {'total': 'scores_structured', 'vision': 'scores_structured_vision', 'text': 'scores_structured_text'}, 'legacy': {'total': 'scores', 'vision': 'scores_vision', 'text': 'scores_text'}}

def percentile_rank(values: torch.Tensor) -> torch.Tensor:
    shape = values.shape
    flat = values.detach().to(device='cpu', dtype=torch.float64).reshape(-1)
    if flat.numel() == 0:
        return torch.empty(shape, dtype=torch.float32)
    if not bool(torch.isfinite(flat).all().item()):
        raise ValueError('cannot rank NaN/Inf values')
    sorted_values, order = torch.sort(flat, stable=True)
    n = int(flat.numel())
    if n == 1:
        return torch.zeros(shape, dtype=torch.float32)
    starts = torch.ones(n, dtype=torch.bool)
    starts[1:] = sorted_values[1:] != sorted_values[:-1]
    group_id = starts.cumsum(0) - 1
    group_count = torch.bincount(group_id)
    group_end = group_count.cumsum(0) - 1
    group_begin = group_end - group_count + 1
    group_average = (group_begin.double() + group_end.double()) / 2.0
    sorted_ranks = group_average[group_id] / float(n - 1)
    ranks = torch.empty(n, dtype=torch.float64)
    ranks[order] = sorted_ranks
    return ranks.reshape(shape).float()

def matrix(obj: Mapping[str, Any], field: str) -> torch.Tensor:
    score_map = obj.get(field)
    if not isinstance(score_map, Mapping):
        raise KeyError(f'missing score field: {field}')
    result = torch.stack([torch.as_tensor(score_map[layer], dtype=torch.float32) for layer in range(LAYERS)])
    if tuple(result.shape) != (LAYERS, INTERMEDIATE):
        raise ValueError(f'{field}: expected {(LAYERS, INTERMEDIATE)}, got {tuple(result.shape)}')
    if not bool(torch.isfinite(result).all().item()) or bool((result < 0).any().item()):
        raise ValueError(f'{field}: scores must be finite and non-negative')
    return result

def rank_valid_layers(values: torch.Tensor, invalid_layers: Sequence[int]) -> torch.Tensor:
    invalid = set((int(layer) for layer in invalid_layers))
    valid = [layer for layer in range(LAYERS) if layer not in invalid]
    result = torch.zeros((LAYERS, INTERMEDIATE), dtype=torch.float32)
    ranked = percentile_rank(values[valid].reshape(-1)).reshape(len(valid), INTERMEDIATE)
    result[valid] = ranked
    return result

def layer_dict(values: torch.Tensor) -> Dict[int, torch.Tensor]:
    return {layer: values[layer].contiguous() for layer in range(LAYERS)}

def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    default_input = root / 'outputs' / 'ffn_act' / 'per_task'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input_dir', type=Path, default=default_input)
    parser.add_argument('--output_dir', type=Path, default=default_input.parent / 'rankings_4cat_univmax_structured')
    parser.add_argument('--expected_samples', type=int, default=200, help='required successful image samples in every benchmark file')
    parser.add_argument('--source_metric', choices=('structured', 'legacy'), default='structured', help='structured = whole-neuron Taylor (recommended); legacy = mean token |h*dL/dh|')
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    if args.expected_samples <= 0:
        raise ValueError('--expected_samples must be positive')
    paths = sorted(args.input_dir.expanduser().resolve().glob(f'taylor_*_n{args.expected_samples}.pt'))
    if len(paths) != 14:
        raise RuntimeError(f'expected 14 per-task files in {args.input_dir}, found {len(paths)}')
    objects: Dict[str, Mapping[str, Any]] = {}
    path_by_task: Dict[str, Path] = {}
    for path in paths:
        obj = torch.load(path, map_location='cpu')
        task = str(obj.get('task', ''))
        if not task or task in objects:
            raise ValueError(f'missing/duplicate task metadata: {path}')
        if int(obj.get('num_samples', -1)) != args.expected_samples:
            raise ValueError(f'{task}: expected {args.expected_samples} successful samples')
        if int(obj.get('layers', -1)) != LAYERS or int(obj.get('intermediate_size', -1)) != INTERMEDIATE:
            raise ValueError(f'{task}: unexpected FFN shape')
        objects[task] = obj
        path_by_task[task] = path
    expected_tasks = {task for units in CATEGORIES.values() for unit in units for task in unit}
    if set(objects) != expected_tasks:
        raise ValueError(f'task set mismatch: missing={sorted(expected_tasks - set(objects))}, extra={sorted(set(objects) - expected_tasks)}')
    model_hashes = {str(obj.get('model_config_sha256', '')) for obj in objects.values()}
    mask_hashes = {str(obj.get('ablation_sha256', '')) for obj in objects.values()}
    if len(model_hashes) != 1 or len(mask_hashes) != 1:
        raise ValueError('per-task files do not share one model/mask fingerprint')
    any_obj = next(iter(objects.values()))
    checkpoint = str(any_obj.get('pretrained', any_obj.get('ckpt', '')))
    ablation = str(any_obj.get('ablation_config', any_obj.get('head_mask_config', '')))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = FIELD_BY_METRIC[args.source_metric]
    category_scores: Dict[str, Dict[str, torch.Tensor]] = {category: {} for category in CATEGORIES}
    invalid_by_modality: Dict[str, List[int]] = {}
    for modality, field in fields.items():
        task_matrices = {task: matrix(obj, field) for task, obj in objects.items()}
        invalid_layers = [layer for layer in range(LAYERS) if all((int(torch.count_nonzero(values[layer]).item()) == 0 for values in task_matrices.values()))]
        invalid_by_modality[modality] = invalid_layers
        print(f'[{modality}] source={field} invalid/all-zero layers={invalid_layers}')
        for category, units in CATEGORIES.items():
            unit_ranks: List[torch.Tensor] = []
            unit_metadata: List[List[str]] = []
            for unit in units:
                raw_unit = torch.stack([task_matrices[task] for task in unit]).mean(dim=0)
                unit_ranks.append(rank_valid_layers(raw_unit, invalid_layers))
                unit_metadata.append(list(unit))
            category_mean = torch.stack(unit_ranks).mean(dim=0)
            category_rank = rank_valid_layers(category_mean, invalid_layers)
            category_scores[category][modality] = category_rank
            source_tasks = [task for unit in units for task in unit]
            payload = {'metric': f'taylor_4cat_category_{args.source_metric}_rank', 'mode': 'taylor', 'num_samples': sum((int(objects[task]['num_samples']) for task in source_tasks)), 'intermediate_size': INTERMEDIATE, 'layers': LAYERS, 'ckpt': checkpoint, 'head_mask_config': ablation, 'scores': layer_dict(category_rank), 'scope': 'category', 'category': category, 'modality': modality, 'source_metric': args.source_metric, 'source_field': field, 'rank_scope': 'global_across_valid_layers', 'rank_direction': 'lower_is_less_important', 'aggregation': 'mean of 3 equal task-unit percentile ranks, then category percentile rank', 'task_units': unit_metadata, 'source_files': [str(path_by_task[task]) for task in source_tasks], 'invalid_all_zero_layers': invalid_layers, 'model_config_sha256': next(iter(model_hashes)), 'ablation_sha256': next(iter(mask_hashes))}
            destination = output_dir / f'ffn_{category}_{modality}_taylor.pt'
            atomic_save(payload, destination)
            print(f'  wrote {destination.name}')
        universal = torch.stack([category_scores[category][modality] for category in CATEGORIES]).max(dim=0).values
        if invalid_layers:
            universal[invalid_layers] = 0
        universal_payload = {'metric': f'taylor_4cat_univmax_{args.source_metric}_rank', 'mode': 'taylor', 'num_samples': sum((int(obj['num_samples']) for obj in objects.values())), 'intermediate_size': INTERMEDIATE, 'layers': LAYERS, 'ckpt': checkpoint, 'head_mask_config': ablation, 'scores': layer_dict(universal), 'scope': 'universal_max', 'category': 'UniversalMax', 'modality': modality, 'source_metric': args.source_metric, 'source_field': field, 'rank_scope': 'global_across_valid_layers', 'rank_direction': 'lower_is_less_important', 'aggregation': 'elementwise max of four category percentile rankings', 'categories': list(CATEGORIES), 'category_files': [f'ffn_{category}_{modality}_taylor.pt' for category in CATEGORIES], 'source_files': [str(path_by_task[task]) for task in sorted(objects)], 'invalid_all_zero_layers': invalid_layers, 'model_config_sha256': next(iter(model_hashes)), 'ablation_sha256': next(iter(mask_hashes))}
        destination = output_dir / f'ffn_univmax_{modality}_taylor.pt'
        atomic_save(universal_payload, destination)
        print(f'  wrote {destination.name}')
    produced = sorted(output_dir.glob('ffn_*_taylor.pt'))
    if len(produced) != 15:
        raise RuntimeError(f'expected exactly 15 ranking files, found {len(produced)} in {output_dir}')
    print(f'DONE: {len(produced)} ranking files -> {output_dir}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
