from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

def load_json(path: Path) -> Dict[str, Any]:
    with path.open(encoding='utf-8') as stream:
        obj = json.load(stream)
    if not isinstance(obj, dict):
        raise ValueError(f'expected JSON object: {path}')
    return obj

def components(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    if obj.get('ablation_type') == 'composite':
        raw = obj.get('components')
        if not isinstance(raw, list):
            raise ValueError('composite path config has no components list')
        return [dict(item) for item in raw]
    return [dict(obj)]

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--path_config', type=Path, required=True)
    parser.add_argument('--ffn_config', type=Path, action='append', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    path_obj = load_json(args.path_config.expanduser().resolve())
    ffn_paths = [path.expanduser().resolve() for path in args.ffn_config]
    ffn_objs = [load_json(path) for path in ffn_paths]
    path_components = components(path_obj)
    path_kinds = [str(item.get('ablation_type', '')) for item in path_components]
    expected = {'mask_v2v', 'mask_t2v', 'mask_t2t'}
    if not expected.issubset(path_kinds):
        raise ValueError(f'path config must contain {sorted(expected)}, got {path_kinds}')
    ffn_components: List[Dict[str, Any]] = []
    ffn_metadata: List[Dict[str, Any]] = []
    seen_scopes = set()
    for ffn_path, ffn_obj in zip(ffn_paths, ffn_objs):
        if ffn_obj.get('ablation_type') != 'ffn_prune':
            raise ValueError(f'FFN config must have ablation_type=ffn_prune: {ffn_path}')
        scope = str(ffn_obj.get('token_scope', 'all'))
        if scope not in ('all', 'vision', 'text'):
            raise ValueError(f'invalid FFN token_scope in {ffn_path}: {scope!r}')
        if scope in seen_scopes:
            raise ValueError(f'duplicate FFN token_scope={scope!r}; combine each scope before merging')
        if not isinstance(ffn_obj.get('neurons'), dict):
            raise ValueError(f'FFN config has no neurons mapping: {ffn_path}')
        seen_scopes.add(scope)
        ffn_components.append({'ablation_type': 'ffn_prune', 'token_scope': scope, 'neurons': ffn_obj['neurons']})
        ffn_metadata.append({'ffn_config': str(ffn_path), 'metric': ffn_obj.get('metric'), 'ratio': ffn_obj.get('ratio'), 'token_scope': scope, 'removed_total': ffn_obj.get('removed_total'), 'excluded_layers': ffn_obj.get('excluded_layers', []), 'forced_pruned_layers': ffn_obj.get('forced_pruned_layers', [])})
    output = {'ablation_type': 'composite', 'components': path_components + ffn_components, 'metadata': {'path_config': str(args.path_config.expanduser().resolve()), 'ffn_configs': ffn_metadata, 'ffn_config': ffn_metadata[0]['ffn_config'], 'ffn_metric': ffn_metadata[0]['metric'], 'ffn_ratio': ffn_metadata[0]['ratio'], 'ffn_token_scope': ffn_metadata[0]['token_scope'], 'ffn_removed_total': ffn_metadata[0]['removed_total'], 'ffn_excluded_layers': ffn_metadata[0]['excluded_layers']}}
    destination = args.out.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False)
    os.replace(temporary, destination)
    print(f'wrote {destination} | path={path_kinds} | ffn=' + ', '.join((f"{item['token_scope']}:{item['removed_total']} neurons" for item in ffn_metadata)))
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
