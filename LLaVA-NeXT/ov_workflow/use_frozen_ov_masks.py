"""Export the exact OV attention and FFN masks used by the decoder benchmark."""
import argparse
import hashlib
import json
from pathlib import Path


def export_masks(output):
    source = Path(__file__).resolve().parents[2] / 'acceleration' / 'pruning_config' / 'llava_onevision'
    names = ['V2V_40.json', 'T2V_60.json', 'T2T_10.json', 'FFN_visual50_text0_L27first.json']
    configs = [json.loads((source / name).read_text('utf-8')) for name in names]
    heads = configs[:3]
    ffn = configs[3]
    vision = next(c['neurons'] for c in ffn['components'] if c['token_scope'] == 'vision')
    if sum(map(len, vision.values())) != 265216 or len(vision.get('27', [])) != 18944:
        raise ValueError('Unexpected frozen OV FFN pruning budget')
    for name, config, count in zip(names, heads, [314, 470, 78]):
        if len(config['heads']) != count or len(set(map(tuple, config['heads']))) != count:
            raise ValueError(f'Unexpected frozen attention mask: {name}')
    metadata = {'source_sha256': {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}, 'ffn_policy': 'Frozen global 50% visual channels with layer 27 fully pruned; text channels retained.'}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    products = {'path_mask.json': {'ablation_type': 'composite', 'components': heads, '_meta': metadata}, 'ffn_mask.json': ffn, 'joint_mask.json': {'ablation_type': 'composite', 'components': heads + ffn['components'], '_meta': metadata}}
    for name, data in products.items():
        (output / name).write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    return products


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_masks(args.output)
    print(args.output / 'joint_mask.json')
