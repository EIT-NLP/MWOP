import argparse
import json
from pathlib import Path

def causal(n):
    return n * (n + 1) // 2

def compute(config_dir, visual=3215, text=19, ffn_alignment=1):
    """Report logical decoder FLOPs and FFN packing overhead with the same token scope."""
    if ffn_alignment < 1 or visual < 1 or text < 0:
        raise ValueError('ffn_alignment and visual must be positive; text must be non-negative')
    root = Path(config_dir)
    masks = [json.loads((root / name).read_text(encoding='utf-8')) for name in ('V2V_40.json', 'T2V_60.json', 'T2T_10.json')]
    flags = [[[False] * 28 for _ in range(3)] for _ in range(28)]
    for role, obj in enumerate(masks):
        for layer, head in obj['heads']:
            flags[layer][role][head] = True
    obj = json.loads((root / 'FFN_visual50_text0_L27first.json').read_text(encoding='utf-8'))
    neurons = next(component['neurons'] for component in obj['components'] if component['token_scope'] == 'vision')
    rows = []
    for method in ['dense', 'mwop', 'zoo', 'pdrop', 'mwop_z', 'mwop_p']:
        attention, ffn, logical_ffn = 0, 0, 0
        for layer in range(28):
            v = visual
            if method in ('zoo', 'mwop_z'):
                v = visual // 2
            elif method in ('pdrop', 'mwop_p'):
                for boundary, ratio in [(7, 0.5436), (14, 0.2956), (21, 0.1607)]:
                    if layer >= boundary:
                        v = int(visual * ratio)
            pairs = 28 * (causal(v) + text * v + causal(text))
            width = 18944
            if method.startswith('mwop'):
                pairs -= sum(flags[layer][0]) * causal(v) + sum(flags[layer][1]) * text * v + sum(flags[layer][2]) * causal(text)
                width -= len(neurons.get(str(layer), []))
            attention += 4 * (v + text) * 3584 * (3584 + 512) + 4 * 128 * pairs
            packed_width = min(18944, (width + ffn_alignment - 1) // ffn_alignment * ffn_alignment)
            logical_ffn += 6 * 3584 * (width * v + 18944 * text)
            ffn += 6 * 3584 * (packed_width * v + 18944 * text)
        rows.append({'method': method, 'attn_tflops': attention / 1e12, 'ffn_tflops': ffn / 1e12, 'total_tflops': (attention + ffn) / 1e12, 'logical_ffn_tflops': logical_ffn / 1e12, 'logical_total_tflops': (attention + logical_ffn) / 1e12, 'ffn_padding_tflops': (ffn - logical_ffn) / 1e12})
    for row in rows:
        row['dense_retain_ratio'] = row['total_tflops'] / rows[0]['total_tflops']
    return {'visual': visual, 'text': text, 'system_tokens_included': False, 'probe_flops_included': False, 'lm_head_included': False, 'ffn_alignment': ffn_alignment, 'ffn_padding_included': ffn_alignment > 1, 'rows': rows}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', default=str(Path(__file__).resolve().parent / 'pruning_config/llava_onevision'))
    parser.add_argument('--visual', type=int, default=3215)
    parser.add_argument('--text', type=int, default=19)
    parser.add_argument('--ffn-alignment', type=int, default=1, help='retained visual FFN width packing; 1 gives logical FLOPs')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compute(args.config_dir, args.visual, args.text, args.ffn_alignment)
    value = json.dumps(result, indent=2)
    print(value)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(value + '\n', encoding='utf-8')

if __name__ == '__main__':
    main()
