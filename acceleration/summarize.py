import argparse
import json
import math
from pathlib import Path
from theory_ov import compute

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def component_sum(rows, key):
    if len(rows) != 28 or sorted(row['layer'] for row in rows) != list(range(28)):
        raise ValueError('Expected complete measurements for all 28 layers')
    values = [float(row[key]['ms']) for row in rows]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise ValueError(f'Invalid latency: {key}')
    return sum(values)

def measurement_alignment(directory):
    """Read packing width from each measurement's own manifest, never infer it from latency."""
    manifest = read(directory / 'manifest.json')
    value = manifest.get('args', {}).get('alignment')
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f'Missing or invalid FFN alignment in {directory / "manifest.json"}')
    return value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--config-dir', type=Path, default=Path(__file__).resolve().parent / 'pruning_config/llava_onevision')
    args = parser.parse_args()
    base = args.results / 'mwop'
    if read(base / 'status.json')['status'] != 'complete':
        raise ValueError('The Dense/MWOP run is incomplete')
    modules = read(base / 'modules.json')
    ttft = read(base / 'ttft.json')
    latency = {
        'dense': [component_sum(modules, 'attn_dense'), component_sum(modules, 'ffn_dense'), float(ttft['dense']['ms'])],
        'mwop': [component_sum(modules, 'attn_mwop'), component_sum(modules, 'ffn_mwop'), float(ttft['both']['ms'])],
    }
    base_alignment = measurement_alignment(base)
    alignments = {'dense': base_alignment, 'mwop': base_alignment}
    for method in ('mwop_p', 'mwop_z'):
        directory = args.results / method
        if not directory.exists():
            continue
        if read(directory / 'status.json')['status'] != 'complete':
            raise ValueError(f'The {method} run is incomplete')
        rows = read(directory / 'modules.json')
        latency[method] = [component_sum(rows, 'attention_module'), component_sum(rows, 'ffn_module'), float(read(directory / 'ttft.json')['ms'])]
        alignments[method] = measurement_alignment(directory)
    theoretical = {row['method']: row for row in compute(args.config_dir)['rows']}
    fields = ['attn_tflops', 'ffn_tflops', 'total_tflops']
    labels = {'dense': 'Dense', 'mwop': 'MWOP', 'mwop_p': 'MWOP_p', 'mwop_z': 'MWOP_z'}
    lines = [
        '# OV-7B acceleration results', '',
        'Logical theory: V=3215, T=19; system tokens, probes and LM head are excluded. Packed Total adds visual FFN padding at the alignment recorded in each run manifest.', '',
        'Latency: fixed-shape decoder prefill with 18 prefix tokens + 3215 visual tokens + 19 suffix tokens. Attn/FFN columns sum separately measured layer modules; Prefill measures the complete decoder path, so they need not add up. Vision frontend and online token-selection scoring are excluded.', '',
        '| Method | Logical Attn (TFLOP) | Logical FFN (TFLOP) | Logical Total (TFLOP) | Packed Total (TFLOP) | FFN Alignment | Attn Latency (ms) | FFN Latency (ms) | Prefill Latency (ms) |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    report = []
    for method, times in latency.items():
        if not all(math.isfinite(value) and value > 0 for value in times):
            raise ValueError(f'Invalid latency for {method}')
        theory = theoretical[method]
        packed = next(row for row in compute(args.config_dir, ffn_alignment=alignments[method])['rows'] if row['method'] == method)
        values = [f"{theory[field]:.2f} ({100 * (1 - theory[field] / theoretical['dense'][field]):.1f}% reduction)" for field in fields]
        values += [f"{packed['total_tflops']:.4f}", str(alignments[method])]
        values += [f'{value:.4f} ({latency["dense"][i] / value:.1f}x)' for i, value in enumerate(times)]
        lines.append('| ' + ' | '.join([labels[method]] + values) + ' |')
        report.append(dict(theory, ffn_alignment=alignments[method], packed_ffn_tflops=packed['ffn_tflops'], packed_total_tflops=packed['total_tflops'], ffn_padding_tflops=packed['ffn_padding_tflops'], attn_latency_ms=times[0], ffn_latency_ms=times[1], prefill_latency_ms=times[2]))
    (args.results / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.results / 'summary.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(args.results / 'REPORT.md')

if __name__ == '__main__':
    main()
