import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--alignment', type=int, default=64)
    parser.add_argument('--rep', type=int, default=100)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--tiles')
    parser.add_argument('--methods', nargs='+', choices=['mwop', 'mwop_p', 'mwop_z'], default=['mwop', 'mwop_p', 'mwop_z'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    config_dir = root / 'pruning_config/llava_onevision'
    commands = []
    for method in args.methods:
        output = args.output / method
        if output.exists():
            raise FileExistsError(output)
        common = ['--model-path', args.model, '--config-dir', str(config_dir), '--alignment', str(args.alignment), '--rep', str(args.rep), '--samples', str(args.samples), '--output', str(output)]
        if args.tiles:
            common += ['--tiles', args.tiles]
        if method == 'mwop':
            command = [sys.executable, str(root / 'benchmark_mwop.py'), '--scenario', 'base', '--task', 'all', '--prefix', '18', '--visual', '3215', '--suffix', '19'] + common
        else:
            command = [sys.executable, str(root / 'benchmark_mwop_composed.py'), '--method', method, '--config', str(root / 'configs/llava_onevision.json'), '--worker', 'serial'] + common
        commands.append(command)
    (args.output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    for command in commands:
        subprocess.run(command, cwd=root, check=True)
    if 'mwop' in args.methods:
        subprocess.run([sys.executable, str(root / 'summarize.py'), '--results', str(args.output)], cwd=root, check=True)

if __name__ == '__main__':
    main()
