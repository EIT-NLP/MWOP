import argparse
import json
import subprocess
import sys
from pathlib import Path
from _bootstrap import WORKFLOW

TASKS = ['gqa', 'vqav2_val', 'ok_vqa_val2014', 'textvqa_val', 'docvqa_val', 'ocrbench', 'scienceqa_img', 'ai2d', 'mmstar', 'refcoco_bbox_rec_testA', 'refcoco_bbox_rec_testB', 'refcoco+_bbox_rec_testA', 'refcoco+_bbox_rec_testB', 'refcocog_bbox_rec_test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--kind', choices=['path', 'ffn', 'both'], default='path')
    parser.add_argument('--tasks', nargs='+', default=TASKS)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--mask')
    parser.add_argument('--max-image-side', type=int, default=768)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    mask = Path(args.mask).resolve() if args.mask else WORKFLOW / 'configs/baseline_noop.json'
    commands = []
    for kind in (['path', 'ffn'] if args.kind == 'both' else [args.kind]):
        for task in args.tasks:
            directory = output / kind
            directory.mkdir(exist_ok=True)
            suffix = 'csv' if kind == 'path' else 'pt'
            script = 'path_analysis/collect_path_taylor.py' if kind == 'path' else 'prune/collect_ffn_taylor_bench.py'
            command = [sys.executable, str(WORKFLOW / 'scripts' / script), '--pretrained', args.model, '--task', task, '--n_samples', str(args.samples), '--seed', str(args.seed), '--device', args.device, '--max_image_side', str(args.max_image_side), '--out', str(directory / f'taylor_{task}_n{args.samples}.{suffix}')]
            if kind == 'ffn':
                command += ['--ablation_config', str(mask)]
            elif args.mask:
                parser.error('Path Taylor uses the original unmasked OV collector; --mask applies only to --kind ffn')
            commands.append(command)
    (output / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n', encoding='utf-8')
    for command in commands:
        subprocess.run(command, cwd=WORKFLOW, check=True)

if __name__ == '__main__':
    main()
