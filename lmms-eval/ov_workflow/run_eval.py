import argparse
import os
import subprocess
import sys
from pathlib import Path

TASKS = ['gqa', 'vqav2_val', 'ok_vqa_val2014', 'textvqa_val', 'docvqa_val', 'ocrbench', 'scienceqa_img', 'ai2d', 'mmstar', 'refcoco_bbox_rec_testA', 'refcoco_bbox_rec_testB', 'refcoco+_bbox_rec_testA', 'refcoco+_bbox_rec_testB', 'refcocog_bbox_rec_test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--mask', type=Path)
    parser.add_argument('--tasks', default=','.join(TASKS))
    parser.add_argument('--output', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--video-frames', type=int, default=32)
    args, extra = parser.parse_known_args()
    bundle = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(bundle / 'LLaVA-NeXT'), str(bundle / 'lmms-eval'), env.get('PYTHONPATH', '')])
    mask = args.mask or Path(args.model) / 'ablation_config.json'
    if args.mask and not mask.is_file():
        raise FileNotFoundError(mask)
    parts = [f'pretrained={args.model}', 'conv_template=qwen_1_5', 'attn_implementation=sdpa', f'max_frames_num={args.video_frames}']
    if mask.is_file():
        parts.append(f'ablation_config_path={mask.resolve()}')
    command = [sys.executable, '-m', 'lmms_eval', '--model', 'llava_onevision', '--model_args', ','.join(parts), '--tasks', args.tasks, '--batch_size', '1', '--output_path', args.output, '--log_samples']
    if args.limit:
        command += ['--limit', str(args.limit)]
    subprocess.run(command + extra, cwd=bundle, env=env, check=True)

if __name__ == '__main__':
    main()
