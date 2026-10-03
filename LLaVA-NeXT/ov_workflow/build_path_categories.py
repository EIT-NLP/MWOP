import argparse
import csv
import math
from pathlib import Path

CATEGORIES = {
    'General': [('gqa',), ('vqav2_val',), ('ok_vqa_val2014',)],
    'OCR': [('textvqa_val',), ('docvqa_val',), ('ocrbench',)],
    'Reasoning': [('scienceqa_img',), ('ai2d',), ('mmstar',)],
    'Grounding': [('refcoco_bbox_rec_testA', 'refcoco_bbox_rec_testB'), ('refcoco+_bbox_rec_testA', 'refcoco+_bbox_rec_testB'), ('refcocog_bbox_rec_test',)]
}
PATHS = ['v2v', 't2v', 't2t']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=256)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = {}
    keys = {(layer, head) for layer in range(28) for head in range(28)}
    for task in {task for units in CATEGORIES.values() for unit in units for task in unit}:
        path = args.input / f'taylor_{task}_n{args.samples}.csv'
        with path.open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        table = {(int(row['layer']), int(row['head'])): row for row in rows}
        if set(table) != keys or len(rows) != len(keys):
            raise ValueError(f'Incorrect layer/head coverage: {path}')
        for row in rows:
            for role in PATHS:
                if int(row[f'taylor_{role}_count']) <= 0 or not math.isfinite(float(row[f'taylor_{role}_abs_mean_pq'])):
                    raise ValueError(f'Incomplete/nonfinite Taylor entry: {path}')
        data[task] = table
    args.output.mkdir(parents=True, exist_ok=True)
    cols = ['layer', 'head'] + [f'taylor_{role}_{field}' for role in PATHS for field in ['abs_mean', 'abs_mean_pq', 'count']]
    groups = dict(CATEGORIES)
    groups['overall'] = [unit for units in CATEGORIES.values() for unit in units]
    for category, units in groups.items():
        with (args.output / f'path_{category}.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=cols)
            writer.writeheader()
            for layer, head in sorted(keys):
                row = {'layer': layer, 'head': head}
                for col in cols[2:]:
                    means = [sum(float(data[task][(layer, head)][col]) for task in unit) / len(unit) for unit in units]
                    row[col] = round(sum(means)) if col.endswith('_count') else sum(means) / len(means)
                writer.writerow(row)

if __name__ == '__main__':
    main()
