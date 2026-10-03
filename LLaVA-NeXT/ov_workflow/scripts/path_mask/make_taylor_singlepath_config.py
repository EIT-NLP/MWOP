from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAYLOR_CSV = REPO_ROOT / 'configs' / 'path_calibration' / 'data' / 'taylor_gqa_full.csv'
PATHS = ('v2v', 't2v', 't2t')
ABLATION_TYPE = {'v2v': 'mask_v2v', 't2v': 'mask_t2v', 't2t': 'mask_t2t'}

def _load_scored_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {'layer': int(raw['layer']), 'head': int(raw['head'])}
            for p in PATHS:
                col = f'taylor_{p}_abs_mean_pq'
                if col not in raw:
                    raise KeyError(f'missing column {col!r} in {csv_path}')
                val = raw[col].strip()
                try:
                    fval = float(val)
                except ValueError:
                    fval = float('inf')
                if fval != fval:
                    fval = float('inf')
                row[f'score_{p}'] = fval
            rows.append(row)
    return rows

def _bottom_k_heads(rows: List[dict], path: str, cut_pct: int) -> Tuple[List[Tuple[int, int]], int]:
    n_total = len(rows)
    n_masked = int(round(n_total * cut_pct / 100))
    if n_masked <= 0:
        return ([], 0)
    if n_masked >= n_total:
        n_masked = n_total
    ranked = sorted(rows, key=lambda r: r[f'score_{path}'])
    picked = [(r['layer'], r['head']) for r in ranked[:n_masked]]
    picked.sort()
    return (picked, n_masked)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--taylor_csv', type=Path, default=DEFAULT_TAYLOR_CSV, help=f'Taylor CSV (default: {DEFAULT_TAYLOR_CSV})')
    ap.add_argument('--output_dir', type=Path, default=REPO_ROOT / 'configs' / 'auto_path_mask' / 'taylor_singlepath', help='Output dir for taylor_{path}_bot{cut}.json files.')
    ap.add_argument('--cuts', default='25,50,75', help='Comma-separated mask percentages (default 25,50,75).')
    args = ap.parse_args()
    if not args.taylor_csv.exists():
        ap.error(f'taylor CSV not found: {args.taylor_csv}')
    rows = _load_scored_rows(args.taylor_csv)
    print(f'loaded {len(rows)} (layer, head) rows with Taylor scores')
    cuts = [int(c) for c in str(args.cuts).split(',') if c.strip()]
    overview = []
    for path in PATHS:
        for cut in cuts:
            heads, n_masked = _bottom_k_heads(rows, path, cut)
            tag = f'taylor_{path}_bot{cut:02d}'
            out_path = args.output_dir / f'{tag}.json'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {'ablation_type': ABLATION_TYPE[path], 'heads': [[int(L), int(H)] for L, H in heads], '_meta': {'source': str(args.taylor_csv), 'selection': 'taylor_abs_mean_pq_bottom_k', 'path': path, 'ablation_type': ABLATION_TYPE[path], 'cut_pct': cut, 'n_total': len(rows), 'n_masked': n_masked, 'tag': tag}}
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            print(f'  {tag}: {n_masked}/{len(rows)} heads -> {out_path.name}')
            overview.append({'tag': tag, 'path': path, 'ablation_type': ABLATION_TYPE[path], 'cut_pct': cut, 'n_masked': n_masked, 'n_total': len(rows)})
    overview_csv = args.output_dir / 'overview.csv'
    with open(overview_csv, 'w', newline='', encoding='utf-8') as f:
        if overview:
            w = csv.DictWriter(f, fieldnames=list(overview[0].keys()))
            w.writeheader()
            w.writerows(overview)
    print(f'\n=== overview -> {overview_csv}')
    print(f'=== generated {len(overview)} Taylor single-path configs')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
