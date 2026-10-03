from __future__ import annotations
import argparse
import csv
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]
PATHS = ('v2v', 't2v', 't2t')

def _load(cat_csv: Path, path: str) -> dict:
    col = f'taylor_{path}_abs_mean_pq'
    d = {}
    with open(cat_csv, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            v = (r.get(col) or '').strip()
            try:
                fv = float(v)
            except ValueError:
                fv = float('inf')
            if fv != fv:
                fv = float('inf')
            d[int(r['layer']), int(r['head'])] = fv
    return d

def _to_percentile(mag: dict, heads: list) -> dict:
    order = sorted(heads, key=lambda h: mag[h])
    n = len(heads)
    return {h: i / (n - 1) if n > 1 else 0.0 for i, h in enumerate(order)}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cats', default='General,OCR,Reasoning,Grounding', help='comma-separated task categories to aggregate over')
    ap.add_argument('--rankings_dir', type=Path, default=REPO_ROOT / 'configs' / 'rankings_4cat' / 'path')
    ap.add_argument('--agg', choices=['max', 'mean'], default='max', help='across-category aggregation (default max = worst-category)')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cats = [c.strip() for c in args.cats.split(',') if c.strip()]
    cat_csvs = {c: args.rankings_dir / f'path_{c}.csv' for c in cats}
    for c, p in cat_csvs.items():
        if not p.exists():
            ap.error(f'category csv not found: {p}')
    any_mag = _load(next(iter(cat_csvs.values())), 'v2v')
    heads = sorted(any_mag.keys())
    print(f'[{args.agg}] over {cats} | {len(heads)} heads')
    universal = {p: {} for p in PATHS}
    for p in PATHS:
        pct = {c: _to_percentile(_load(cat_csvs[c], p), heads) for c in cats}
        for h in heads:
            vals = [pct[c][h] for c in cats]
            universal[p][h] = max(vals) if args.agg == 'max' else sum(vals) / len(vals)
    cols = ['layer', 'head']
    for p in PATHS:
        cols += [f'taylor_{p}_abs_mean', f'taylor_{p}_abs_mean_pq', f'taylor_{p}_count']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for L, H in heads:
            row = [L, H]
            for p in PATHS:
                s = universal[p][L, H]
                row += [s, s, 3584]
            w.writerow(row)
    print(f'wrote {args.out}')
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
