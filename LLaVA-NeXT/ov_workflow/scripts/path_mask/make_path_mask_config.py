from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional, Tuple
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAYLOR_CSV = REPO_ROOT / 'configs' / 'rankings_4cat' / 'path' / 'path_univmax.csv'
PATHS = ('v2v', 't2v', 't2t')
ABLATION_TYPE = {'v2v': 'mask_v2v', 't2v': 'mask_t2v', 't2t': 'mask_t2t'}

def _load_taylor_rows(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {'layer': int(raw['layer']), 'head': int(raw['head'])}
            for p in PATHS:
                col = f'taylor_{p}_abs_mean_pq'
                if col not in raw:
                    raise KeyError(f'CSV {csv_path} is missing column {col!r}; expected the taylor_gqa_full.csv layout.')
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

def _select_bottom_pct(rows: List[dict], path: str, pct: int) -> List[Tuple[int, int]]:
    if pct <= 0:
        return []
    n_total = len(rows)
    n_picked = int(round(n_total * pct / 100))
    if n_picked <= 0:
        return []
    if n_picked >= n_total:
        n_picked = n_total
    score_col = f'score_{path}'
    ranked = sorted(rows, key=lambda r: r[score_col])
    return [(r['layer'], r['head']) for r in ranked[:n_picked]]

def _make_component(path: str, heads: List[Tuple[int, int]]) -> Optional[dict]:
    if not heads:
        return None
    return {'ablation_type': ABLATION_TYPE[path], 'heads': [[int(L), int(H)] for L, H in heads]}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--v2v', type=int, default=0, help='percent of V2V heads to mask (0-100)')
    ap.add_argument('--t2v', type=int, default=0, help='percent of T2V heads to mask (0-100)')
    ap.add_argument('--t2t', type=int, default=0, help='percent of T2T heads to mask (0-100)')
    ap.add_argument('--taylor_csv', type=Path, default=DEFAULT_TAYLOR_CSV, help=f'Path to taylor_gqa_full.csv (default: {DEFAULT_TAYLOR_CSV})')
    ap.add_argument('--output', type=Path, required=True, help='Output JSON path (will be created in any parent dir).')
    args = ap.parse_args()
    for name, pct in (('v2v', args.v2v), ('t2v', args.t2v), ('t2t', args.t2t)):
        if pct < 0 or pct > 100:
            ap.error(f'--{name} must be in [0, 100], got {pct}')
    if not args.taylor_csv.exists():
        ap.error(f'taylor CSV not found: {args.taylor_csv}')
    rows = _load_taylor_rows(args.taylor_csv)
    n_total = len(rows)
    print(f'loaded {n_total} (layer, head) rows from {args.taylor_csv}')
    pct_by_path = {'v2v': args.v2v, 't2v': args.t2v, 't2t': args.t2t}
    components: List[dict] = []
    counts: dict = {}
    for path in PATHS:
        heads = _select_bottom_pct(rows, path, pct_by_path[path])
        counts[path] = len(heads)
        comp = _make_component(path, heads)
        if comp is not None:
            components.append(comp)
    if not components:
        print('WARN: all three percentages are 0 -> the generated JSON has an empty `components` list. apply_ablation_config will treat this as a no-op (mask nothing).')
    out_payload = {'ablation_type': 'composite', '_meta': {'source': str(args.taylor_csv), 'v2v_pct': args.v2v, 't2v_pct': args.t2v, 't2t_pct': args.t2t, 'n_total_heads_per_path': n_total, 'n_masked': counts}, 'components': components}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"masked: v2v={counts['v2v']}, t2v={counts['t2v']}, t2t={counts['t2t']}  -> {args.output}")
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
