from __future__ import annotations
import argparse, csv, glob, math, os
PATHS = ('v2v', 't2v', 't2t')

def taskname(f):
    return os.path.basename(f)[len('taylor_'):-len('_n256.csv')]

def spearman(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a == a and b == b]
    if len(pairs) < 10:
        return float('nan')
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    n = len(xs)

    def rank(v):
        order = sorted(range(n), key=lambda i: v[i])
        rk = [0.0] * n
        for r, i in enumerate(order):
            rk[i] = r
        return rk
    rx, ry = (rank(xs), rank(ys))
    mx, my = (sum(rx) / n, sum(ry) / n)
    cov = sum(((rx[i] - mx) * (ry[i] - my) for i in range(n)))
    sx = math.sqrt(sum(((rx[i] - mx) ** 2 for i in range(n))))
    sy = math.sqrt(sum(((ry[i] - my) ** 2 for i in range(n))))
    return cov / (sx * sy) if sx > 0 and sy > 0 else float('nan')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='path_result/path_analysis')
    ap.add_argument('--gqa_full', default='configs/path_calibration/data/taylor_gqa_full.csv')
    ap.add_argument('--out_csv', default='docs/path_analysis/taylor_multitask_meanrank.csv')
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.dir, 'taylor_*_n256.csv')))
    data, counts = ({}, {})
    for f in files:
        t = taskname(f)
        data[t] = {}
        cnts = []
        for r in csv.DictReader(open(f)):
            k = (int(r['layer']), int(r['head']))
            data[t][k] = {p: float(r[f'taylor_{p}_abs_mean_pq']) for p in PATHS}
            cnts.append(int(r['taylor_v2v_count']))
        counts[t] = max(cnts) if cnts else 0
    tasks = sorted(data)
    keys = sorted(data[tasks[0]])
    vec = lambda t, p: [data[t][k][p] for k in keys]
    print(f'=== {len(tasks)} tasks loaded ===')
    print('  per-task processed (max count):')
    for t in tasks:
        print(f'    {t:32s} {counts[t]}')
    print('\n=== cross-task consistency: pairwise Spearman of per-head importance ===')
    for p in PATHS:
        rhos = []
        for i in range(len(tasks)):
            for j in range(i + 1, len(tasks)):
                rhos.append(spearman(vec(tasks[i], p), vec(tasks[j], p)))
        rhos = [r for r in rhos if r == r]
        print(f'  {p}: mean pairwise rho = {sum(rhos) / len(rhos):.3f}  (min {min(rhos):.3f}, max {max(rhos):.3f})')
    gqa_full = {}
    if os.path.isfile(args.gqa_full):
        for r in csv.DictReader(open(args.gqa_full)):
            gqa_full[int(r['layer']), int(r['head'])] = {p: float(r[f'taylor_{p}_abs_mean_pq']) for p in PATHS}
    if gqa_full:
        print('\n=== does GQA-only Taylor generalize? Spearman(GQA-full vs each task) ===')
        print(f"  {'task':32s}" + ''.join((f'{p:>8}' for p in PATHS)))
        gv = lambda p: [gqa_full[k][p] for k in keys]
        for t in tasks:
            print(f'  {t:32s}' + ''.join((f'{spearman(gv(p), vec(t, p)):>8.3f}' for p in PATHS)))
    print('\n=== per-task bottom-50% Venn: only_V2V / all_three / none (structure stability) ===')
    print(f"  {'task':32s}{'oV2V':>6}{'oT2V':>6}{'oT2T':>6}{'all3':>6}{'none':>6}")
    K = len(keys) // 2
    for t in tasks:
        bot = {}
        for p in PATHS:
            sl = sorted(keys, key=lambda k: data[t][k][p] if data[t][k][p] == data[t][k][p] else 1e+18)
            bot[p] = set(sl[:K])
        V, T, G = (bot['v2v'], bot['t2v'], bot['t2t'])
        allk = set(keys)
        print(f'  {t:32s}{len(V - T - G):>6}{len(T - V - G):>6}{len(G - V - T):>6}{len(V & T & G):>6}{len(allk - V - T - G):>6}')
    n = len(keys)
    agg = {k: {p: 0.0 for p in PATHS} for k in keys}
    ntask = {k: {p: 0 for p in PATHS} for k in keys}
    for t in tasks:
        for p in PATHS:
            v = [(data[t][k][p], k) for k in keys if data[t][k][p] == data[t][k][p]]
            v.sort()
            for r, (_, k) in enumerate(v):
                agg[k][p] += r / max(len(v) - 1, 1)
                ntask[k][p] += 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as fo:
        w = csv.writer(fo)
        cols = ['layer', 'head'] + [f'taylor_{p}_abs_mean_pq' for p in PATHS]
        w.writerow(cols)
        for l, h in keys:
            row = [l, h]
            for p in PATHS:
                row.append(agg[l, h][p] / max(ntask[l, h][p], 1))
            w.writerow(row)
    print(f'\n  wrote aggregated multi-task importance (mean percentile rank) -> {args.out_csv}')
    print('  (lower = consistently least-important across tasks = safest to mask)')
if __name__ == '__main__':
    main()
