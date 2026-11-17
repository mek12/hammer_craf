#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for CRAF vs. HAMMER on the stored test predictions.

Reads the three test_comparison_predictions.jsonl files (no model execution) and resamples
test examples with replacement, stratified by label, to obtain 95% percentile intervals for
the AUC, accuracy and cross-entropy differences (CRAF - HAMMER).
Assumption: examples are exchangeable. News-cluster dependence cannot be modelled because the
prediction rows carry no news_id, so the intervals are likely too narrow.
Usage: python scripts/bootstrap_uncertainty.py [--reps 2000] [--seed 0]
"""
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'v1_lr1e4': 'results/server_records/hammer-craf/craf_results/test_comparison_predictions.jsonl',
    'v1_lr1e5': 'results/server_records/hammer-craf/craf_lr_1e5/test_comparison_predictions.jsonl',
    'v2_combined': 'results/server_records/hammer-craf-v2/craf_v2_combined_seed42/test_comparison_predictions.jsonl',
}

def load(path):
    y, b, c = [], [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            y.append(r['target']); b.append(r['baseline_fake_probability']); c.append(r['craf_fake_probability'])
    return np.array(y, dtype=np.int8), np.array(b, dtype=np.float64), np.array(c, dtype=np.float64)

def auc(y, s):
    order = np.argsort(s, kind='mergesort')
    s_sorted = s[order]
    ranks = np.empty(len(s), dtype=np.float64)
    # average ranks for ties
    _, first, counts = np.unique(s_sorted, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0 + 1
    ranks[order] = np.repeat(avg, counts)
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def metrics(y, p):
    eps = np.finfo(np.float32).eps
    q = np.clip(p, eps, 1 - eps)
    ce = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
    return auc(y, p), np.mean((p > 0.5) == (y == 1)), ce

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=2000); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); out = {'reps': a.reps, 'seed': a.seed,
        'assumption': 'label-stratified iid resampling of test rows; news-cluster dependence not modelled', 'runs': {}}
    for name, rel in RUNS.items():
        y, b, c = load(ROOT / rel)
        idx1, idx0 = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        point = [x - z for x, z in zip(metrics(y, c), metrics(y, b))]
        draws = np.empty((a.reps, 3))
        for r in range(a.reps):
            i = np.concatenate([rng.choice(idx1, len(idx1)), rng.choice(idx0, len(idx0))])
            draws[r] = [x - z for x, z in zip(metrics(y[i], c[i]), metrics(y[i], b[i]))]
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        out['runs'][name] = {k: {'delta': float(point[j]), 'ci95': [float(lo[j]), float(hi[j])],
                                 'share_draws_gt0': float((draws[:, j] > 0).mean())}
                             for j, k in enumerate(['auc', 'accuracy', 'cross_entropy'])}
        print(name, json.dumps(out['runs'][name]))
    (ROOT / 'results/bootstrap_uncertainty.json').write_text(json.dumps(out, indent=1))

if __name__ == '__main__':
    main()
