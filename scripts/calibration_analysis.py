#!/usr/bin/env python3
"""Calibration / failure analysis on stored test predictions (no model execution).

For HAMMER and each CRAF run: expected calibration error (15 equal-width bins on the
predicted-class confidence), mean confidence on wrong decisions, the share of total
cross-entropy coming from wrong decisions, and the number of very confident errors (conf>0.99).
Usage: python scripts/calibration_analysis.py  -> results/calibration_analysis.json
"""
import json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    'v1_lr1e4': 'results/server_records/hammer-craf/craf_results/test_comparison_predictions.jsonl',
    'v1_lr1e5': 'results/server_records/hammer-craf/craf_lr_1e5/test_comparison_predictions.jsonl',
    'v2_combined': 'results/server_records/hammer-craf-v2/craf_v2_combined_seed42/test_comparison_predictions.jsonl',
}

def stats(y, p, bins=15):
    eps = np.finfo(np.float32).eps
    q = np.clip(p, eps, 1 - eps)
    nll = -(y * np.log(q) + (1 - y) * np.log(1 - q))
    pred = (p > 0.5).astype(int); conf = np.where(pred == 1, p, 1 - p); correct = pred == y
    edges = np.linspace(0.5, 1.0, bins + 1); ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.any(): ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    wrong = ~correct
    return {'ece': float(ece), 'ce': float(nll.mean()), 'errors': int(wrong.sum()),
            'mean_conf_wrong': float(conf[wrong].mean()), 'ce_share_from_errors': float(nll[wrong].sum() / nll.sum()),
            'confident_errors_gt099': int((wrong & (conf > 0.99)).sum())}

out = {}
for name, rel in RUNS.items():
    rows = [json.loads(l) for l in open(ROOT / rel)]
    y = np.array([r['target'] for r in rows]); b = np.array([r['baseline_fake_probability'] for r in rows]); c = np.array([r['craf_fake_probability'] for r in rows])
    if 'hammer' not in out: out['hammer'] = stats(y, b)
    out[name] = stats(y, c)
    margin = np.log(np.clip(c,1e-7,1-1e-7)/np.clip(1-c,1e-7,1)) - np.log(np.clip(b,1e-7,1-1e-7)/np.clip(1-b,1e-7,1))
    out[name]['mean_abs_margin_change'] = float(np.abs(margin).mean())
for k, v in out.items(): print(k, v)
(ROOT / 'results/calibration_analysis.json').write_text(json.dumps(out, indent=1))
