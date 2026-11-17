#!/usr/bin/env python3
"""Reproduce test tables from canonical archived server reports.

Uses only the Python standard library. Does not run a model or infer significance.
"""
import argparse
import json
import math
from pathlib import Path

from evidence import normalized_sha, transcription_record

ROOT = Path(__file__).resolve().parents[1]
FILES = [('CRAF v1, lr=1e-4', 'craf_v1_lr1e4_test.json'),
         ('CRAF v1, lr=1e-5', 'craf_v1_lr1e5_test.json'),
         ('CRAF v2, combined', 'craf_v2_combined_test.json')]
SERVER_PATHS = {
    'craf_v1_lr1e4_test.json': 'hammer-craf/craf_results/test_comparison.json',
    'craf_v1_lr1e5_test.json': 'hammer-craf/craf_lr_1e5/test_comparison.json',
    'craf_v2_combined_test.json': 'hammer-craf-v2/craf_v2_combined_seed42/test_comparison.json',
}


def from_matrix(matrix):
    (tn, fp), (fn, tp) = matrix
    n = tn + fp + fn + tp
    f1_real = 2 * tn / (2 * tn + fp + fn)
    f1_fake = 2 * tp / (2 * tp + fp + fn)
    return dict(count=n, correct=tn + tp, accuracy=(tn + tp) / n,
                balanced_accuracy=.5 * (tn / (tn + fp) + tp / (tp + fn)),
                macro_f1=.5 * (f1_real + f1_fake), fp=fp, fn=fn)


def load_results(root=ROOT):
    records = []
    baseline = None
    identity = None
    for name, filename in FILES:
        server_path = root / 'results/server_records' / SERVER_PATHS[filename]
        result = json.loads(server_path.read_text())
        previous = transcription_record(filename, root)
        if normalized_sha(result) != previous['normalized_json_sha256']:
            raise ValueError(f'Historical transcription fingerprint differs: {filename}')
        if result['split'] != 'test' or result['count'] != 50705:
            raise ValueError(f'Unexpected split/count: {filename}')
        pair = (result['cache_identity']['backbone_sha256'], result['cache_identity']['metadata_sha256'])
        if baseline is None:
            baseline, identity = result['baseline'], pair
        elif result['baseline'] != baseline or pair != identity:
            raise ValueError('Comparisons must use identical baseline and evaluation identities.')
        for kind in ['baseline', 'craf']:
            measured = result[kind]
            recomputed = from_matrix(measured['confusion_matrix'])
            if recomputed['count'] != result['count']:
                raise ValueError('Confusion matrix count differs')
            for key in ['accuracy', 'balanced_accuracy', 'macro_f1']:
                if not math.isclose(measured[key], recomputed[key], abs_tol=1e-12):
                    raise ValueError(f'{filename}: {key} disagrees with confusion matrix')
            if not all(math.isfinite(measured[k]) for k in ['auc', 'cross_entropy']):
                raise ValueError('Nonfinite metric')
        records.append((name, result))
    return baseline, records


def summary(root=ROOT):
    baseline, records = load_results(root)
    columns = ['accuracy', 'balanced_accuracy', 'macro_f1', 'auc', 'cross_entropy']
    lines = ['# Experimental results - reproducible summary', '',
             'Current source: original server reports in `results/server_records/`.',
             'Recorded normalized JSON hashes of historical chat transcriptions are verified. No new GPU measurements were performed.', '',
             'Sample count: 50,705. Units: the first four metrics are percentages; cross-entropy is natural-log loss.', '',
             '| Model | Accuracy % | Balanced accuracy % | Macro-F1 % | AUC % | Cross-entropy |',
             '|---|---:|---:|---:|---:|---:|']
    all_models = [('HAMMER baseline', baseline)] + [(name, r['craf']) for name, r in records]
    for name, metrics in all_models:
        values = [f'{100 * metrics[k]:.4f}' if k != 'cross_entropy' else f'{metrics[k]:.6f}' for k in columns]
        lines.append('| ' + name + ' | ' + ' | '.join(values) + ' |')
    lines += ['', '## Differences from baseline', '',
              '| Model | Delta AUC (percentage points) | Delta accuracy (percentage points) | Delta macro-F1 (percentage points) | Net additional correct predictions | Delta CE |',
              '|---|---:|---:|---:|---:|---:|']
    base_count = from_matrix(baseline['confusion_matrix'])
    for name, result in records:
        m = result['craf']; counts = from_matrix(m['confusion_matrix'])
        lines.append(f"| {name} | {100*(m['auc']-baseline['auc']):+.4f} | "
                     f"{100*(m['accuracy']-baseline['accuracy']):+.4f} | "
                     f"{100*(m['macro_f1']-baseline['macro_f1']):+.4f} | "
                     f"{counts['correct']-base_count['correct']:+d} | "
                     f"{m['cross_entropy']-baseline['cross_entropy']:+.6f} |")
    lines += ['', '## Error types', '', '| Model | False alarms (real → fake) | Missed manipulations (fake → real) | Selected epoch (one-based) |',
              '|---|---:|---:|---:|']
    for name, result in records:
        c = from_matrix(result['craf']['confusion_matrix'])
        lines.append(f"| {name} | {c['fp']} | {c['fn']} | {result['selected_epoch']+1} |")
    lines += ['', '## Limitations', '',
              '- These are binary detection results only; CRAF grounding performance has not been measured.',
              '- AUC and cross-entropy cannot be recomputed from the aggregated confusion matrix; the JSON values are retained.',
              '- Three example-level test prediction files are now available; additional analysis is in `SERVER_SUMMARY.md`. The single-seed limitation remains.',
              '- V1 patience changed along with the learning rate; differences cannot be attributed solely to the learning rate.',
              '- Validation results for all four V2 conditions are now available; a separate validation table is in `SERVER_SUMMARY.md`.',
              '- No V2 control/penalty/decision test reports are available, so these variants receive no scores in the test table.',
              '- Test results influenced subsequent variant designs: the comparison is exploratory, not an independent final holdout evaluation.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    content = summary()
    if args.check_only:
        if (ROOT / 'results/SUMMARY.md').read_text() != content:
            raise ValueError('Result summary is stale; run scripts/summarize_results.py')
    else:
        (ROOT / 'results/SUMMARY.md').write_text(content)
    print('Verified 3 reports: identical baseline/identities, counts and matrix-derived metrics.')
