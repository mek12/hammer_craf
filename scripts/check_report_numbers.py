#!/usr/bin/env python3
"""Check every measured number quoted in the report against the original evidence files.

Each row = (report location, quoted value, evidence file, value read from the file).
A row passes if the quoted value equals the file value after rounding to the precision used
in the report. Writes results/report_number_check.json and exits non-zero on any mismatch.
Usage (repo root): python scripts/check_report_numbers.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SR = ROOT / 'results/server_records'
RUNS = {
    'v1_lr1e4': SR / 'hammer-craf/craf_results',
    'v1_lr1e5': SR / 'hammer-craf/craf_lr_1e5',
    'v2_control': SR / 'hammer-craf-v2/craf_v2_control_seed42',
    'v2_penalty': SR / 'hammer-craf-v2/craf_v2_penalty_seed42',
    'v2_decision': SR / 'hammer-craf-v2/craf_v2_decision_seed42',
    'v2_combined': SR / 'hammer-craf-v2/craf_v2_combined_seed42',
}


def load(path):
    return json.loads(Path(path).read_text())


def rel(path):
    return str(Path(path).relative_to(ROOT))


rows = []


def check(where, quoted, path, value, digits):
    ok = round(value, digits) == round(quoted, digits)
    rows.append({'where': where, 'quoted': quoted, 'file': rel(path), 'file_value': value,
                 'digits': digits, 'ok': ok})


# --- HAMMER re-training: test metrics (Table 3) ---
tm_path = SR / 'hammer_training/test_metrics.json'
tm = load(tm_path)
m = tm['metrics']
for key, quoted in [('AUC_cls', 92.67), ('ACC_cls', 85.34), ('EER_cls', 15.14), ('MAP', 86.44), ('CF1', 78.01),
                    ('OF1', 78.88), ('IOU_score', 70.61), ('IOU_ACC_50', 78.93), ('IOU_ACC_75', 61.77),
                    ('Precision_tok', 74.40), ('Recall_tok', 67.47), ('F1_tok', 70.77)]:
    check(f'Table 3 ours {key}', quoted, tm_path, m[key] * 100, 2)
check('Table 3 sample count', 50705, tm_path, tm['sample_count'], 0)
check('Sec 6 HAMMER test seconds', 269, tm_path, tm['seconds'], 0)

# --- HAMMER training log (Figure 2, Sec 6) ---
log_path = SR / 'hammer_training/log.jsonl'
log = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
check('Table 2 logged epochs', 11, log_path, len(log), 0)
check('Sec 6 HAMMER logged epoch seconds', 14907, log_path, sum(r['seconds'] for r in log), 0)
check('Figure 2 last val AUC %', 92.32, log_path, log[-1]['val']['AUC_cls'] * 100, 2)
args_path = SR / 'hammer_training/run_args.json'
args = load(args_path)
check('Table 2 batch per GPU', 64, args_path, args['batch_size'], 0)
check('Table 2 world size', 4, args_path, args['world_size'], 0)
check('Table 2 planned epochs', 25, args_path, args['epochs'], 0)
check('Table 2 seed', 42, args_path, args['seed'], 0)

# --- Splits (Table 4) ---
for split, rows_q, news_q in [('train', 208184, 56894), ('val', 22126, 7975), ('test', 50705, 18360)]:
    p = SR / f'hammer-craf/craf_features/{split}.manifest.json'
    mf = load(p)
    check(f'Table 4 {split} rows', rows_q, p, mf['count'], 0)
    check(f'Table 4 {split} unique news ids', news_q, p, len(set(mf['news_ids'])), 0)
ids = {s: set(load(SR / f'hammer-craf/craf_features/{s}.manifest.json')['news_ids']) for s in ['train', 'val', 'test']}
p = SR / 'hammer-craf/craf_features/test.manifest.json'
check('Sec 6 news-id overlap train/val', 0, p, len(ids['train'] & ids['val']), 0)
check('Sec 6 news-id overlap train/test', 0, p, len(ids['train'] & ids['test']), 0)
check('Sec 6 news-id overlap val/test', 0, p, len(ids['val'] & ids['test']), 0)

# --- Test comparison (Table 5) ---
quoted_test = {
    'baseline': (85.343, 83.851, 83.612, 92.666, 0.5261),
    'v1_lr1e4': (85.514, 83.888, 83.755, 92.861, 0.5631),
    'v1_lr1e5': (85.534, 84.056, 83.823, 92.859, 0.5647),
    'v2_combined': (85.416, 83.772, 83.643, 92.747, 0.5304),
}
errors_q = {'baseline': 7432, 'v1_lr1e4': 7345, 'v1_lr1e5': 7335, 'v2_combined': 7395}
for run in ['v1_lr1e4', 'v1_lr1e5', 'v2_combined']:
    p = RUNS[run] / 'test_comparison.json'
    t = load(p)
    for who in (['baseline', 'craf'] if run == 'v1_lr1e4' else ['craf']):
        name = 'baseline' if who == 'baseline' else run
        r = t[who]
        for label, key, scale, digits, q in zip(['acc', 'bal_acc', 'macro_f1', 'auc', 'ce'],
                                                ['accuracy', 'balanced_accuracy', 'macro_f1', 'auc', 'cross_entropy'],
                                                [100, 100, 100, 100, 1], [3, 3, 3, 3, 4], quoted_test[name]):
            check(f'Table 5 {name} {label}', q, p, r[key] * scale, digits)
        cm = r['confusion_matrix']
        check(f'Table 5 {name} errors', errors_q[name], p, cm[0][1] + cm[1][0], 0)
        if name == 'baseline':
            check('Sec 7.3 baseline false alarms', 3478, p, cm[0][1], 0)
            check('Sec 7.3 baseline misses', 3954, p, cm[1][0], 0)
        if name == 'v1_lr1e4':
            check('Sec 7.3 v1 1e-4 false alarms', 3540, p, cm[0][1], 0)
            check('Sec 7.3 v1 1e-4 misses', 3805, p, cm[1][0], 0)
        if name == 'v1_lr1e5':
            check('Sec 7.3 v1 1e-5 false alarms', 3437, p, cm[0][1], 0)
            check('Sec 7.3 v1 1e-5 misses', 3898, p, cm[1][0], 0)
        if name == 'v2_combined':
            check('Sec 7.3 v2 combined false alarms', 3568, p, cm[0][1], 0)

# --- Validation ablation (Table 7) ---
quoted_val = {'baseline': (0.923246, 0.5413), 'v1_lr1e4': (0.925029, 0.5830), 'v2_control': (0.924935, 0.5847),
              'v2_decision': (0.924980, 0.5707), 'v2_penalty': (0.923851, 0.5471), 'v2_combined': (0.923829, 0.5472)}
sel_q = {'v1_lr1e4': (2, 7), 'v2_control': (9, 17), 'v2_decision': (3, 11), 'v2_penalty': (16, 24), 'v2_combined': (13, 21)}
for run, (auc_q, ce_q) in quoted_val.items():
    src = 'v1_lr1e4' if run == 'baseline' else run
    p = RUNS[src] / 'validation_comparison.json'
    v = load(p)
    r = v['baseline' if run == 'baseline' else 'craf']
    check(f'Table 7 {run} AUC', auc_q, p, r['auc'], 6)
    check(f'Table 7 {run} CE', ce_q, p, r['cross_entropy'], 4)
    if run in sel_q:
        hp = RUNS[run] / 'history.json'
        h = load(hp)
        check(f'Table 7 {run} selected epoch (1-based)', sel_q[run][0], p, v['selected_epoch'] + 1, 0)
        check(f'Table 7 {run} completed epochs', sel_q[run][1], hp, len(h), 0)
for run, secs in [('v1_lr1e4', 45.745), ('v2_penalty', 157.439)]:
    hp = RUNS[run] / 'history.json'
    check(f'Sec 6 {run} logged head seconds', secs, hp, sum(e['seconds'] for e in load(hp)), 3)
for run in RUNS:
    rp = RUNS[run] / 'run.json'
    check(f'Sec 6 {run} seed', 42, rp, load(rp)['config']['seed'], 0)

# --- Derived analyses produced by our scripts ---
sa_path = ROOT / 'results/server_analysis.json'
sa = load(sa_path)
for run, fixed, new, pval in [('v1_lr1e4', 707, 620, 0.018), ('v1_lr1e5', 730, 633, 0.009), ('v2_combined', 368, 331, 0.173)]:
    paired = sa['runs'][run]['test']['paired_vs_baseline'] if 'paired_vs_baseline' in sa['runs'][run].get('test', {}) else None
    if paired is None:
        # locate the paired block wherever the analysis script stored it
        blob = json.dumps(sa['runs'][run])
        import re
        f = int(re.search(r'"errors_fixed": (\d+)', blob).group(1))
        n = int(re.search(r'"new_errors": (\d+)', blob).group(1))
        pv = float(re.search(r'"mcnemar_exact_two_sided_p": ([0-9.e-]+)', blob).group(1))
    else:
        f, n, pv = paired['errors_fixed'], paired['new_errors'], paired['mcnemar_exact_two_sided_p']
    check(f'Table 6 {run} fixed', fixed, sa_path, f, 0)
    check(f'Table 6 {run} new errors', new, sa_path, n, 0)
    check(f'Table 6 {run} McNemar p', pval, sa_path, pv, 3)
bs_path = ROOT / 'results/bootstrap_uncertainty.json'
bs = load(bs_path)['runs']
for run, auc_ci in [('v1_lr1e4', (0.148, 0.246)), ('v1_lr1e5', (0.142, 0.242)), ('v2_combined', (0.058, 0.104))]:
    check(f'Table 6 {run} AUC CI low', auc_ci[0], bs_path, bs[run]['auc']['ci95'][0] * 100, 3)
    check(f'Table 6 {run} AUC CI high', auc_ci[1], bs_path, bs[run]['auc']['ci95'][1] * 100, 3)
ca_path = ROOT / 'results/calibration_analysis.json'
ca = load(ca_path)
for run, conf_err, ece in [('hammer', 1999, 0.1010), ('v1_lr1e4', 2325, 0.1048), ('v1_lr1e5', 2322, 0.1048), ('v2_combined', 2059, 0.1023)]:
    check(f'Table 8 {run} errors conf>0.99', conf_err, ca_path, ca[run]['confident_errors_gt099'], 0)
    check(f'Table 8 {run} ECE', ece, ca_path, ca[run]['ece'], 4)
mr_path = ROOT / 'results/model_resources.json'
mr = {r['model']: r for r in load(mr_path)['rows']}
check('Sec 5 CRAF parameters', 922117, mr_path, mr['v1']['trainable_parameters'], 0)
check('Sec 7.5 v2-decision parameters', 922885, mr_path, mr['v2_decision']['trainable_parameters'], 0)

failed = [r for r in rows if not r['ok']]
out = {'checked': len(rows), 'failed': len(failed), 'rows': rows}
(ROOT / 'results/report_number_check.json').write_text(json.dumps(out, indent=1))
for r in rows:
    print(('OK   ' if r['ok'] else 'FAIL ') + f"{r['where']:<45} quoted={r['quoted']:<10} file={r['file_value']:.6g}  <- {r['file']}")
print(f'\n{len(rows) - len(failed)}/{len(rows)} checks passed')
sys.exit(1 if failed else 0)
