#!/usr/bin/env python3
"""Audit supplied server evidence only; never import trainers or load/train models."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import scipy
from scipy.stats import binomtest
import sklearn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, log_loss, roc_auc_score

from evidence import evidence_path, normalized_sha, transcription_record

ROOT = Path(__file__).absolute().parent.parent
SPECS = {
    'v1_lr1e4': ('hammer-craf', 'craf_results'),
    'v1_lr1e5': ('hammer-craf', 'craf_lr_1e5'),
    **{f'v2_{x}': ('hammer-craf-v2', f'craf_v2_{x}_seed42') for x in ('control', 'penalty', 'decision', 'combined')},
}
TRANSCRIPTIONS = {'v1_lr1e4': 'craf_v1_lr1e4_test.json', 'v1_lr1e5': 'craf_v1_lr1e5_test.json', 'v2_combined': 'craf_v2_combined_test.json'}
SOURCES = {}
MISSING = []
CHECKS = []


def locate(relative, required=True):
    try:
        p = evidence_path(relative)
    except FileNotFoundError:
        p = None
    if p is not None and p.is_file():
        SOURCES[str(relative)] = str(p.relative_to(ROOT))
        return p
    if required:
        MISSING.append(str(relative))
    return None


def read(relative):
    p = locate(relative)
    return json.loads(p.read_text()) if p else None


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def check(name, value):
    CHECKS.append({'name': name, 'passed': bool(value)})
    return bool(value)


def metrics(y, p):
    # Only p_fake is exported. Original independently rounded p_real is unavailable.
    p = np.asarray(p, dtype=np.float32)
    probs = np.column_stack((np.float32(1) - p, p)).astype(np.float32)
    pred = (p > np.float32(.5)).astype(np.int64)  # argmax chooses real on exact ties
    return dict(accuracy=float(accuracy_score(y, pred)), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, average='macro')), auc=float(roc_auc_score(y, p)),
                cross_entropy=float(log_loss(y, probs, labels=[0, 1])), confusion_matrix=confusion_matrix(y, pred, labels=[0, 1]).tolist())


def transitions(y, old, new):
    a, b = old == y, new == y
    fixed, introduced = int(np.sum(~a & b)), int(np.sum(a & ~b))
    changed = int(np.sum(old != new))
    return dict(changed_cases=changed, errors_fixed=fixed, new_errors=introduced, net_correct=fixed-introduced,
                both_correct=int(np.sum(a & b)), both_wrong=int(np.sum(~a & ~b)),
                prediction_transition_matrix=confusion_matrix(old, new, labels=[0, 1]).tolist(),
                accuracy_delta_pp=100*(fixed-introduced)/len(y),
                by_true_label={str(k): dict(support=int(np.sum(y == k)), errors_fixed=int(np.sum(~a & b & (y == k))),
                                          new_errors=int(np.sum(a & ~b & (y == k)))) for k in (0, 1)},
                mcnemar_exact_two_sided_p=float(binomtest(fixed, fixed+introduced, .5).pvalue) if fixed+introduced else 1.,
                p_value_qualification='Conditional on independent examples; news clusters cannot be reconstructed from sorted unique manifest IDs. Exploratory, no cluster-independent significance claim.')


def main():
    out = {'status': 'in_progress', 'environment': {'python_executable': str(Path(sys.executable).absolute()), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'scipy': scipy.__version__}, 'runs': {}, 'manifests': {}}
    # Write a reviewable first artifact before expensive prediction parsing.
    emit(out)
    manifests = {}
    for split in ('train', 'val', 'test'):
        rel = Path('hammer-craf/craf_features') / f'{split}.manifest.json'
        m = read(rel)
        if m is None:
            continue
        manifests[split] = m
        path = locate(rel)
        out['manifests'][split] = dict(count=m['count'], unique_news_ids=len(set(m['news_ids'])), sha256=sha(path), identity=m['identity'],
                                      shards=m['shards'], shards_present=[s['file'] for s in m['shards'] if (path.parent/s['file']).is_file()])
        check(f'{split}: news IDs unique', len(m['news_ids']) == len(set(m['news_ids'])))
    out['news_id_overlaps'] = {}
    for a, b in itertools.combinations(manifests, 2):
        overlap = len(set(manifests[a]['news_ids']) & set(manifests[b]['news_ids']))
        out['news_id_overlaps'][f'{a}/{b}'] = overlap
        check(f'{a}/{b}: disjoint news IDs', overlap == 0)
    for name, parts in SPECS.items():
        rel = Path(*parts)
        run, history, val = [read(rel/f'{x}.json') for x in ('run', 'history', 'validation_comparison')]
        if any(x is None for x in (run, history, val)):
            continue
        c = run['config']
        best_row = max(history, key=lambda x: x['validation'][c['selection_metric']])
        best = -float('inf')
        bad = 0
        patience_valid = True
        for i, row in enumerate(history):
            score = row['validation'][c['selection_metric']]
            improved = score > best
            best, bad = max(best, score), 0 if improved else bad+1
            patience_valid &= row['epoch'] == i and row['best'] == best and row['patience_count'] == bad
        check(f'{name}: history best/patience sequence', patience_valid)
        check(f'{name}: selected epoch and validation', val['selected_epoch'] == best_row['epoch'] and val['craf'] == best_row['validation'])
        check(f'{name}: validation baseline', run['baseline_validation'] == val['baseline'])
        check(f'{name}: no test used for epoch selection flag', val.get('test_used') is False)
        for split in ('train', 'val'):
            check(f'{name}: {split} manifest SHA256', c[f'{split}_manifest_sha256'] == out['manifests'][split]['sha256'])
            identity = {k: v for k, v in manifests[split]['identity'].items() if k not in ('split', 'metadata_sha256')}
            check(f'{name}: {split} backbone identity', c['backbone'] == identity)
        source_names = {'head_source_sha256': 'craf_residual_v2.py' if name.startswith('v2') else 'craf_residual.py',
                        'trainer_source_sha256': 'hammer_craf_v2_server.py' if name.startswith('v2') else 'hammer_craf_server.py',
                        'helper_source_sha256': 'hammer_craf_server.py'}
        source_checks = {}
        for key, filename in source_names.items():
            if key in c:
                path = locate(Path(parts[0])/filename)
                source_checks[key] = bool(path and sha(path) == c[key])
                check(f'{name}: {key}', source_checks[key])
        stop = bad >= c['patience'] and len(history) < run['target_epochs']
        check(f'{name}: stopping correspondence', bad >= c['patience'] or len(history) == run['target_epochs'])
        out['runs'][name] = dict(config=c, target_epochs=run['target_epochs'], epochs_completed=len(history),
                                best_epoch_zero_based=best_row['epoch'], best_epoch_one_based=best_row['epoch']+1,
                                final_patience_count=bad, early_stopped=stop, logged_epoch_seconds_sum=sum(x['seconds'] for x in history),
                                validation=val, history=history, source_hash_checks=source_checks)
    low, control = out['runs']['v1_lr1e5'], out['runs']['v2_control']
    out['control_comparison'] = dict(selected_validation_identical=all(low['validation'][k] == control['validation'][k] for k in ('split','selected_epoch','baseline','craf','test_used')),
        full_epoch_validation_sequence_identical=[r['validation'] for r in low['history']] == [r['validation'] for r in control['history']],
        epoch_count_identical=low['epochs_completed'] == control['epochs_completed'])
    for k, v in out['control_comparison'].items(): check(f'v2 control/v1 lowLR: {k}', v)
    common = ('learning_rate','weight_decay','batch_size','seed','patience','selection_metric','backbone','train_manifest_sha256','val_manifest_sha256')
    ref = out['runs']['v2_control']['config']
    out['v2_matched_settings'] = {k: ref[k] for k in common}
    for name, expected in {'control': (0., False), 'penalty': (.01, False), 'decision': (0., True), 'combined': (.01, True)}.items():
        r = out['runs'][f'v2_{name}']; c = r['config']
        check(f'v2 {name}: matched settings', all(c[k] == ref[k] for k in common) and r['target_epochs'] == 30 and c['head']['hidden_dim'] == 256 and c['head']['dropout'] == .2 and c['head']['feature_dim'] == 768)
        check(f'v2 {name}: lambda/decision flag', (c['correction_lambda'], c['head']['condition_on_decision']) == expected)
    out['user_transcription_comparisons'] = {}
    arrays = {}
    for name, filename in TRANSCRIPTIONS.items():
        rel = Path(*SPECS[name]); report = read(rel/'test_comparison.json')
        old = transcription_record(filename)
        equality = report is not None and normalized_sha(report) == old['normalized_json_sha256']
        out['user_transcription_comparisons'][name] = dict(normalized_json_hash_equal=equality, includes_cache_and_checkpoint_identities=True,
            fingerprint_file='results/user_reported/TRANSCRIPTIONS.json', historical_name=filename,
            original_file=SOURCES.get(str(rel/'test_comparison.json')))
        check(f'{name}: recorded transcription fingerprint matches original', equality)
        if report is None: continue
        check(f'{name}: test cache identity', report['cache_identity'] == manifests['test']['identity'])
        check(f'{name}: test selected epoch', report['selected_epoch'] == out['runs'][name]['best_epoch_zero_based'])
        path = locate(rel/'test_comparison_predictions.jsonl')
        if not path: continue
        rows = [json.loads(line) for line in path.open() if line.strip()]
        idx = np.array([r['index'] for r in rows]); y = np.array([r['target'] for r in rows])
        base = np.array([r['baseline_fake_probability'] for r in rows], dtype=np.float32)
        p = np.array([r['craf_fake_probability'] for r in rows], dtype=np.float32)
        check(f'{name}: count 50705 and manifest/report agree', len(rows) == report['count'] == manifests['test']['count'] == 50705)
        check(f'{name}: exact sequential indices', np.array_equal(idx, np.arange(50705)))
        check(f'{name}: valid binary labels and finite probabilities', np.isin(y, [0,1]).all() and all(np.isfinite(x).all() and ((x >= 0)&(x <= 1)).all() for x in (base,p)))
        recomputed, differences = {}, {}
        for label, probs in [('baseline',base), ('craf',p)]:
            recomputed[label] = metrics(y, probs)
            differences[label] = {}
            for metric, value in recomputed[label].items():
                if metric == 'confusion_matrix':
                    differences[label][metric] = (np.array(value)-np.array(report[label][metric])).tolist()
                    check(f'{name}/{label}: confusion exact', value == report[label][metric])
                else:
                    diff = value-report[label][metric]
                    differences[label][metric] = diff
                    check(f'{name}/{label}: {metric} tolerance', abs(diff) <= (5e-6 if metric == 'cross_entropy' else 1e-12))
        paired = transitions(y, (base > .5).astype(int), (p > .5).astype(int))
        check(f'{name}: changed equals fixed + introduced', paired['changed_cases'] == paired['errors_fixed']+paired['new_errors'])
        out['runs'][name]['test'] = dict(original_report=report, recomputed=recomputed, differences_recomputed_minus_report=differences,
            label_support=np.bincount(y,minlength=2).tolist(), exact_paired=paired, ce_reconstruction_is_approximate=True, ce_tolerance=5e-6, ce_within_1e6={k: abs(differences[k]['cross_entropy']) <= 1e-6 for k in ('baseline','craf')}, possible_threshold_ambiguities={k: int(np.sum(v == np.float32(.5))) for k,v in [('baseline',base),('craf',p)]}, probability_half_ties={'baseline':int(np.sum(base==.5)), 'craf':int(np.sum(p==.5))})
        arrays[name] = (idx,y,base,p)
    out['cross_run_test_alignment'] = {}
    out['between_craf_paired'] = {}
    for a,b in itertools.combinations(arrays,2):
        aa,bb=arrays[a],arrays[b]
        equal = all(np.array_equal(aa[i],bb[i]) for i in range(3))
        out['cross_run_test_alignment'][f'{a}/{b}'] = dict(indices_labels_baseline_scores_exact=equal)
        check(f'{a}/{b}: indices/labels/baseline scores align',equal)
        if equal: out['between_craf_paired'][f'{a}->{b}'] = transitions(aa[1],(aa[3]>.5).astype(int),(bb[3]>.5).astype(int))
    out['test_unavailable_runs'] = [k for k in SPECS if k not in arrays]
    out['ce_reconstruction_tolerance'] = {'acceptance': 5e-6, 'strict_reference': 1e-6, 'reason': 'Only fake-class float32 probabilities are exported; complementary original probability is not available. All residuals retained.'}
    out['limitations'] = [
        'All six runs use seed 42: no estimate of training-seed variability.',
        'Test-guided exploration remains: these are exploratory comparisons, not an untouched confirmatory final test.',
        'Manifest news_ids are sorted unique groups, not prediction-row-to-news mappings; no news-cluster-independent p-value or cluster bootstrap is supported.',
        'Only binary detection is evaluated; no generic SOTA, robustness, localization or other-head improvement claim.',
        'Feature .pt shards and model checkpoints are absent. Their manifest/report hashes are recorded identities, not independently verified file hashes. No new training or checkpoint execution.',
        'Logged epoch seconds include head training and per-epoch validation only; cache loading, initial validation and feature extraction are excluded. They are not total wall-clock time.',
        'CE reconstructs [1-p_fake,p_fake] in float32 with sklearn dtype-epsilon clipping. Original independently rounded two-class probabilities are unavailable; residual differences are reported with 5e-6 acceptance tolerance (the largest observed residual is 1.28e-6); separate flags record the stricter 1e-6 threshold, never bitwise equality.',
        'Decisions use p_fake > 0.5 (argmax class 0 on exact ties); probabilities rounded to 0.5 could hide a nonzero raw-logit margin. Tie/possible ambiguity counts and confusion differences are reported rather than modifying predictions.',
        'AUC uses exact stored float32 scores and sklearn tie handling; other numeric tolerances are 1e-12; confusion and transition counts are exact.',
    ]
    out['checks'] = CHECKS
    out['missing_required_files'] = MISSING
    out['input_sources'] = SOURCES
    out['status'] = 'passed' if all(c['passed'] for c in CHECKS) and not MISSING else 'needs_review'
    emit(out)
    print(json.dumps({'status':out['status'], 'checks':len(CHECKS),'failed':[c['name'] for c in CHECKS if not c['passed']], 'missing':MISSING},indent=2))
    return 0 if out['status']=='passed' else 1


def emit(out):
    (ROOT/'results/server_analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    lines = ['# Sunucu sonuç analizi', '', f"Durum: {out['status']}", '', 'Kaynak: results/SERVER_EVIDENCE_INDEX.json ile çözümlenen tekil dosyalar. Yeni eğitim yapılmaz.', '']
    if out.get('runs'):
        lines += ['## Doğrulama (oranlar; yüksek iyi, CE düşük iyi)', '', '| Model | ACC | Dengeli ACC | Makro F1 | AUC | CE | Tamamlanan epoch | En iyi (0/1) | Loglanan epoch sn |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        base = next(iter(out['runs'].values()))['validation']['baseline']
        lines += [f"| HAMMER baseline | {base['accuracy']:.9f} | {base['balanced_accuracy']:.9f} | {base['macro_f1']:.9f} | {base['auc']:.9f} | {base['cross_entropy']:.9f} | — | — | — |"]
        for name,r in out['runs'].items():
            m=r['validation']['craf']
            lines += [f"| {name} | {m['accuracy']:.9f} | {m['balanced_accuracy']:.9f} | {m['macro_f1']:.9f} | {m['auc']:.9f} | {m['cross_entropy']:.9f} | {r['epochs_completed']} | {r['best_epoch_zero_based']}/{r['best_epoch_one_based']} | {r['logged_epoch_seconds_sum']:.3f} |"]
        lines += ['', 'Epoch toplamları yalnızca kayıtlı head eğitim ve epoch doğrulama döngüsüdür; toplam çalışma süresi değildir.', '', '## Test (50.705 örnek; özgün rapor metrikleri)' , '', '| Model | ACC | Dengeli ACC | Makro F1 | AUC | CE | Düzeltilen hata | Yeni hata | Net | Değişen karar |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        first=True
        for name,r in out['runs'].items():
            if 'test' not in r: continue
            t=r['test']; m=t['original_report']['craf']; p=t['exact_paired']
            if first:
                b=t['original_report']['baseline']; lines += [f"| HAMMER baseline | {b['accuracy']:.9f} | {b['balanced_accuracy']:.9f} | {b['macro_f1']:.9f} | {b['auc']:.9f} | {b['cross_entropy']:.9f} | — | — | — | — |"]
                first=False
            lines += [f"| {name} | {m['accuracy']:.9f} | {m['balanced_accuracy']:.9f} | {m['macro_f1']:.9f} | {m['auc']:.9f} | {m['cross_entropy']:.9f} | {p['errors_fixed']} | {p['new_errors']} | {p['net_correct']} | {p['changed_cases']} |"]
        lines += ['', 'Control: V2 control ile V1 düşük LR seçilmiş doğrulama ve tüm epoch doğrulama dizileri birebir aynıdır. Dört V2 koşulunun LR, ağırlık azaltma, batch, seed, patience, seçim ölçütü, backbone ve manifestleri eşleştirilmiştir; λ/decision bayrakları ayrıca denetlenmiştir.', '', '## Manifestler', '', '| Bölüm | Örnek | Benzersiz haber | Sağlanan .pt |', '|---|---:|---:|---:|']
        for k,m in out['manifests'].items(): lines += [f"| {k} | {m['count']} | {m['unique_news_ids']} | {len(m['shards_present'])} |"]
        lines += ['', 'Haber kimliklerinin bölümler arası kesişimleri: '+json.dumps(out.get('news_id_overlaps',{})), '', 'Üç eski kullanıcı aktarımının kayıtlı normalize JSON hashleri, özgün raporlarla cache/checkpoint kimlikleri dahil karşılaştırıldı: '+json.dumps(out.get('user_transcription_comparisons',{}),ensure_ascii=False), '', '## Yeniden hesaplama ve sınırlar', '']
        for name,r in out['runs'].items():
            if 'test' in r:
                t=r['test']; lines += [f"- {name}: etiket desteği {t['label_support']}; yaklaşık yeniden kurulan CE baseline={t['recomputed']['baseline']['cross_entropy']:.12g}, CRAF={t['recomputed']['craf']['cross_entropy']:.12g}; CE farkları (yeniden−rapor) baseline={t['differences_recomputed_minus_report']['baseline']['cross_entropy']:.12g}, CRAF={t['differences_recomputed_minus_report']['craf']['cross_entropy']:.12g}."]
        lines += ['', 'CE için kabul eşiği 5×10⁻⁶; katı 10⁻⁶ eşiğini V1 CRAF yeniden hesaplamaları az miktarda aşar. Float32 tamamlayıcı olasılık yeniden kurulduğu için iki sınıflı özgün olasılıklarla bit düzeyinde eşitlik iddia edilmez. ACC/dengeli ACC/AUC ve karışıklık sayıları eşleşir; F1 farkı yalnızca makine yuvarlaması düzeyindedir. 0.5 olasılık eşitliği üç dosyada da yoktur.', '', 'V2 combined, HAMMER’a göre 37 net ek doğru karar verir; V1 düşük LR’ye göre 60 daha az doğru karar verir. V2 combined CE değeri V1’e göre daha düşüktür, ancak HAMMER baseline CE değerinden hâlâ yüksektir; genel bir kalibrasyon kazanımı sonucu çıkarılmaz. Combined için gerçek sınıf 0: 120 düzeltilen / 210 yeni hata; sınıf 1: 248 düzeltilen / 121 yeni hata. Etiket destekleri 16.876 ve 33.829’dur.', '', '## Erken durdurma', '', '| Koşul | Son patience | Yapılandırılmış patience | Erken durdu |', '|---|---:|---:|---|']
        for name,r in out['runs'].items():
            lines += [f"| {name} | {r['final_patience_count']} | {r['config']['patience']} | {r['early_stopped']} |"]
        lines += ['', 'Test raporu/prediction sağlanmayan koşullar: '+', '.join(out.get('test_unavailable_runs',[]))+'. Bu koşullar için test sonucu tahmin edilmedi.', '', '## Sınırlılıklar', ''] + ['- '+x for x in out.get('limitations',[])]
        lines += ['', 'Eksik zorunlu dosyalar: '+json.dumps(out.get('missing_required_files',[])), '', 'Başarısız kontroller: '+json.dumps([c['name'] for c in out.get('checks',[]) if not c['passed']],ensure_ascii=False), '', 'Tam sayılar, kaynak SHA256 değerleri, epoch geçmişleri, her metrik farkı, eşleşmiş sınıf geçişleri ve koşullu McNemar değerleri: results/server_analysis.json.']
    text='\n'.join(lines)+'\n'
    (ROOT/'results/SERVER_SUMMARY.md').write_text(text)


if __name__ == '__main__':
    sys.exit(main())
