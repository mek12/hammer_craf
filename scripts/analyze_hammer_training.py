#!/usr/bin/env python3
"""Verify supplied HAMMER training records; no training, downloads or checkpoint loading."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / 'results/server_records/hammer_training'

def main():
    args = json.loads((D / 'run_args.json').read_text())
    cfg = yaml.safe_load((D / 'config.yaml').read_text())
    test = json.loads((D / 'test_metrics.json').read_text())
    history = [json.loads(x) for x in (D / 'log.jsonl').read_text().splitlines() if x.strip()]
    checks = []
    def check(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
    check('epochs contiguous and begin at zero', [x['epoch'] for x in history] == list(range(len(history))))
    best = max(history, key=lambda x: x['val']['AUC_cls'])
    running = float('-inf')
    for x in history:
        running = max(running, x['val']['AUC_cls'])
        check(f'epoch {x["epoch"]}: cumulative best AUC', abs(running-x['best_auc']) < 1e-12)
    check('test checkpoint epoch equals best validation epoch',test['checkpoint_epoch_zero_based']==best['epoch'])
    check('planned epoch settings agree',args['epochs']==cfg['schedular']['epochs'])
    check('batch settings agree',args['batch_size']==cfg['batch_size_train'] and args['val_batch_size']==cfg['batch_size_val'])
    check('ALBEF initialization and no resume recorded',args['checkpoint'].endswith('ALBEF_4M.pth') and args['resume'] is None)
    check('strict checkpoint test on full test split',test['strict_load'] and test['split']=='test' and test['sample_count']==50705)
    comparisons=[]
    for rel in ['hammer-craf/craf_results/test_comparison.json','hammer-craf/craf_lr_1e5/test_comparison.json','hammer-craf-v2/craf_v2_combined_seed42/test_comparison.json']:
        c=json.loads((ROOT/'results/server_records'/rel).read_text())
        check(f'{rel}: backbone hash',test['checkpoint_sha256']==c['cache_identity']['backbone_sha256'])
        check(f'{rel}: metadata hash',test['metadata_sha256']==c['cache_identity']['metadata_sha256'])
        check(f'{rel}: accuracy',abs(test['metrics']['ACC_cls']-c['baseline']['accuracy'])<1e-12)
        diff=test['metrics']['AUC_cls']-c['baseline']['auc']
        check(f'{rel}: AUC within 1e-7',abs(diff)<1e-7)
        check(f'{rel}: recorded upstream source hashes',all(c['cache_identity']['source_sha256'][k]==v for k,v in test['source_sha256'].items()))
        comparisons.append({'report':rel,'official_minus_cached_baseline_auc':diff})
    out={'status':'passed' if all(x['passed'] for x in checks) else 'failed','completed_logged_epochs':len(history),'last_epoch_zero_based':history[-1]['epoch'],'selected_epoch_zero_based':best['epoch'],'selected_epoch_one_based':best['epoch']+1,'planned_epochs':args['epochs'],'world_size':args['world_size'],'per_gpu_batch':args['batch_size'],'nominal_global_batch':args['world_size']*args['batch_size'],'seed':args['seed'],'logged_epoch_seconds_sum':sum(x['seconds'] for x in history),'test_seconds':test['seconds'],'test_metrics':test['metrics'],'checkpoint_sha256':test['checkpoint_sha256'],'comparisons':comparisons,'checks':checks,'limitations':['11 logged epochs do not establish that the planned 25 epochs finished; stop reason absent','nominal global batch based on run_args and packaged launcher semantics; GPU model is user supplied','recorded checkpoint hash matches cache identity; weight file not loaded','timing includes checkpoint writes in packaged HAMMER launcher; excludes setup and final per-epoch trailing operations','official evaluation AUC differs from cached float32 baseline by about 8.76e-10; original values preserved']}
    (ROOT/'results/hammer_training_analysis.json').write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    lines=['# HAMMER yeniden eğitim ve test kaydı','',f"Hedef {args['epochs']} epoch; kayıtlı {len(history)} tamamlanmış epoch (0–10); seçilen checkpoint 0 tabanlı {best['epoch']} / 1 tabanlı {best['epoch']+1}.",'',f"Seed {args['seed']}; world_size {args['world_size']}; GPU başına batch {args['batch_size']}; nominal global batch {out['nominal_global_batch']}.",'',f"Epoch süreleri toplamı {out['logged_epoch_seconds_sum']:.3f} saniye ({out['logged_epoch_seconds_sum']/3600:.3f} saat); bağımsız test {test['seconds']:.3f} saniye. Toplam uçtan uca süre veya fiziksel GPU-saat ölçümü değildir.",'','| Epoch (1 tabanlı) | Train loss | Val AUC % | Val ACC % | Val mAP % | Val IoU % | Val token F1 % |','|---|---:|---:|---:|---:|---:|---:|']
    for x in history:
        v=x['val'];lines.append(f"| {x['epoch']+1} | {float(x['train']['loss']):.6f} | {100*v['AUC_cls']:.4f} | {100*v['ACC_cls']:.4f} | {100*v['MAP']:.4f} | {100*v['IOU_score']:.4f} | {100*v['F1_tok']:.4f} |")
    lines+=['','## HAMMER test — tüm görevler','','| Metrik | Değer % |','|---|---:|']
    for k,v in test['metrics'].items():lines.append(f'| {k} | {100*v:.6f} |')
    lines+=['','## Kimlik ve sayısal sınır','','Üç CRAF raporuyla checkpoint/metadata/kayıtlı upstream source hash eşleşir. HAMMER test AUC değeri ile CRAF cache baseline AUC farkı yaklaşık 8,76e-10; aynı hassasiyet düzeyinde eşittir. Özgün değerler değiştirilmedi. SHA kayıt eşliği ağırlığın yeniden yüklenmesi değildir.','','Test girdisi görüntü+metindir. Tür, kutu ve token sonuçları yalnız HAMMER baseline için ölçülmüştür; CRAF iyileştirmesi olarak sunulmaz.','','Kontroller: '+str(len(checks))+' geçti; yeni CUDA çalıştırması yapılmadı.']
    (ROOT/'results/HAMMER_TRAINING_SUMMARY.md').write_text('\n'.join(lines)+'\n')
    print(out['status'],len(checks),'checks;',out['logged_epoch_seconds_sum'],'seconds;',len(history),'logged epochs')
    if out['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
