# Sunucu sonuç analizi

Durum: passed

Kaynak: results/SERVER_EVIDENCE_INDEX.json ile çözümlenen tekil dosyalar. Yeni eğitim yapılmaz.

## Doğrulama (oranlar; yüksek iyi, CE düşük iyi)

| Model | ACC | Dengeli ACC | Makro F1 | AUC | CE | Tamamlanan epoch | En iyi (0/1) | Loglanan epoch sn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HAMMER baseline | 0.848955979 | 0.833899611 | 0.832081765 | 0.923246474 | 0.541329179 | — | — | — |
| v1_lr1e4 | 0.851170569 | 0.834948712 | 0.834081078 | 0.925029376 | 0.582958190 | 7 | 1/2 | 45.745 |
| v1_lr1e5 | 0.851893700 | 0.837101132 | 0.835321440 | 0.924935432 | 0.584701314 | 17 | 8/9 | 109.325 |
| v2_control | 0.851893700 | 0.837101132 | 0.835321440 | 0.924935432 | 0.584701314 | 17 | 8/9 | 116.408 |
| v2_penalty | 0.851758113 | 0.835195598 | 0.834602124 | 0.923851341 | 0.547143180 | 24 | 15/16 | 157.439 |
| v2_decision | 0.850899394 | 0.835301318 | 0.833992913 | 0.924979650 | 0.570732362 | 11 | 2/3 | 74.972 |
| v2_combined | 0.851170569 | 0.834915927 | 0.834070343 | 0.923829359 | 0.547165487 | 21 | 12/13 | 140.049 |

Epoch toplamları yalnızca kayıtlı head eğitim ve epoch doğrulama döngüsüdür; toplam çalışma süresi değildir.

## Test (50.705 örnek; özgün rapor metrikleri)

| Model | ACC | Dengeli ACC | Makro F1 | AUC | CE | Düzeltilen hata | Yeni hata | Net | Değişen karar |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HAMMER baseline | 0.853426684 | 0.838513272 | 0.836121479 | 0.926660032 | 0.526090820 | — | — | — | — |
| v1_lr1e4 | 0.855142491 | 0.838878596 | 0.837545484 | 0.928614576 | 0.563057452 | 707 | 620 | 87 | 1327 |
| v1_lr1e5 | 0.855339710 | 0.840555707 | 0.838225576 | 0.928592289 | 0.564652052 | 730 | 633 | 97 | 1363 |
| v2_combined | 0.854156395 | 0.837723851 | 0.836425308 | 0.927468473 | 0.530418579 | 368 | 331 | 37 | 699 |

Control: V2 control ile V1 düşük LR seçilmiş doğrulama ve tüm epoch doğrulama dizileri birebir aynıdır. Dört V2 koşulunun LR, ağırlık azaltma, batch, seed, patience, seçim ölçütü, backbone ve manifestleri eşleştirilmiştir; λ/decision bayrakları ayrıca denetlenmiştir.

## Manifestler

| Bölüm | Örnek | Benzersiz haber | Sağlanan .pt |
|---|---:|---:|---:|
| train | 208184 | 56894 | 0 |
| val | 22126 | 7975 | 0 |
| test | 50705 | 18360 | 0 |

Haber kimliklerinin bölümler arası kesişimleri: {"train/val": 0, "train/test": 0, "val/test": 0}

Üç eski kullanıcı aktarımının kayıtlı normalize JSON hashleri, özgün raporlarla cache/checkpoint kimlikleri dahil karşılaştırıldı: {"v1_lr1e4": {"normalized_json_hash_equal": true, "includes_cache_and_checkpoint_identities": true, "fingerprint_file": "results/user_reported/TRANSCRIPTIONS.json", "historical_name": "craf_v1_lr1e4_test.json", "original_file": "results/server_records/hammer-craf/craf_results/test_comparison.json"}, "v1_lr1e5": {"normalized_json_hash_equal": true, "includes_cache_and_checkpoint_identities": true, "fingerprint_file": "results/user_reported/TRANSCRIPTIONS.json", "historical_name": "craf_v1_lr1e5_test.json", "original_file": "results/server_records/hammer-craf/craf_lr_1e5/test_comparison.json"}, "v2_combined": {"normalized_json_hash_equal": true, "includes_cache_and_checkpoint_identities": true, "fingerprint_file": "results/user_reported/TRANSCRIPTIONS.json", "historical_name": "craf_v2_combined_test.json", "original_file": "results/server_records/hammer-craf-v2/craf_v2_combined_seed42/test_comparison.json"}}

## Yeniden hesaplama ve sınırlar

- v1_lr1e4: etiket desteği [16876, 33829]; yaklaşık yeniden kurulan CE baseline=0.5260906492, CRAF=0.563056176309; CE farkları (yeniden−rapor) baseline=-1.70617655781e-07, CRAF=-1.27567454133e-06.
- v1_lr1e5: etiket desteği [16876, 33829]; yaklaşık yeniden kurulan CE baseline=0.5260906492, CRAF=0.5646509126; CE farkları (yeniden−rapor) baseline=-1.70617655781e-07, CRAF=-1.13955076253e-06.
- v2_combined: etiket desteği [16876, 33829]; yaklaşık yeniden kurulan CE baseline=0.5260906492, CRAF=0.530418418911; CE farkları (yeniden−rapor) baseline=-1.70617655781e-07, CRAF=-1.59954335444e-07.

CE için kabul eşiği 5×10⁻⁶; katı 10⁻⁶ eşiğini V1 CRAF yeniden hesaplamaları az miktarda aşar. Float32 tamamlayıcı olasılık yeniden kurulduğu için iki sınıflı özgün olasılıklarla bit düzeyinde eşitlik iddia edilmez. ACC/dengeli ACC/AUC ve karışıklık sayıları eşleşir; F1 farkı yalnızca makine yuvarlaması düzeyindedir. 0.5 olasılık eşitliği üç dosyada da yoktur.

V2 combined, HAMMER’a göre 37 net ek doğru karar verir; V1 düşük LR’ye göre 60 daha az doğru karar verir. V2 combined CE değeri V1’e göre daha düşüktür, ancak HAMMER baseline CE değerinden hâlâ yüksektir; genel bir kalibrasyon kazanımı sonucu çıkarılmaz. Combined için gerçek sınıf 0: 120 düzeltilen / 210 yeni hata; sınıf 1: 248 düzeltilen / 121 yeni hata. Etiket destekleri 16.876 ve 33.829’dur.

## Erken durdurma

| Koşul | Son patience | Yapılandırılmış patience | Erken durdu |
|---|---:|---:|---|
| v1_lr1e4 | 5 | 5 | True |
| v1_lr1e5 | 8 | 8 | True |
| v2_control | 8 | 8 | True |
| v2_penalty | 8 | 8 | True |
| v2_decision | 8 | 8 | True |
| v2_combined | 8 | 8 | True |

Test raporu/prediction sağlanmayan koşullar: v2_control, v2_penalty, v2_decision. Bu koşullar için test sonucu tahmin edilmedi.

## Sınırlılıklar

- All six runs use seed 42: no estimate of training-seed variability.
- Test-guided exploration remains: these are exploratory comparisons, not an untouched confirmatory final test.
- Manifest news_ids are sorted unique groups, not prediction-row-to-news mappings; no news-cluster-independent p-value or cluster bootstrap is supported.
- Only binary detection is evaluated; no generic SOTA, robustness, localization or other-head improvement claim.
- Feature .pt shards and model checkpoints are absent. Their manifest/report hashes are recorded identities, not independently verified file hashes. No new training or checkpoint execution.
- Logged epoch seconds include head training and per-epoch validation only; cache loading, initial validation and feature extraction are excluded. They are not total wall-clock time.
- CE reconstructs [1-p_fake,p_fake] in float32 with sklearn dtype-epsilon clipping. Original independently rounded two-class probabilities are unavailable; residual differences are reported with 5e-6 acceptance tolerance (the largest observed residual is 1.28e-6); separate flags record the stricter 1e-6 threshold, never bitwise equality.
- Decisions use p_fake > 0.5 (argmax class 0 on exact ties); probabilities rounded to 0.5 could hide a nonzero raw-logit margin. Tie/possible ambiguity counts and confusion differences are reported rather than modifying predictions.
- AUC uses exact stored float32 scores and sklearn tie handling; other numeric tolerances are 1e-12; confusion and transition counts are exact.

Eksik zorunlu dosyalar: []

Başarısız kontroller: []

Tam sayılar, kaynak SHA256 değerleri, epoch geçmişleri, her metrik farkı, eşleşmiş sınıf geçişleri ve koşullu McNemar değerleri: results/server_analysis.json.
