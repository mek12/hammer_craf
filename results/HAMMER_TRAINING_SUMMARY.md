# HAMMER yeniden eğitim ve test kaydı

Hedef 25 epoch; kayıtlı 11 tamamlanmış epoch (0–10); seçilen checkpoint 0 tabanlı 10 / 1 tabanlı 11.

Seed 42; world_size 4; GPU başına batch 64; nominal global batch 256.

Epoch süreleri toplamı 14907.237 saniye (4.141 saat); bağımsız test 269.182 saniye. Toplam uçtan uca süre veya fiziksel GPU-saat ölçümü değildir.

| Epoch (1 tabanlı) | Train loss | Val AUC % | Val ACC % | Val mAP % | Val IoU % | Val token F1 % |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 1.648425 | 88.2588 | 79.4992 | 68.7325 | 6.0534 | 52.9597 |
| 2 | 1.202426 | 90.1362 | 80.6653 | 80.2697 | 20.2398 | 60.7299 |
| 3 | 0.962942 | 90.8054 | 81.9036 | 82.8384 | 34.6937 | 64.1367 |
| 4 | 0.834696 | 91.2848 | 83.1013 | 84.0309 | 62.1092 | 65.7077 |
| 5 | 0.739218 | 91.1677 | 82.6449 | 84.2064 | 65.3997 | 66.4332 |
| 6 | 0.676409 | 91.4886 | 83.4900 | 84.0653 | 65.5127 | 66.5125 |
| 7 | 0.623943 | 91.7173 | 84.1679 | 84.8310 | 67.3204 | 67.5315 |
| 8 | 0.582759 | 91.6857 | 84.1092 | 84.4276 | 66.8993 | 68.0008 |
| 9 | 0.542210 | 91.9654 | 84.4301 | 85.0880 | 68.7451 | 68.3867 |
| 10 | 0.514645 | 91.7237 | 84.2131 | 85.0171 | 69.3074 | 67.3851 |
| 11 | 0.492259 | 92.3246 | 84.8956 | 85.2096 | 70.2088 | 68.1275 |

## HAMMER test — tüm görevler

| Metrik | Değer % |
|---|---:|
| AUC_cls | 92.666003 |
| ACC_cls | 85.342668 |
| EER_cls | 15.137899 |
| MAP | 86.440289 |
| OP | 80.436424 |
| OR | 77.388188 |
| OF1 | 78.882869 |
| CP | 78.596637 |
| CR | 77.427848 |
| CF1 | 78.007865 |
| OP_k | 80.436424 |
| OR_k | 77.388188 |
| OF1_k | 78.882869 |
| CP_k | 78.596637 |
| CR_k | 77.427848 |
| CF1_k | 78.007865 |
| IOU_score | 70.611840 |
| IOU_ACC_50 | 78.931072 |
| IOU_ACC_75 | 61.771028 |
| IOU_ACC_95 | 39.696282 |
| ACC_tok | 91.274916 |
| Precision_tok | 74.401909 |
| Recall_tok | 67.467371 |
| F1_tok | 70.765161 |

## Kimlik ve sayısal sınır

Üç CRAF raporuyla checkpoint/metadata/kayıtlı upstream source hash eşleşir. HAMMER test AUC değeri ile CRAF cache baseline AUC farkı yaklaşık 8,76e-10; aynı hassasiyet düzeyinde eşittir. Özgün değerler değiştirilmedi. SHA kayıt eşliği ağırlığın yeniden yüklenmesi değildir.

Test girdisi görüntü+metindir. Tür, kutu ve token sonuçları yalnız HAMMER baseline için ölçülmüştür; CRAF iyileştirmesi olarak sunulmaz.

Kontroller: 32 geçti; yeni CUDA çalıştırması yapılmadı.
