# Sonuçların kaynağı ve kanıt düzeyi

## 13 Kasım 2026: özgün sunucu kayıtları alındı

Kullanıcı projeye `hammer-craf/` ve `hammer-craf-v2/` klasörlerini ekledi.
İlk aktarımda 44 kod, JSON, JSONL ve belge dosyası arşivlendi.
17 Kasım 2026 düzenlemesinde kökteki birebir kopyalar kaldırıldı; kodun tek
kopyası `gpu/`, deney verileri `server_records/` altında tutulur. Kaynak
yolu, boyut ve SHA-256 eşlemesi `SERVER_EVIDENCE_INDEX.json` içindedir.
`.lock` ve macOS yardımcı dosyaları kanıt arşivine alınmadı. Bu aktarım yeni
bir GPU deneyinin çalıştırılması değildir; sunucu dosyaları yerelde incelendi.

Kanıt indeksindeki `source` özgün sunucu adını, `archived` güncel tekil dosyayı
gösterir. Kod kopyaları hash eşitliği doğrulandıktan sonra tekilleştirildi.
Model ağırlıkları ve `.pt` shard dosyaları sağlanan klasörlerde yoktur.

## Eldeki kaynaklar

| Kaynak | Kapsam |
|---|---|
| V1 `craf_results` | run/history, validation/test raporu ve 50.705 satır test tahmini |
| V1 `craf_lr_1e5` | run/history, validation/test raporu ve 50.705 satır test tahmini |
| V2 control/penalty/decision/combined | Dört koşunun run/history ve seçilen validation raporları |
| V2 combined | Ayrıca test raporu ve 50.705 satır test tahmini |
| Üç cache manifest'i | Ayrım sayımları, kaynak haber ID'leri, backbone/metadata kimlikleri ve shard hash'leri |
| Sunucudan gelen Python/shell | Paket kodu ve kaydedilen source hash'leriyle karşılaştırma |

V2 control/penalty/decision için **test çıktısı yoktur**; validation sonuçları
test sonucu olarak yazılmaz. V2 control'un validation değerlerinin v1 düşük
öğrenme oranı koşusuyla eşitliği ayrı bir test çıktısı yerine geçmez.

Üç eski test JSON'u özgün raporlarla JSON düzeyinde aynı olduğu doğrulandıktan
sonra kaldırıldı. `user_reported/TRANSCRIPTIONS.json`, eski dosya SHA256
ve normalize JSON SHA256 değerlerini kaydeder. Analiz bu kayıtlı hashlerle
karşılaştırır; ayrı bir transkripsiyon dosyasını yeniden okuduğunu iddia etmez.
`hammer_epoch10_validation_excerpt.json` tarihsel tek-epoch kesiti olarak korunur.
Sonraki aktarımda gelen dört özgün HAMMER dosyası `server_records/hammer_training/`
altına eklendi; artık toplam 48 dosya kanıt indeksindedir. 11 epoch logu (0–10),
config, run_args ve dört görev grubunun test sonucu mevcuttur. Hedef25 epoch’un
tamamlandığı veya durma nedeni bu kayıttan çıkarılamaz. Ayrı sayısal doğrulama
`hammer_training_analysis.json` ve `HAMMER_TRAINING_SUMMARY.md` içindedir.

## Yeni doğrulama kapsamı

Altı koşunun lr, batch 512, seed 42, patience, hedef epoch, karar koşullama
ve ceza ayarları artık `run.json` kayıtlarından doğrulanır. `history.json`
kaydedilen epoch sayısını ve seçimi gösterir. `seconds`, her iki kaynak
trainer'da eğitim ve epoch validation'ı sonrasında, checkpoint yazımından
önce hesaplanır. Toplamı; başlangıç, cache yükleme, ilk validation,
özellik çıkarma, kayıt yazımı ve HAMMER eğitiminin tam maliyeti değildir.

Üç testte index/etiket/baseline/CRAF sahte olasılığı vardır. Bunlarla AUC ve
sınıflandırma metrikleri yeniden hesaplanır, eşleştirilmiş hata geçişleri
sayılır. JSONL iki sınıfın orijinal logits/probability vektörünü taşımaz.
Cross-entropy, float32 ve clipping etkileri nedeniyle yaklaşık yeniden
kurulur; ana tabloda özgün rapor değeri korunur ve sayısal farklar ayrı yazılır.
Bu kayıtlar haberin görüntü/metin içeriği değildir; tek başına anlamsal hata
nedenini göstermez.

Sayısal yöntemler `SERVER_SUMMARY.md`, `server_analysis.json` ve
`../docs/SERVER_RESULTS_ANALYSIS.md` içindedir. Kod/hyperparametre doğrulaması
`../docs/SERVER_CODE_AUDIT.md` ve `../qa/SERVER_SOURCE_AUDIT.json`
içindedir. Ham dosyaların incelenmesi onları yeni GPU çalıştırmasıyla
bağımsız yeniden üretmek anlamına gelmez.

## Devam eden sınırlılıklar

- HAMMER/CRAF checkpoint'leri, 12 feature shard'ı, dataset içerikleri,
  ortam envanteri sağlanmadı. HAMMER’ın verilen 11 epoch logu artık mevcuttur. Hash'in kayıtlı
  olması bu eksik dosyanın yeniden yüklenip doğrulandığı anlamına gelmez.
- Bütün koşular seed 42 kullanır; eğitim seed varyansı ölçülmemiştir.
- Manifestte kaynak ID ayrıklığı kontrol edilebilir, fakat prediction index'ini
  kaynak haber ID'sine eşleyen satır sırası yoktur. Aynı kaynaktan çoklu
  örnekler olabilir; örnek bağımsızlığı varsayımı otomatik kabul edilmez.
- Fiziksel GPU, driver/paket sürümleri, tam duvar saati ve peak VRAM ayrıca
  doğrulanmadı. 4 × RTX A6000 48 GB bilgisi kullanıcı bildirimidir.
- İlk test sonuçları sonraki tasarımları etkiledi; yeni kanıtlar bu geçmişi
  bağımsız final holdout'a dönüştürmez. Sonraki karşılaştırmalar keşifseldir.
- V1 lr ile patience birlikte değişti. V2 dört eş ayarlı validation koşusu
  daha iyi bir bileşen karşılaştırmasıdır; diğer üç v2 koşusunun testi yoktur.

HAMMER kayıtları: ALBEF başlangıcı, seed42, world_size4, per-GPU batch64
(nominal global256), hedef25/kayıtlı11 epoch. En iyi validation ve test checkpoint
epoch10; hash CRAF omurgasıyla eşit. HAMMER epoch süreleri toplamı14907,237s
(4,141h), test269,182s. Paket HAMMER launcher zamanlaması checkpoint yazımını
içerir; CRAF zamanlaması içermez. Ortam/başlatma dahil uçtan uca maliyet değildir.
