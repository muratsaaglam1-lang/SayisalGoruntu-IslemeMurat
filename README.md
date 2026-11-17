# Klasik Görüntü İşleme ile Hücre Segmentasyonu, Sayımı ve Zamansal Analizi

**Görüntü İşleme — Yüksek Lisans Ara Sınav Projesi**
Öğrenci: **Murat Sağlam** (264327008)

Bu projede Cell Tracking Challenge **Fluo-N2DH-GOWT1** veri setindeki ardışık floresan mikroskopi
görüntülerinde hücre çekirdekleri sayılmaktadır. Yalnızca **klasik** yöntemler kullanılmıştır:
yoğunluk dönüşümleri, uzamsal filtreler, eşikleme, morfoloji, distance transform ve watershed.
Derin öğrenme veya önceden eğitilmiş model kullanılmamıştır.

Yöntemler 10 geliştirme görüntüsünde **46 kontrollü deneyle** karşılaştırılmıştır. 4 aday pipeline
5 doğrulama görüntüsünde değerlendirilmiş ve nihai pipeline doğrulama MAE değerine göre seçilmiştir.
Seçilen pipeline, test referansları okunmadan önce **kilitlenmiş**, ardından 15 bağımsız test
görüntüsünde baseline ile birlikte değerlendirilmiştir.

## Veri seti

* Sayfa: <https://celltrackingchallenge.net/2d-datasets/>
* Eğitim arşivi (resmi indirme bağlantısı): <https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip>
* Arşiv `data/` klasörüne açılmalıdır: `data/Fluo-N2DH-GOWT1/{01,02,01_GT,02_GT,...}`.
  Farklı bir konum için `CTC_DATA_DIR` ortam değişkeni tanımlanabilir.
* Veri setinin görüntüleri ve maskeleri bu repoya **yüklenmemiştir** (`.gitignore`). Notebook,
  veri bulunamazsa arşivi resmi bağlantıdan otomatik indirir; bu davranış Colab için uygundur.

## Rastgele seçim

* **Seed:** öğrenci numarasının son 5 hanesi + 2026 = `27008 + 2026 = 29034`
* Geliştirme: Sequence 01'den 10 kare. Doğrulama: Sequence 01'den 5 kare. Geliştirme ve doğrulama
  kareleri çakışmaz ve aralarında en az 5 karelik boşluk vardır.
* Test: Sequence 02'den ardışık 5'er karelik 3 blok. Bloklar çakışmaz ve aralarında en az 5 kare boşluk vardır.
* Koşullar sağlanamazsa aynı `numpy.random.Generator` ile deterministik olarak yeniden çekilir.
  Deneme sayısı da kaydedilir. Seçim `selected_frames.csv` dosyasındadır.

## Kurulum ve çalıştırma

```bash
pip install -r requirements.txt
# Jupyter / Google Colab:
jupyter notebook main.ipynb          # hücreleri yukarıdan aşağı sırayla çalıştırın
# veya komut satırı:
python run_pipeline.py --stage until-lock   # adım 2-12: seçim -> deneyler -> doğrulama -> TEST KİLİDİ
python run_pipeline.py --stage evaluate     # adım 13-15: test metrikleri, zamansal analiz, şekiller
```

**Google Colab:** Repoyu klonlayın (`!git clone <repo-url>` ve `%cd <repo>`), ardından
`!pip install -r requirements.txt` komutunu çalıştırıp `main.ipynb` hücrelerini sırayla yürütün.
Veri seti ilk hücrelerde otomatik indirilir.

### Çalıştırma sırası

| Sıra | Notebook bölümü | Modül | Çıktı |
|---|---|---|---|
| 1 | 2. Veri inceleme | `eda.py` | `dataset_stats.csv`, `figures/fig_dataset_histograms.png` |
| 2 | 3. Kare seçimi | `selection.py` | `selected_frames.csv` |
| 3 | 4. Referans sayımlar | `references.py` | `reference_counts_dev_val.csv`, `cell_stats_dev_val.csv` |
| 4 | 5-10. Baseline ve deneyler | `experiments.py` | `experiments.csv`, `experiments_per_frame.csv`, `baseline_min_area_calibration.csv` |
| 5 | 11. Doğrulama | `validation.py` | `validation_results.csv`, `validation_per_frame.csv`, `final_pipeline.json` |
| 6 | 12. Kilit | `testing.py` | `test_predictions_locked.csv`, `test_lock_manifest.json` |
| 7 | 13. Test | `testing.py` | `test_results.csv`, `test_summary.csv`, `test_object_errors.csv` |
| 8 | 14. Zamansal analiz | `testing.py` | `temporal_analysis.csv`, `figures/fig5_temporal_counts.png` |
| 9 | 15. Hata analizi | `posthoc.py` | `posthoc_background_noise.csv`, `posthoc_object_shapes.csv` |

## Nihai pipeline

Nihai pipeline **Pipeline B**'dir: Gaussian 9×9 → log dönüşümü (c·log(1+r), p99.9 normalizasyonu) →
adaptif Gauss eşik (blok 151, C = −20) → opening (elips 7) → closing (elips 11) → delik doldurma →
bağlı bileşen analizi (alan ≥ 300 px, görüntü kenarına değen nesneler çıkarılır) → sayım.
Tüm parametreler `final_pipeline.json` dosyasındadır.

**Baseline:** ham gri görüntü → Otsu → bağlı bileşen analizi → sayım. Minimum alan 50 px'tir ve
geliştirme verisinde belirlenmiştir.

## Test güvenliği

1. `final_pipeline.json` doğrulamadan sonra yazılır. Ardından `testing.lock_test_predictions()`
   yalnızca ham test görüntülerini okuyarak `test_predictions_locked.csv` dosyasını üretir.
2. `test_lock_manifest.json`, kilit anındaki dosyaların SHA-256 özetlerini saklar.
3. `references.load_test_references()` önce `verify_lock()` çağırır. Kilit yoksa veya dosyalar
   değiştirilmişse test referansları **okunamaz**.
4. Kilit ve değerlendirme iki ayrı commit ile yüklenmiştir (bkz. `GIT_ADIMLARI.md`).

## Dosyalar

| Dosya / Klasör | İçerik |
|---|---|
| `main.ipynb` | Tüm adımları sırayla çalıştıran notebook (görüntü çıktıları temizlenmiş) |
| `src/cellcount/` | Kaynak kodlar: `ops.py` (operatörler), `pipeline.py`, `experiments.py`, `validation.py`, `testing.py`, `figures.py`, `posthoc.py` |
| `run_pipeline.py` | Komut satırı alternatifi |
| `selected_frames.csv` | Seçilen kareler, gruplar, test blokları, seed |
| `experiments.csv` | 46 kontrollü deneyin yöntem, parametre ve performansı (MAE, RMSE, Bias, süre, nesne hata türleri) |
| `validation_results.csv` | Aday pipeline'ların doğrulama sonuçları ve seçim |
| `final_pipeline.json` | Nihai pipeline ve baseline konfigürasyonu, yazılım sürümleri |
| `test_predictions_locked.csv` | Test referansları okunmadan kaydedilen tahminler |
| `test_results.csv` / `test_summary.csv` | Kare bazlı test sonuçları ve özet metrikler |
| `temporal_analysis.csv` | Blok içi ardışık sayım değişimleri |
| `figures/` | Veri setini yeniden dağıtmayan grafikler (akış diyagramı, deney/doğrulama/zamansal grafikler) |

Orijinal görüntülerden üretilen şekiller (`report_figures/`) yalnızca teknik raporda ve dersin özel
teslim ortamında yer alır; repoya yüklenmez.

## Yazılım

Test edilen ortam: Python 3.14, OpenCV 5.0, scikit-image 0.26, SciPy 1.18, NumPy 2.5, pandas 3.1,
matplotlib 3.11. Kullanılan OpenCV ve scikit-image fonksiyonları bu kütüphanelerin eski sürümlerinde
de bulunmaktadır.
