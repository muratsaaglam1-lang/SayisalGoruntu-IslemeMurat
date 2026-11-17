# GitHub'a yükleme adımları (iki ayrı commit)

Ödev Bölüm 12 ve 18.4, kilitleme ile test değerlendirmesinin **ayrı commit'lerde** olmasını ister.
Bu yüzden yükleme iki adımda yapılır.

| Commit | İçerik |
|---|---|
| 1 — Kilit | Kaynak kodlar, çıktısız `main.ipynb`, seçim/referans/deney/doğrulama CSV'leri, `final_pipeline.json`, `test_predictions_locked.csv`, `test_lock_manifest.json`, test sonucu içermeyen grafikler |
| 2 — Değerlendirme | `test_results.csv`, `test_summary.csv`, `test_object_errors.csv`, `temporal_analysis.csv`, `posthoc_*.csv`, `figures/fig5_temporal_counts.png` ve test çıktılarını içeren `main.ipynb` |

## Otomatik (Windows PowerShell)

1. GitHub'da boş bir repo oluşturun (README eklemeden), örneğin `hucre-sayimi`.
2. Bu klasörde:

```powershell
powershell -ExecutionPolicy Bypass -File git_adimlari\git_commit_adimlari.ps1 -RepoUrl https://github.com/<kullanici>/hucre-sayimi.git
```

## Elle (bash)

```bash
git init -b main
cp main.ipynb git_adimlari/main_full.ipynb.bak && cp git_adimlari/main_commit1.ipynb main.ipynb
git add .gitignore README.md requirements.txt run_pipeline.py GIT_ADIMLARI.md src main.ipynb \
  selected_frames.csv reference_counts_dev_val.csv dataset_stats.csv cell_stats_dev_val.csv \
  baseline_min_area_calibration.csv experiments.csv experiments_per_frame.csv \
  validation_results.csv validation_per_frame.csv final_pipeline.json \
  test_predictions_locked.csv test_lock_manifest.json \
  figures/fig1_final_pipeline_flow.png figures/fig_dataset_histograms.png \
  figures/fig_experiments_mae.png figures/fig_parameter_sensitivity.png figures/fig_validation.png
git commit -m "Nihai pipeline sabitlendi ve test tahminleri kilitlendi (test referanslari okunmadan)"
cp git_adimlari/main_full.ipynb.bak main.ipynb
git add -A
git commit -m "Test degerlendirmesi, zamansal analiz ve hata analizi sonuclari"
git remote add origin https://github.com/<kullanici>/hucre-sayimi.git
git push -u origin main
```

## Notlar

* `data/`, `report_figures/` ve `git_adimlari/` `.gitignore` içindedir. Veri setinin görüntüleri ve
  görüntü içeren şekiller repoya yüklenmez. Repodaki `main.ipynb` sürümünde görüntü çıktıları temizlenmiştir.
* Kilidin bütünlüğü `test_lock_manifest.json` içindeki SHA-256 özetleriyle doğrulanır
  (`testing.verify_lock()`). Kilitten sonra kodda yapılan eklemeler yalnızca raporlama ve hata analizine
  yöneliktir: `posthoc.py` ve şekil başlıkları. Pipeline parametreleri ve tahminler değişmemiştir;
  özetler bunu doğrular.
* Repo private oluşturulursa öğretim elemanına erişim yetkisi verilmelidir.
