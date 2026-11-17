# Ödev Bölüm 12 ve 18.4: kilit ve test değerlendirmesi iki AYRI commit olarak yüklenir.
# Kullanım (repo kökünde, PowerShell):
#   powershell -ExecutionPolicy Bypass -File git_adimlari\git_commit_adimlari.ps1 -RepoUrl https://github.com/<kullanici>/hucre-sayimi.git
param([Parameter(Mandatory = $true)][string]$RepoUrl)
$ErrorActionPreference = "Stop"

if (-not (Test-Path .git)) { git init -b main }

# --- COMMIT 1: kod + deneyler + doğrulama + nihai pipeline + kilitli test tahminleri ---
Copy-Item main.ipynb git_adimlari\main_full.ipynb.bak -Force
Copy-Item git_adimlari\main_commit1.ipynb main.ipynb -Force   # çıktısız notebook (test sonucu içermez)
git add .gitignore README.md requirements.txt run_pipeline.py GIT_ADIMLARI.md src main.ipynb `
    selected_frames.csv reference_counts_dev_val.csv dataset_stats.csv cell_stats_dev_val.csv `
    baseline_min_area_calibration.csv experiments.csv experiments_per_frame.csv `
    validation_results.csv validation_per_frame.csv final_pipeline.json `
    test_predictions_locked.csv test_lock_manifest.json `
    figures/fig1_final_pipeline_flow.png figures/fig_dataset_histograms.png `
    figures/fig_experiments_mae.png figures/fig_parameter_sensitivity.png figures/fig_validation.png
git commit -m "Nihai pipeline sabitlendi ve test tahminleri kilitlendi (test referanslari okunmadan)"

# --- COMMIT 2: test değerlendirmesi, zamansal analiz, hata analizi ---
Copy-Item git_adimlari\main_full.ipynb.bak main.ipynb -Force
git add -A
git commit -m "Test degerlendirmesi, zamansal analiz ve hata analizi sonuclari"

if ((git remote) -contains "origin") { git remote remove origin }
git remote add origin $RepoUrl
git push -u origin main
git log --oneline
