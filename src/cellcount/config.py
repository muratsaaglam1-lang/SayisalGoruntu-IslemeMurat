"""Proje genelindeki sabitler ve dosya yolları.

Veri seti konumu varsayılan olarak ``<repo>/data/Fluo-N2DH-GOWT1`` klasörüdür.
Farklı bir konum kullanmak için ``CTC_DATA_DIR`` ortam değişkeni tanımlanabilir.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Öğrenci / tekrarlanabilirlik bilgileri -------------------------------
STUDENT_NAME = "Murat Sağlam"
STUDENT_NO = "264327008"
SEED = int(STUDENT_NO[-5:]) + 2026  # 27008 + 2026 = 29034

# --- Yollar ---------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("CTC_DATA_DIR", REPO_ROOT / "data" / "Fluo-N2DH-GOWT1"))
DATASET_URL = "https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip"

# GitHub'a gidecek sayısal çıktılar repo kökünde tutulur.
SELECTED_FRAMES_CSV = REPO_ROOT / "selected_frames.csv"
REFERENCE_COUNTS_CSV = REPO_ROOT / "reference_counts_dev_val.csv"
DATASET_STATS_CSV = REPO_ROOT / "dataset_stats.csv"
EXPERIMENTS_CSV = REPO_ROOT / "experiments.csv"
EXPERIMENTS_PER_FRAME_CSV = REPO_ROOT / "experiments_per_frame.csv"
VALIDATION_CSV = REPO_ROOT / "validation_results.csv"
VALIDATION_PER_FRAME_CSV = REPO_ROOT / "validation_per_frame.csv"
FINAL_PIPELINE_JSON = REPO_ROOT / "final_pipeline.json"
TEST_LOCKED_CSV = REPO_ROOT / "test_predictions_locked.csv"
TEST_LOCK_MANIFEST = REPO_ROOT / "test_lock_manifest.json"
TEST_RESULTS_CSV = REPO_ROOT / "test_results.csv"
TEST_SUMMARY_CSV = REPO_ROOT / "test_summary.csv"
TEMPORAL_CSV = REPO_ROOT / "temporal_analysis.csv"

# Veri setini yeniden dağıtmayan grafikler (GitHub'a yüklenebilir)
FIG_DIR = REPO_ROOT / "figures"
# Orijinal görüntü içeren şekiller (yalnızca rapor / özel teslim; .gitignore'da)
REPORT_FIG_DIR = REPO_ROOT / "report_figures"

# --- Veri bölünmesi -------------------------------------------------------
DEV_SEQ = "01"
TEST_SEQ = "02"
N_VAL = 5
N_DEV = 10
N_TEST_BLOCKS = 3
TEST_BLOCK_LEN = 5
MIN_GAP = 5  # gruplar/bloklar arasında en az 5 karelik boşluk (arada >=5 kare)
MAX_SELECTION_ATTEMPTS = 1000


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
