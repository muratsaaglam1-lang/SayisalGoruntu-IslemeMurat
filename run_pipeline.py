"""Komut satırından tüm projeyi çalıştırır (notebook'a alternatif).

    python run_pipeline.py --stage until-lock   # veri seçimi -> deneyler -> doğrulama -> test kilidi
    python run_pipeline.py --stage evaluate     # test değerlendirmesi, zamansal analiz, şekiller
    python run_pipeline.py --stage all
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
if hasattr(sys.stdout, "reconfigure"):  # Windows konsolunda Türkçe/Δ karakterleri için
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from cellcount.run_all import main  # noqa: E402

if __name__ == "__main__":
    main()
