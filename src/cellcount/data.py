"""Veri seti okuma yardımcıları (Cell Tracking Challenge, Fluo-N2DH-GOWT1)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import config


def check_dataset() -> None:
    """Veri setinin beklenen konumda olduğunu doğrular."""
    needed = [config.DATA_DIR / "01", config.DATA_DIR / "02",
              config.DATA_DIR / "01_GT" / "TRA", config.DATA_DIR / "02_GT" / "TRA"]
    missing = [str(p) for p in needed if not p.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Veri seti bulunamadı. Arşivi şu adresten indirip "
            f"{config.DATA_DIR} konumuna açın: {config.DATASET_URL}\nEksik: {missing}")


def frame_path(seq: str, t: int) -> Path:
    return config.DATA_DIR / seq / f"t{t:03d}.tif"


def tra_path(seq: str, t: int) -> Path:
    return config.DATA_DIR / f"{seq}_GT" / "TRA" / f"man_track{t:03d}.tif"


def list_frames(seq: str) -> list[int]:
    """Sekanstaki kare numaralarını sıralı döndürür."""
    files = sorted((config.DATA_DIR / seq).glob("t*.tif"))
    return [int(f.stem[1:]) for f in files]


def read_frame(seq: str, t: int) -> np.ndarray:
    """Ham gri seviye kareyi (uint8, 1024x1024) okur."""
    img = cv2.imread(str(frame_path(seq, t)), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(frame_path(seq, t))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _read_tra(seq: str, t: int) -> np.ndarray:
    """TRA takip etiketi görüntüsünü okur (uint16). Doğrudan çağırmayın;
    sayım için :mod:`references` modülündeki korumalı fonksiyonları kullanın."""
    lab = cv2.imread(str(tra_path(seq, t)), cv2.IMREAD_UNCHANGED)
    if lab is None:
        raise FileNotFoundError(tra_path(seq, t))
    return lab
