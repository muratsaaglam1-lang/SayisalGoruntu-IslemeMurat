"""Referans hücre sayılarının TRA anotasyonlarından elde edilmesi.

Referans sayı = TRA etiket görüntüsündeki, 0 (arka plan) hariç farklı pozitif
hücre kimliklerinin sayısı.

Test sızıntısını önlemek için Sequence 02 (test) referansları yalnızca
``test_predictions_locked.csv`` ve kilit manifestosu oluşturulduktan ve
bütünlükleri doğrulandıktan sonra okunabilir (bkz. :func:`load_test_references`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def count_tra_labels(seq: str, t: int) -> dict:
    lab = data._read_tra(seq, t)
    ids = np.unique(lab)
    ids = ids[ids > 0]
    areas = [int((lab == i).sum()) for i in ids]
    return {"ref_count": int(len(ids)),
            "min_marker_area": int(min(areas)) if areas else 0,
            "median_marker_area": float(np.median(areas)) if areas else 0.0,
            "n_small_markers_lt300px": int(sum(a < 300 for a in areas))}


def build_dev_val_references(selection: pd.DataFrame) -> pd.DataFrame:
    """Geliştirme ve doğrulama karelerinin referans sayılarını hesaplar."""
    rows = []
    for r in selection[selection.group.isin(["development", "validation"])].itertuples():
        if r.sequence != config.DEV_SEQ:
            raise RuntimeError("Geliştirme/doğrulama yalnızca Sequence 01'den olmalı.")
        info = count_tra_labels(r.sequence, r.frame)
        rows.append({"image_id": r.image_id, "group": r.group, "sequence": r.sequence,
                     "frame": r.frame, **info})
    df = pd.DataFrame(rows)
    df.to_csv(config.REFERENCE_COUNTS_CSV, index=False)
    return df


def load_dev_val_references() -> pd.DataFrame:
    return pd.read_csv(config.REFERENCE_COUNTS_CSV, dtype={"sequence": str})


def load_test_references(selection: pd.DataFrame) -> pd.DataFrame:
    """Test referanslarını okur — yalnızca kilit doğrulandıktan sonra."""
    from .testing import verify_lock  # döngüsel içe aktarmayı önlemek için
    verify_lock()
    rows = []
    for r in selection[selection.group == "test"].itertuples():
        info = count_tra_labels(r.sequence, r.frame)
        rows.append({"image_id": r.image_id, "frame": r.frame, "test_block": r.test_block,
                     **info})
    return pd.DataFrame(rows)
