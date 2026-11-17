"""Veri setinin ön incelemesi (ödev Bölüm 2).

Sequence 01 ve 02 görüntülerinin boyut, veri tipi, yoğunluk ve histogram
özellikleri incelenir. Hücre boyut/parlaklık istatistikleri yalnızca
geliştirme+doğrulama (Sequence 01) referanslarından hesaplanır; test
referansları bu aşamada okunmaz.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def image_statistics() -> pd.DataFrame:
    rows = []
    for seq in (config.DEV_SEQ, config.TEST_SEQ):
        frames = data.list_frames(seq)
        stack = np.stack([data.read_frame(seq, t) for t in frames])
        per_frame_p999 = np.percentile(stack.reshape(len(frames), -1), 99.9, axis=1)
        rows.append({
            "sequence": seq, "n_frames": len(frames),
            "shape": f"{stack.shape[1]}x{stack.shape[2]}", "dtype": str(stack.dtype),
            "min": int(stack.min()), "max": int(stack.max()),
            "mean": round(float(stack.mean()), 2), "std": round(float(stack.std()), 2),
            "p1": float(np.percentile(stack, 1)), "p50": float(np.percentile(stack, 50)),
            "p99": float(np.percentile(stack, 99)), "p99_9": float(np.percentile(stack, 99.9)),
            "zero_fraction": round(float((stack == 0).mean()), 3),
            "p99_9_per_frame_min": float(per_frame_p999.min()),
            "p99_9_per_frame_max": float(per_frame_p999.max()),
        })
    df = pd.DataFrame(rows)
    df.to_csv(config.DATASET_STATS_CSV, index=False)
    return df


def sequence_histograms() -> dict:
    """Her sekansın tüm karelerinin birleşik histogramı (256 kutu)."""
    out = {}
    for seq in (config.DEV_SEQ, config.TEST_SEQ):
        h = np.zeros(256, np.int64)
        for t in data.list_frames(seq):
            h += np.bincount(data.read_frame(seq, t).ravel(), minlength=256)[:256]
        out[seq] = h
    return out


def cell_statistics(selection: pd.DataFrame) -> pd.DataFrame:
    """Geliştirme + doğrulama karelerindeki hücrelerin alan ve parlaklık dağılımı."""
    rows = []
    for r in selection[selection.group.isin(["development", "validation"])].itertuples():
        img = data.read_frame(r.sequence, r.frame)
        lab = data._read_tra(r.sequence, r.frame)
        for i in np.unique(lab)[1:]:
            m = lab == i
            ys, xs = np.nonzero(m)
            rows.append({"image_id": r.image_id, "frame": r.frame, "cell_id": int(i),
                         "marker_area_px": int(m.sum()),
                         "eq_diameter_px": round(float(2 * np.sqrt(m.sum() / np.pi)), 1),
                         "mean_intensity": round(float(img[m].mean()), 2),
                         "touches_border": bool(ys.min() == 0 or xs.min() == 0 or
                                                ys.max() == img.shape[0] - 1 or
                                                xs.max() == img.shape[1] - 1)})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPO_ROOT / "cell_stats_dev_val.csv", index=False)
    return df
