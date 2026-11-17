"""Kilit sonrası hata analizi (yalnızca yorum amaçlı; kilitli tahminleri DEĞİŞTİRMEZ).

1. Arka plan gürültüsü: nihai pipeline'ın log-uzayı görüntüsünde, referans hücrelerden
   uzak (40 px genişletilmiş maskenin dışı) piksellerin ortalama ve standart sapması.
   Sequence 01 (geliştirme) ile Sequence 02 (test) karşılaştırılır.
2. Yanlış pozitif nesnelerin şekli: doğru tespitlerle (TP) yanlış pozitiflerin (FP)
   solidity (alan / dışbükey zarf alanı) ve alan dağılımı.
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
from skimage.measure import regionprops

from . import config, data, experiments, ops, pipeline
from .testing import verify_lock


def _log_image(img: np.ndarray, cfg: dict) -> np.ndarray:
    x = ops.apply_filter(img, cfg["filter"])
    return ops.apply_intensity(x, cfg["intensity"])


def background_noise(selection: pd.DataFrame) -> pd.DataFrame:
    verify_lock()
    cfg = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))["final_pipeline"]
    se = ops.structuring_element(81)
    rows = []
    for r in selection[selection.group.isin(["development", "test"])].itertuples():
        img = data.read_frame(r.sequence, r.frame)
        gt = data._read_tra(r.sequence, r.frame)
        far = cv2.dilate((gt > 0).astype(np.uint8), se) == 0
        x = _log_image(img, cfg).astype(np.float32)
        rows.append({"group": r.group, "sequence": r.sequence, "frame": r.frame,
                     "raw_p99_9": float(np.percentile(img, 99.9)),
                     "bg_mean_log": float(x[far].mean()), "bg_std_log": float(x[far].std()),
                     "cell_median_log": float(np.median(x[gt > 0]))})
    df = pd.DataFrame(rows)
    out = df.groupby(["group", "sequence"]).mean(numeric_only=True).drop(columns="frame").round(2)
    out["cell_bg_contrast_in_std"] = ((out.cell_median_log - out.bg_mean_log) / out.bg_std_log).round(2)
    out.reset_index().to_csv(config.REPO_ROOT / "posthoc_background_noise.csv", index=False)
    return out.reset_index()


def false_positive_shapes(selection: pd.DataFrame) -> pd.DataFrame:
    verify_lock()
    cfg = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))["final_pipeline"]
    ctx = experiments.build_context(selection)
    rows = []
    for r in selection[selection.group.isin(["development", "test"])].itertuples():
        img = data.read_frame(r.sequence, r.frame)
        gt = data._read_tra(r.sequence, r.frame)
        lab = pipeline.run(img, cfg, ctx)["labels"]
        for p in regionprops(lab):
            hit = np.any(gt[tuple(p.coords.T)] > 0)
            rows.append({"group": r.group, "kind": "TP" if hit else "FP", "area": p.area,
                         "solidity": p.solidity, "eccentricity": p.eccentricity})
    df = pd.DataFrame(rows)
    summ = df.groupby(["group", "kind"]).agg(n=("area", "size"), area_median=("area", "median"),
                                             solidity_median=("solidity", "median"),
                                             solidity_p10=("solidity", lambda s: s.quantile(0.1)),
                                             solidity_p90=("solidity", lambda s: s.quantile(0.9)))
    summ = summ.round(3).reset_index()
    summ.to_csv(config.REPO_ROOT / "posthoc_object_shapes.csv", index=False)
    return summ


def candidates_on_test(selection: pd.DataFrame) -> pd.DataFrame:
    """Seçilmeyen adayların test performansı — YALNIZCA BİLGİ AMAÇLI.

    Seçim doğrulama sonuçlarıyla yapılmış ve kilitlenmiştir; bu tablo seçimi veya
    kilitli tahminleri değiştirmek için kullanılmaz, yalnızca doğrulama->test
    genellemesini tartışmak içindir (ödev Bölüm 15, soru 6).
    """
    from . import metrics
    verify_lock()
    ctx = experiments.build_context(selection)
    tst = selection[selection.group == "test"]
    imgs = {r.image_id: data.read_frame(r.sequence, r.frame) for r in tst.itertuples()}
    refs = {r.image_id: len(np.unique(data._read_tra(r.sequence, r.frame))) - 1 for r in tst.itertuples()}
    val = pd.read_csv(config.VALIDATION_CSV).set_index("pipeline")
    rows = []
    for name, c in experiments.candidate_pipelines().items():
        preds = [pipeline.run(imgs[k], c["cfg"], ctx)["count"] for k in imgs]
        m = metrics.count_metrics(list(refs.values()), preds)
        rows.append({"pipeline": name, "val_MAE": val.loc[name, "val_MAE"],
                     "test_MAE_posthoc": round(m["MAE"], 3), "test_RMSE_posthoc": round(m["RMSE"], 3),
                     "test_bias_posthoc": round(m["Bias"], 3),
                     "selected": "SEÇİLDİ (kilitli)" if val.loc[name, "selected"] == "SEÇİLDİ" else ""})
    df = pd.DataFrame(rows)
    df.to_csv(config.REPO_ROOT / "posthoc_candidates_on_test.csv", index=False)
    return df
