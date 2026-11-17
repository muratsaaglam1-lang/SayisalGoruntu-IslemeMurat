"""Kontrollü deneyler (ödev Bölüm 5-10).

Her deney grubunda bir *sabit referans ayarı* korunur ve yalnızca incelenen
faktör değiştirilir. Bütün deneyler yalnızca 10 geliştirme görüntüsünde
değerlendirilir. Doğrulama ve test görüntüleri burada kullanılmaz.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from . import config, data, metrics, pipeline, references

# ---------------------------------------------------------------------------
# Sabit (referans) ayarlar
# ---------------------------------------------------------------------------
MORPH_REF = [{"op": "open", "size": 7}, {"op": "close", "size": 11}, {"op": "fill_holes"}]

# Yoğunluk/filtre/segmentasyon gruplarında sabit tutulan çerçeve
FRAME_REF: dict = {
    "intensity": {"method": "log", "p_high": 99.9},
    "filter": {"method": "gaussian", "ksize": 9},
    "order": "filter_first",
    "segmentation": {"method": "otsu"},
    "morphology": MORPH_REF,
    "separation": {"method": "none"},
    "min_area": 300,
    "clear_border": True,
}
ADAPTIVE_REF = {"method": "adaptive_gaussian", "block_size": 151, "C": -20}


def _cfg(base: dict, **changes) -> dict:
    c = copy.deepcopy(base)
    c.update(copy.deepcopy(changes))
    return c


def experiment_definitions(baseline_min_area: int) -> list[dict]:
    """Deney listesi: her öğe {id, group, factor, cfg}."""
    B = _cfg(pipeline.BASELINE, min_area=baseline_min_area)
    F = FRAME_REF
    FA = _cfg(F, segmentation=ADAPTIVE_REF)  # morfoloji grubunun sabit çerçevesi
    E = []

    def add(group, factor, cfg):
        E.append({"group": group, "factor": factor, "cfg": cfg})

    # --- Bölüm 5: Baseline --------------------------------------------------
    add("5-Baseline", "Baseline (ham, Otsu, CCL)", B)
    # --- Bölüm 6: Yoğunluk dönüşümleri (G9 önce, Otsu, MORPH_REF sabit) -----
    add("6-Yoğunluk", "Dönüşüm yok (orijinal)", _cfg(F, intensity={"method": "none"}))
    add("6-Yoğunluk", "Contrast stretching", _cfg(F, intensity={"method": "stretch", "p_low": 1.0, "p_high": 99.8}))
    add("6-Yoğunluk", "Histogram equalization", _cfg(F, intensity={"method": "histeq"}))
    add("6-Yoğunluk", "CLAHE", _cfg(F, intensity={"method": "clahe", "clip_limit": 2.0, "tile": 8}))
    add("6-Yoğunluk", "Histogram matching", _cfg(F, intensity={"method": "match"}))
    add("6-Yoğunluk", "Gamma (0.5)", _cfg(F, intensity={"method": "gamma", "gamma": 0.5, "p_high": 99.9}))
    add("6-Yoğunluk", "Log dönüşümü", _cfg(F))
    # --- Bölüm 7: Filtreler (log, Otsu, MORPH_REF sabit) ----------------------
    add("7-Filtre", "Filtre yok", _cfg(F, filter={"method": "none"}))
    add("7-Filtre", "Gaussian 3x3", _cfg(F, filter={"method": "gaussian", "ksize": 3}))
    add("7-Filtre", "Gaussian 5x5", _cfg(F, filter={"method": "gaussian", "ksize": 5}))
    add("7-Filtre", "Gaussian 13x13", _cfg(F, filter={"method": "gaussian", "ksize": 13}))
    add("7-Filtre", "Median 5x5", _cfg(F, filter={"method": "median", "ksize": 5}))
    add("7-Filtre", "Median 9x9", _cfg(F, filter={"method": "median", "ksize": 9}))
    add("7-Filtre", "Laplacian keskinleştirme", _cfg(F, filter={"method": "laplacian_sharpen", "amount": 1.0}))
    add("7-Filtre", "Gaussian 9x9 + Laplacian", _cfg(F, filter={"method": "gaussian_then_laplacian",
                                                                 "ksize": 9, "amount": 1.0}))
    add("7-Filtre", "Gaussian 9x9 + unsharp",_cfg(F, filter={"method": "gaussian_then_unsharp", "ksize": 9,
                                                               "sharp_ksize": 9, "amount": 1.0}))
    add("7-Filtre", "Sıra: log -> Gaussian 9x9", _cfg(F, order="intensity_first"))
    # --- Bölüm 8: Segmentasyon (G9 önce, log, MORPH_REF sabit) ---------------
    for t in (90, 110, 130):
        add("8-Segmentasyon", f"Sabit eşik T={t}", _cfg(F, segmentation={"method": "fixed", "threshold": t}))
    add("8-Segmentasyon", "Triangle eşik", _cfg(F, segmentation={"method": "triangle"}))
    add("8-Segmentasyon", "K-means (k=4, 2 arka plan)", _cfg(F, segmentation={"method": "kmeans", "k": 4, "n_background": 2}))
    add("8-Segmentasyon", "Adaptif ortalama (151, C=-20)", _cfg(F, segmentation={"method": "adaptive_mean", "block_size": 151, "C": -20}))
    add("8-Segmentasyon", "Adaptif Gauss (151, C=-20)", FA)
    add("8-Segmentasyon", "Top-hat(81) + Otsu", _cfg(F, background={"method": "tophat", "size": 81}))
    for b, c in ((101, -20), (201, -20), (151, -15), (151, -25)):
        add("8-Segmentasyon", f"Adaptif Gauss ({b}, C={c})",
            _cfg(F, segmentation={"method": "adaptive_gaussian", "block_size": b, "C": c}))
    # --- Bölüm 9: Morfoloji (G9 önce, log, adaptif Gauss sabit) ---------------
    add("9-Morfoloji", "Morfoloji yok", _cfg(FA, morphology=[]))
    add("9-Morfoloji", "Erosion 5", _cfg(FA, morphology=[{"op": "erode", "size": 5}]))
    add("9-Morfoloji", "Dilation 5", _cfg(FA, morphology=[{"op": "dilate", "size": 5}]))
    add("9-Morfoloji", "Opening 3", _cfg(FA, morphology=[{"op": "open", "size": 3}]))
    add("9-Morfoloji", "Opening 7", _cfg(FA, morphology=[{"op": "open", "size": 7}]))
    add("9-Morfoloji", "Opening 11", _cfg(FA, morphology=[{"op": "open", "size": 11}]))
    add("9-Morfoloji", "Closing 11", _cfg(FA, morphology=[{"op": "close", "size": 11}]))
    add("9-Morfoloji", "Opening 7 -> Closing 11", _cfg(FA, morphology=MORPH_REF[:2]))
    add("9-Morfoloji", "Closing 11 -> Opening 7 (ters sıra)", _cfg(FA, morphology=[MORPH_REF[1], MORPH_REF[0]]))
    add("9-Morfoloji", "Open7 -> Close11 -> Fill", FA)
    add("9-Morfoloji", "Kenar nesneleri korunur", _cfg(FA, clear_border=False))
    for ma in (100, 500):
        add("9-Morfoloji", f"min_area={ma}", _cfg(FA, min_area=ma))
    add("9-Morfoloji", "Watershed (DT, h=3, yumuşatma yok)",
        _cfg(FA, separation={"method": "watershed", "marker": "hmax", "h": 3, "smooth_sigma": 0}))
    add("9-Morfoloji", "Watershed (DT, sigma=2, h=3)",
        _cfg(FA, separation={"method": "watershed", "marker": "hmax", "h": 3, "smooth_sigma": 2}))
    add("9-Morfoloji", "Watershed (DT, peak min_dist=35)",
        _cfg(FA, separation={"method": "watershed", "marker": "peak", "min_distance": 35, "smooth_sigma": 2}))

    for i, e in enumerate(E, start=1):
        e["id"] = f"E{i:02d}"
        e["cfg"]["name"] = e["id"]
    return E


# ---------------------------------------------------------------------------
# Aday tam pipeline'lar (Bölüm 10-11)
# ---------------------------------------------------------------------------
def candidate_pipelines() -> dict[str, dict]:
    A = _cfg(FRAME_REF, name="Pipeline A")
    Bp = _cfg(FRAME_REF, name="Pipeline B", segmentation=ADAPTIVE_REF)
    C = _cfg(Bp, name="Pipeline C",
             separation={"method": "watershed", "marker": "hmax", "h": 3, "smooth_sigma": 2})
    D = _cfg(FRAME_REF, name="Pipeline D", background={"method": "tophat", "size": 81})
    return {
        "Pipeline A": {"desc": "Global eşik: G9 -> log -> Otsu -> open7/close11/fill", "cfg": A},
        "Pipeline B": {"desc": "Yerel eşik: G9 -> log -> adaptif Gauss -> open7/close11/fill", "cfg": Bp},
        "Pipeline C": {"desc": "Pipeline B + distance transform/h-maxima watershed", "cfg": C},
        "Pipeline D": {"desc": "Arka plan çıkarma: G9 -> log -> top-hat(81) -> Otsu -> open7/close11/fill", "cfg": D},
    }


# ---------------------------------------------------------------------------
# Çalıştırma
# ---------------------------------------------------------------------------
def build_context(selection: pd.DataFrame) -> dict:
    """Histogram eşleme referansı: geliştirme kümesinde ortalama yoğunluğu
    medyan olan görüntü (deterministik, yalnızca geliştirme verisi)."""
    dev = selection[selection.group == "development"]
    means = {r.frame: float(data.read_frame(r.sequence, r.frame).mean()) for r in dev.itertuples()}
    med = float(np.median(list(means.values())))
    ref_frame = min(means, key=lambda f: (abs(means[f] - med), f))
    return {"match_reference": data.read_frame(config.DEV_SEQ, ref_frame),
            "match_reference_frame": int(ref_frame)}


def evaluate_config(cfg: dict, images: dict, refs: dict, gt_labels: dict | None,
                    context: dict) -> tuple[dict, list[dict]]:
    rows, tot = [], {}
    for key, img in images.items():
        out = pipeline.run(img, cfg, context)
        row = {"key": key, "ref_count": refs[key], "pred_count": out["count"],
               "error": out["count"] - refs[key], "time_s": out["time_s"]}
        if gt_labels is not None:
            ea = metrics.object_error_analysis(out["labels"], gt_labels[key])
            row.update(ea)
            for k, v in ea.items():
                tot[k] = tot.get(k, 0) + v
        rows.append(row)
    m = metrics.count_metrics([r["ref_count"] for r in rows], [r["pred_count"] for r in rows],
                              [r["time_s"] for r in rows])
    m.update(tot)
    return m, rows


def load_group(selection: pd.DataFrame, group: str, with_labels: bool = True):
    if group not in ("development", "validation"):
        raise ValueError("Bu fonksiyon test verisi için kullanılamaz.")
    refs_df = references.load_dev_val_references()
    sub = selection[selection.group == group]
    images = {r.image_id: data.read_frame(r.sequence, r.frame) for r in sub.itertuples()}
    refs = dict(zip(refs_df.image_id, refs_df.ref_count))
    refs = {k: int(refs[k]) for k in images}
    labels = ({r.image_id: data._read_tra(r.sequence, r.frame) for r in sub.itertuples()}
              if with_labels else None)
    return images, refs, labels


def choose_baseline_min_area(images, refs, context, grid=(0, 10, 20, 30, 50, 75, 100, 150, 300)):
    """Baseline için sabit minimum alan eşiğini geliştirme MAE'sine göre seçer."""
    rows = []
    for a in grid:
        cfg = _cfg(pipeline.BASELINE, min_area=a)
        m, _ = evaluate_config(cfg, images, refs, None, context)
        rows.append({"min_area": a, "MAE": m["MAE"], "RMSE": m["RMSE"], "Bias": m["Bias"]})
    df = pd.DataFrame(rows)
    best = int(df.sort_values(["MAE", "RMSE", "min_area"]).iloc[0]["min_area"])
    return best, df


def run_experiments(selection: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, pd.DataFrame]:
    images, refs, labels = load_group(selection, "development")
    context = build_context(selection)
    best_min_area, calib = choose_baseline_min_area(images, refs, context)
    calib.to_csv(config.REPO_ROOT / "baseline_min_area_calibration.csv", index=False)

    summary, per_frame = [], []
    for e in experiment_definitions(best_min_area):
        m, rows = evaluate_config(e["cfg"], images, refs, labels, context)
        d = pipeline.describe(e["cfg"])
        summary.append({
            "experiment_id": e["id"], "group": e["group"], "changed_factor": e["factor"],
            "intensity": d["intensity"], "filter": d["filter"], "segmentation": d["segmentation"],
            "morphology": d["morphology"], "separation": d["separation"], "params": d["params"],
            "count_MAE": round(m["MAE"], 3), "count_RMSE": round(m["RMSE"], 3),
            "bias": round(m["Bias"], 3), "exact_acc": round(m["ExactAcc"], 3),
            "mean_time_s": round(m["MeanTime_s"], 4),
            "missed": m.get("missed"), "false_pos": m.get("false_pos"),
            "merged_objs": m.get("merged"), "split_cells": m.get("split"),
            "pred_counts": " ".join(str(r["pred_count"]) for r in rows),
        })
        for r in rows:
            per_frame.append({"experiment_id": e["id"], "image_id": r["key"], **{
                k: v for k, v in r.items() if k != "key"}})
    exp_df = pd.DataFrame(summary)
    exp_df.to_csv(config.EXPERIMENTS_CSV, index=False)
    pf = pd.DataFrame(per_frame)
    pf.to_csv(config.EXPERIMENTS_PER_FRAME_CSV, index=False)
    return exp_df, pf, best_min_area, calib
