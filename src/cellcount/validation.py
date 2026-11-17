"""Aday pipeline'ların doğrulama kümesinde karşılaştırılması ve nihai seçim
(ödev Bölüm 11).

Seçim kuralı: en düşük doğrulama MAE; eşitlikte sırasıyla RMSE, ortalama süre
ve pipeline karmaşıklığı (adım sayısı) dikkate alınır.
"""
from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import platform

import cv2
import numpy as np
import pandas as pd
import scipy
import skimage

from . import config, experiments, pipeline


def complexity(cfg: dict) -> int:
    """Pipeline karmaşıklığı: etkin işlem adımı sayısı."""
    n = 0
    n += cfg.get("intensity", {}).get("method", "none") != "none"
    n += cfg.get("filter", {}).get("method", "none") != "none"
    n += cfg.get("background", {}).get("method", "none") != "none"
    n += 1  # segmentasyon
    n += len(cfg.get("morphology", []))
    n += cfg.get("separation", {}).get("method", "none") != "none"
    n += bool(cfg.get("clear_border"))
    return int(n)


def run_validation(selection: pd.DataFrame, baseline_cfg: dict, timing_repeats: int = 3):
    images, refs, labels = experiments.load_group(selection, "validation")
    context = experiments.build_context(selection)
    cands = experiments.candidate_pipelines()
    rows, per_frame = [], []
    entries = list(cands.items()) + [("Baseline", {"desc": "Referans amaçlı (seçime katılmaz)",
                                                    "cfg": baseline_cfg})]
    for name, c in entries:
        m, fr = experiments.evaluate_config(c["cfg"], images, refs, labels, context)
        # Süre ölçümünü daha kararlı hale getirmek için tekrarlı ölçüm (medyan)
        times = []
        for _ in range(timing_repeats):
            _, fr2 = experiments.evaluate_config(c["cfg"], images, refs, None, context)
            times.append(np.mean([r["time_s"] for r in fr2]))
        rows.append({"pipeline": name, "description": c["desc"],
                     "val_MAE": round(m["MAE"], 3), "val_RMSE": round(m["RMSE"], 3),
                     "val_bias": round(m["Bias"], 3), "val_exact_acc": round(m["ExactAcc"], 3),
                     "mean_time_s": round(float(np.median(times)), 4),
                     "complexity_steps": complexity(c["cfg"]),
                     "missed": m.get("missed"), "false_pos": m.get("false_pos"),
                     "merged_objs": m.get("merged"), "split_cells": m.get("split")})
        for r in fr:
            per_frame.append({"pipeline": name, "image_id": r["key"], "ref_count": r["ref_count"],
                              "pred_count": r["pred_count"], "error": r["error"]})
    df = pd.DataFrame(rows)
    cand_df = df[df.pipeline != "Baseline"].sort_values(
        ["val_MAE", "val_RMSE", "mean_time_s", "complexity_steps"])
    winner = cand_df.iloc[0]["pipeline"]
    df["selected"] = np.where(df.pipeline == winner, "SEÇİLDİ", "")
    df.loc[df.pipeline == "Baseline", "selected"] = "(referans)"
    df["rank"] = df.pipeline.map({p: i + 1 for i, p in enumerate(cand_df.pipeline)})
    df.to_csv(config.VALIDATION_CSV, index=False)
    pd.DataFrame(per_frame).to_csv(config.VALIDATION_PER_FRAME_CSV, index=False)
    return df, winner, cands[winner]["cfg"]


def save_final_pipeline(winner: str, final_cfg: dict, baseline_cfg: dict,
                        validation_df: pd.DataFrame, baseline_min_area: int,
                        match_reference_frame: int) -> dict:
    """Nihai pipeline ve baseline'ı sabitleyerek final_pipeline.json'a yazar."""
    v = validation_df.set_index("pipeline")
    doc = {
        "student": {"name": config.STUDENT_NAME, "student_no": config.STUDENT_NO},
        "seed": config.SEED,
        "dataset": {"name": "Cell Tracking Challenge - Fluo-N2DH-GOWT1 (training)",
                    "url": config.DATASET_URL},
        "selected_pipeline": winner,
        "selection_rule": "min validation MAE; eşitlikte RMSE, ortalama süre, karmaşıklık",
        "validation_metrics": {k: (float(v.loc[winner, k]) if k in v.columns else None)
                               for k in ["val_MAE", "val_RMSE", "mean_time_s"]},
        "final_pipeline": copy.deepcopy(final_cfg),
        "final_pipeline_readable": pipeline.describe(final_cfg),
        "baseline_pipeline": copy.deepcopy(baseline_cfg),
        "baseline_min_area_selected_on_dev": baseline_min_area,
        "histogram_match_reference_dev_frame": match_reference_frame,
        "software": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__,
                     "scikit-image": skimage.__version__},
        "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    text = json.dumps(doc, ensure_ascii=False, indent=2)
    config.FINAL_PIPELINE_JSON.write_text(text, encoding="utf-8")
    return doc


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
