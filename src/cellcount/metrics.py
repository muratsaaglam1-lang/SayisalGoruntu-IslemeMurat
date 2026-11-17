"""Sayım metrikleri ve nesne düzeyinde hata analizi."""
from __future__ import annotations

import numpy as np


def count_metrics(y_true, y_pred, times=None) -> dict:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_pred - y_true
    out = {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "Bias": float(np.mean(err)),
        "ExactAcc": float(np.mean(err == 0)),
        "MaxAbsErr": float(np.max(np.abs(err))),
    }
    if times is not None:
        out["MeanTime_s"] = float(np.mean(times))
    return out


def object_error_analysis(pred_labels: np.ndarray, gt_labels: np.ndarray) -> dict:
    """Tahmin nesnelerini referans işaretleyicilerle eşleştirerek hata türlerini sayar.

    * missed: hiçbir tahmin nesnesiyle örtüşmeyen referans hücre (FN)
    * false_pos: hiçbir referans hücreyle örtüşmeyen tahmin nesnesi (FP)
    * merged: birden fazla referans hücreyi kapsayan tahmin nesnesi (alt sayım)
    * split: birden fazla tahmin nesnesine bölünen referans hücre (fazla sayım)
    """
    p = pred_labels.ravel()
    g = gt_labels.ravel()
    both = (p > 0) & (g > 0)
    pairs = np.unique(np.stack([p[both], g[both]], 1), axis=0) if both.any() else np.zeros((0, 2), int)
    gt_ids = np.unique(g[g > 0])
    pr_ids = np.unique(p[p > 0])
    gt_hits = {i: 0 for i in gt_ids}
    pr_hits = {i: 0 for i in pr_ids}
    for pi, gi in pairs:
        pr_hits[pi] += 1
        gt_hits[gi] += 1
    return {
        "missed": int(sum(v == 0 for v in gt_hits.values())),
        "false_pos": int(sum(v == 0 for v in pr_hits.values())),
        "merged": int(sum(v >= 2 for v in pr_hits.values())),
        "merged_cells": int(sum(v for v in pr_hits.values() if v >= 2)),
        "split": int(sum(v >= 2 for v in gt_hits.values())),
    }
