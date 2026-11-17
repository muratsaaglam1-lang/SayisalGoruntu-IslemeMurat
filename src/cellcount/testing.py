"""Nihai pipeline'ın kilitlenmesi, test tahminleri ve bağımsız test
değerlendirmesi (ödev Bölüm 12-14).

Akış:
1. :func:`lock_test_predictions` — final_pipeline.json'u okur, test görüntülerinde
   baseline ve nihai pipeline tahminlerini **referanslara erişmeden** hesaplar,
   ``test_predictions_locked.csv`` ve SHA-256 özetli ``test_lock_manifest.json``
   yazar. Bu noktada GitHub'a ayrı bir commit atılır.
2. :func:`evaluate_test` — kilidi doğrular (dosya özetleri değişmemiş olmalı),
   ancak bundan sonra test referanslarını okur ve metrikleri hesaplar.
"""
from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import config, data, experiments, metrics, pipeline, references
from .validation import sha256


def lock_test_predictions(selection: pd.DataFrame, overwrite: bool = False) -> pd.DataFrame:
    if config.TEST_LOCKED_CSV.exists() and not overwrite:
        verify_lock()
        return pd.read_csv(config.TEST_LOCKED_CSV)
    final_doc = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))
    final_cfg, base_cfg = final_doc["final_pipeline"], final_doc["baseline_pipeline"]
    context = experiments.build_context(selection)  # yalnızca geliştirme verisi
    rows = []
    for r in selection[selection.group == "test"].itertuples():
        img = data.read_frame(r.sequence, r.frame)  # yalnızca ham görüntü okunur
        b = pipeline.run(img, base_cfg, context)
        f = pipeline.run(img, final_cfg, context)
        rows.append({"image_id": r.image_id, "sequence": r.sequence, "frame": r.frame,
                     "test_block": r.test_block, "baseline_pred": b["count"],
                     "final_pred": f["count"], "baseline_time_s": round(b["time_s"], 4),
                     "final_time_s": round(f["time_s"], 4)})
    df = pd.DataFrame(rows)
    df.to_csv(config.TEST_LOCKED_CSV, index=False)
    manifest = {
        "locked_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "final_pipeline_sha256": sha256(config.FINAL_PIPELINE_JSON),
        "test_predictions_sha256": sha256(config.TEST_LOCKED_CSV),
        "selected_frames_sha256": sha256(config.SELECTED_FRAMES_CSV),
        "note": "Test referansları bu dosya oluşturulmadan önce okunmamıştır.",
    }
    config.TEST_LOCK_MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    return df


def verify_lock() -> dict:
    """Kilit dosyalarının varlığını ve değişmediğini doğrular."""
    for p in (config.FINAL_PIPELINE_JSON, config.TEST_LOCKED_CSV, config.TEST_LOCK_MANIFEST):
        if not p.exists():
            raise PermissionError(f"Test referansları kilitlemeden önce okunamaz: {p.name} yok.")
    man = json.loads(config.TEST_LOCK_MANIFEST.read_text(encoding="utf-8"))
    checks = {"final_pipeline_sha256": config.FINAL_PIPELINE_JSON,
              "test_predictions_sha256": config.TEST_LOCKED_CSV,
              "selected_frames_sha256": config.SELECTED_FRAMES_CSV}
    for key, path in checks.items():
        if sha256(path) != man[key]:
            raise PermissionError(f"{path.name} kilitlendikten sonra değiştirilmiş!")
    return man


def evaluate_test(selection: pd.DataFrame):
    locked = pd.read_csv(config.TEST_LOCKED_CSV)
    refs = references.load_test_references(selection)  # verify_lock() içeride çağrılır
    df = locked.merge(refs[["image_id", "ref_count"]], on="image_id")
    df["baseline_abs_err"] = (df.baseline_pred - df.ref_count).abs()
    df["final_abs_err"] = (df.final_pred - df.ref_count).abs()
    df["baseline_err"] = df.baseline_pred - df.ref_count
    df["final_err"] = df.final_pred - df.ref_count
    df["frame_label"] = [f"Frame {i}" for i in range(1, len(df) + 1)]
    cols = ["frame_label", "image_id", "frame", "test_block", "ref_count", "baseline_pred",
            "final_pred", "baseline_abs_err", "final_abs_err", "baseline_err", "final_err",
            "baseline_time_s", "final_time_s"]
    df = df[cols]

    summ = []
    for name, col, tcol in (("Baseline", "baseline_pred", "baseline_time_s"),
                            ("Nihai pipeline", "final_pred", "final_time_s")):
        m = metrics.count_metrics(df.ref_count, df[col], df[tcol])
        summ.append({"method": name, "MAE": round(m["MAE"], 3), "RMSE": round(m["RMSE"], 3),
                     "Bias": round(m["Bias"], 3), "ExactCountAccuracy": round(m["ExactAcc"], 3),
                     "MeanTime_s": round(m["MeanTime_s"], 4)})
    summary = pd.DataFrame(summ)

    # test_results.csv: kare bazlı sonuçlar; test_summary.csv: özet metrikler
    df.to_csv(config.TEST_RESULTS_CSV, index=False)
    summary.to_csv(config.TEST_SUMMARY_CSV, index=False)
    return df, summary


def object_level_test_analysis(selection: pd.DataFrame) -> pd.DataFrame:
    """Kilit sonrası: test karelerinde nesne düzeyinde hata türleri (yorum için)."""
    verify_lock()
    final_doc = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))
    context = experiments.build_context(selection)
    rows = []
    for r in selection[selection.group == "test"].itertuples():
        img = data.read_frame(r.sequence, r.frame)
        gt = data._read_tra(r.sequence, r.frame)
        for name, cfg in (("Baseline", final_doc["baseline_pipeline"]),
                          ("Nihai pipeline", final_doc["final_pipeline"])):
            out = pipeline.run(img, cfg, context)
            rows.append({"method": name, "image_id": r.image_id, "frame": r.frame,
                         **metrics.object_error_analysis(out["labels"], gt)})
    return pd.DataFrame(rows)


def temporal_analysis(test_df: pd.DataFrame) -> pd.DataFrame:
    """Blok içi ardışık kare sayım değişimleri (Δ) ve tutarlılık ölçütleri."""
    rows = []
    for blk, g in test_df.groupby("test_block"):
        g = g.sort_values("frame")
        for name, col in (("Gerçek", "ref_count"), ("Baseline", "baseline_pred"),
                          ("Nihai pipeline", "final_pred")):
            v = g[col].to_numpy()
            d = np.diff(v)
            rows.append({"test_block": blk, "frames": f"{g.frame.min()}-{g.frame.max()}",
                         "series": name, "counts": " ".join(map(str, v)),
                         "deltas": " ".join(f"{x:+d}" for x in d),
                         "mean_abs_delta": round(float(np.mean(np.abs(d))), 3),
                         "n_changes": int(np.sum(d != 0)),
                         "range": int(v.max() - v.min())})
        gt_d = np.diff(g.ref_count.to_numpy())
        for name, col in (("Baseline", "baseline_pred"), ("Nihai pipeline", "final_pred")):
            pd_d = np.diff(g[col].to_numpy())
            rows.append({"test_block": blk, "frames": f"{g.frame.min()}-{g.frame.max()}",
                         "series": f"{name} Δ-uyumu",
                         "counts": "", "deltas": "",
                         "mean_abs_delta": round(float(np.mean(np.abs(pd_d - gt_d))), 3),
                         "n_changes": int(np.sum(np.sign(pd_d) != np.sign(gt_d))),
                         "range": None})
    out = pd.DataFrame(rows)
    out.to_csv(config.TEMPORAL_CSV, index=False)
    return out
