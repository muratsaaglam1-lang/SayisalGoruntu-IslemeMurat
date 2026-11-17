"""Tüm aşamaların sırayla çalıştırılması.

Kullanım (repo kökünden):
    python run_pipeline.py --stage until-lock   # 2-12. adımlar (test referansı okunmaz)
    # >>> burada GitHub'a "kilit" commit'i atılır <<<
    python run_pipeline.py --stage evaluate     # 13-15. adımlar + şekiller
    python run_pipeline.py --stage all          # hepsi (kilit dosyası varsa yeniden üretilmez)
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from . import (config, data, eda, experiments, figures, posthoc, references, selection, testing,
               validation)


def stage_dataset():
    data.check_dataset()
    config.ensure_dirs()
    stats = eda.image_statistics()
    figures.fig_dataset_histograms(eda.sequence_histograms())
    return stats


def stage_selection():
    sel = selection.run_selection()
    selection.verify_selection(sel)
    return sel


def stage_references(sel):
    refs = references.build_dev_val_references(sel)
    cells = eda.cell_statistics(sel)
    return refs, cells


def stage_experiments(sel):
    return experiments.run_experiments(sel)


def stage_validation(sel, baseline_min_area):
    from .pipeline import BASELINE
    base_cfg = {**BASELINE, "min_area": int(baseline_min_area)}
    val_df, winner, final_cfg = validation.run_validation(sel, base_cfg)
    if config.TEST_LOCK_MANIFEST.exists():
        # Kilit sonrası yeniden çalıştırma: sabitlenmiş dosya ASLA yeniden yazılmaz,
        # yalnızca yeniden üretilen seçimin kilitli seçimle aynı olduğu doğrulanır.
        testing.verify_lock()
        doc = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))
        same = (doc["selected_pipeline"] == winner
                and {k: v for k, v in doc["final_pipeline"].items()} == final_cfg
                and doc["baseline_pipeline"] == base_cfg)
        if not same:
            raise RuntimeError("Yeniden üretilen seçim kilitli final_pipeline.json ile uyuşmuyor!")
        return val_df, winner, doc
    ctx = experiments.build_context(sel)
    doc = validation.save_final_pipeline(winner, final_cfg, base_cfg, val_df, baseline_min_area,
                                         ctx["match_reference_frame"])
    return val_df, winner, doc


def stage_lock(sel):
    return testing.lock_test_predictions(sel)


def stage_evaluate(sel):
    test_df, summary = testing.evaluate_test(sel)
    temporal = testing.temporal_analysis(test_df)
    obj = testing.object_level_test_analysis(sel)
    obj.to_csv(config.REPO_ROOT / "test_object_errors.csv", index=False)
    # Kilit sonrası hata analizi (yalnızca yorum; tahminleri değiştirmez)
    posthoc.background_noise(sel)
    posthoc.false_positive_shapes(sel)
    posthoc.candidates_on_test(sel)
    return test_df, summary, temporal, obj


def stage_figures(sel, test_df=None):
    config.ensure_dirs()
    doc = json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))
    ctx = experiments.build_context(sel)
    exp = pd.read_csv(config.EXPERIMENTS_CSV)
    calib = pd.read_csv(config.REPO_ROOT / "baseline_min_area_calibration.csv")
    val = pd.read_csv(config.VALIDATION_CSV)
    refs = references.load_dev_val_references()
    demo = sel[sel.group == "development"].iloc[3]  # temsili geliştirme karesi
    demo_ref = int(refs.set_index("image_id").loc[demo.image_id, "ref_count"])
    test_first = int(sel[sel.group == "test"].frame.iloc[0])
    paths = [
        figures.fig1_flow(doc),
        figures.fig2_intensity(int(demo.frame), ctx),
        figures.fig_filters(int(demo.frame)),
        figures.fig3_seg_morph(int(demo.frame), demo_ref, ctx),
        figures.fig4_steps(doc, int(demo.frame), ctx),
        figures.fig_dataset_samples(int(demo.frame), test_first),
        figures.fig_experiments_mae(exp),
        figures.fig_param_sensitivity(exp, calib),
        figures.fig_validation(val),
    ]
    if test_df is not None:
        paths.append(figures.fig5_temporal(test_df))
        paths.append(figures.fig6_examples(doc, test_df, ctx))
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["until-lock", "evaluate", "all"], default="all")
    args = ap.parse_args()
    if args.stage in ("until-lock", "all"):
        print(stage_dataset().to_string())
        sel = stage_selection()
        stage_references(sel)
        exp, _, bma, _ = stage_experiments(sel)
        print(f"{len(exp)} deney tamamlandı; baseline min_area={bma}")
        val_df, winner, _ = stage_validation(sel, bma)
        print(val_df.to_string())
        print("Seçilen:", winner)
        print(stage_lock(sel).to_string())
        print(">>> Kilit tamamlandı. Şimdi final_pipeline.json, test_predictions_locked.csv ve "
              "test_lock_manifest.json dosyalarını AYRI bir commit ile GitHub'a gönderin.")
    if args.stage in ("evaluate", "all"):
        sel = selection.load_selection()
        test_df, summary, temporal, _ = stage_evaluate(sel)
        print(test_df.to_string())
        print(summary.to_string())
        print(temporal.to_string())
        for p in stage_figures(sel, test_df):
            print("şekil:", p)


if __name__ == "__main__":
    main()
