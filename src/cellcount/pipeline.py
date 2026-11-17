"""Yapılandırma sözlüğüyle tanımlanan hücre sayma pipeline'ı.

Adım sırası: yoğunluk dönüşümü -> filtre -> segmentasyon -> morfoloji
-> (isteğe bağlı) watershed ayırma -> nesne filtreleme -> sayım.
``order`` anahtarı ``"filter_first"`` ise filtre yoğunluk dönüşümünden önce uygulanır.
"""
from __future__ import annotations

import copy
import time

import numpy as np

from . import ops

# Ödev Bölüm 5: Baseline = gri seviye -> Otsu -> bağlı bileşen -> sayım.
# Sabit minimum alan eşiği (geliştirme verisinde belirlendi, bkz. E00/E00b).
BASELINE: dict = {
    "name": "Baseline",
    "intensity": {"method": "none"},
    "filter": {"method": "none"},
    "segmentation": {"method": "otsu"},
    "morphology": [],
    "separation": {"method": "none"},
    "min_area": 100,
}


def describe(cfg: dict) -> dict:
    """Deney tablosu için okunabilir özet sütunları üretir."""
    def kv(d: dict) -> str:
        p = {k: v for k, v in d.items() if k != "method"}
        return d.get("method", "none") + (f"({', '.join(f'{k}={v}' for k, v in p.items())})" if p else "")

    morph = " -> ".join(
        o["op"] if o["op"] == "fill_holes" else f"{o['op']}{o['size']}" for o in cfg.get("morphology", []))
    sep = cfg.get("separation", {"method": "none"})
    params = [f"min_area={cfg.get('min_area', 0)}"]
    if cfg.get("clear_border"):
        params.append("kenar nesneleri çıkarıldı")
    if cfg.get("order") == "filter_first":
        params.append("filtre->yoğunluk")
    bg = cfg.get("background", {"method": "none"})
    if bg.get("method", "none") != "none":
        params.append(f"arka plan={kv(bg)}")
    return {
        "intensity": kv(cfg.get("intensity", {"method": "none"})),
        "filter": kv(cfg.get("filter", {"method": "none"})),
        "segmentation": kv(cfg["segmentation"]),
        "morphology": morph or "yok",
        "separation": kv(sep),
        "params": "; ".join(params),
    }


def run(img: np.ndarray, cfg: dict, context: dict | None = None,
        keep_stages: bool = False) -> dict:
    """Pipeline'ı tek görüntüde çalıştırır.

    Dönüş: {"count", "labels", "time_s", "stages" (keep_stages ise)}
    """
    cfg = copy.deepcopy(cfg)
    stages: dict = {}
    t0 = time.perf_counter()

    x = img
    steps = ["filter", "intensity"] if cfg.get("order") == "filter_first" else ["intensity", "filter"]
    for step in steps:
        if step == "intensity":
            x = ops.apply_intensity(x, cfg.get("intensity", {"method": "none"}), context)
        else:
            x = ops.apply_filter(x, cfg.get("filter", {"method": "none"}))
        if keep_stages:
            stages[step] = x
    if cfg.get("background", {"method": "none"}).get("method", "none") != "none":
        x = ops.apply_background(x, cfg["background"])
        if keep_stages:
            stages["background"] = x
    mask, seg_info = ops.segment(x, cfg["segmentation"])
    if keep_stages:
        stages["segmentation"] = mask
    mask = ops.apply_morphology(mask, cfg.get("morphology", []))
    if keep_stages:
        stages["morphology"] = mask

    sep = cfg.get("separation", {"method": "none"})
    if sep.get("method", "none") == "watershed":
        labels = ops.watershed_split(mask, sep)
    else:
        labels = ops.label_components(mask)
    labels = ops.filter_objects(labels, cfg.get("min_area", 0), cfg.get("max_area"),
                                cfg.get("clear_border", False))
    elapsed = time.perf_counter() - t0
    out = {"count": int(labels.max()), "labels": labels, "time_s": elapsed, "seg_info": seg_info}
    if keep_stages:
        stages["labels"] = labels
        out["stages"] = stages
    return out
