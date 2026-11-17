"""Rapor şekilleri.

* ``figures/``        : veri setinin görüntülerini içermeyen grafikler (GitHub'a uygun)
* ``report_figures/`` : orijinal görüntülerden üretilen şekiller (yalnızca rapor; .gitignore)
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config, data, experiments, ops, pipeline  # noqa: E402

# Doğrulanmış kategorik palet (açık tema) ve metin tonları
C_FINAL = "#2a78d6"
C_BASE = "#eb6834"
C_GT = "#2b2b29"
GROUP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
TXT2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
    "axes.edgecolor": "#8a8984", "axes.labelcolor": C_GT, "xtick.color": TXT2,
    "ytick.color": TXT2, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
})

CROP = (slice(380, 892), slice(400, 912))  # geliştirme karesinde temsili bölge


SHOW_TITLES = True  # rapor için False yapılır (başlıklar Word altyazısında verilir)


def _suptitle(fig, text, **kw):
    if SHOW_TITLES:
        fig.suptitle(text, **kw)


def _save(fig, path):
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def disp(img: np.ndarray) -> np.ndarray:
    """Yalnızca görselleştirme için: log ölçekli gösterim."""
    return ops.log_transform(cv2.GaussianBlur(img, (3, 3), 0))


def overlay(base_gray: np.ndarray, labels: np.ndarray | None, color=(42, 120, 214),
            gt: np.ndarray | None = None, thickness: int = 3) -> np.ndarray:
    rgb = cv2.cvtColor(base_gray, cv2.COLOR_GRAY2RGB)
    if labels is not None:
        for lab in range(1, int(labels.max()) + 1):
            m = (labels == lab).astype(np.uint8)
            cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(rgb, cs, -1, color, thickness)
    if gt is not None:
        for g in np.unique(gt)[1:]:
            ys, xs = np.nonzero(gt == g)
            cv2.circle(rgb, (int(xs.mean()), int(ys.mean())), 9, (227, 73, 72), -1)
    return rgb


def _exp_lookup():
    exp = pd.read_csv(config.EXPERIMENTS_CSV)
    return exp, {r.changed_factor: r for r in exp.itertuples()}


# ---------------------------------------------------------------------------
# Veri seti histogramları (GitHub'a uygun)
# ---------------------------------------------------------------------------
def fig_dataset_histograms(hists: dict) -> str:
    fig, ax = plt.subplots(figsize=(7, 3))
    for seq, col in (("01", C_FINAL), ("02", C_BASE)):
        h = hists[seq] / hists[seq].sum()
        ax.semilogy(np.arange(256), np.maximum(h, 1e-9), color=col, lw=2,
                    label=f"Sequence {seq} (tüm kareler)")
    ax.set_xlabel("Piksel yoğunluğu (uint8)")
    ax.set_ylabel("Göreli frekans (log)")
    ax.set_title("Sequence 01 ve 02 yoğunluk histogramları")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False)
    return _save(fig, config.FIG_DIR / "fig_dataset_histograms.png")


def fig_dataset_samples(dev_frame: int, test_frame: int) -> str:
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4), layout="constrained")
    a = data.read_frame("01", dev_frame)
    b = data.read_frame("02", test_frame)
    axs[0].imshow(a, cmap="gray", vmin=0, vmax=255)
    axs[0].set_title(f"Seq 01 t{dev_frame:03d} — ham (0–255 gösterim)")
    axs[1].imshow(disp(a), cmap="gray")
    axs[1].set_title(f"Seq 01 t{dev_frame:03d} — log gösterim")
    axs[2].imshow(disp(b), cmap="gray")
    axs[2].set_title(f"Seq 02 t{test_frame:03d} — log gösterim (yalnızca görüntü)")
    for ax in axs:
        ax.axis("off")
    return _save(fig, config.REPORT_FIG_DIR / "fig_dataset_samples.png")


# ---------------------------------------------------------------------------
# Şekil 1: Nihai pipeline akış diyagramı
# ---------------------------------------------------------------------------
def fig1_flow(final_doc: dict) -> str:
    cfg = final_doc["final_pipeline"]

    def kv(d):
        return ", ".join(f"{k}={v}" for k, v in d.items() if k != "method")

    steps = [("Girdi", "Ham gri görüntü\n1024×1024, uint8")]
    order = ["filter", "intensity"] if cfg.get("order") == "filter_first" else ["intensity", "filter"]
    names = {"filter": "Filtre", "intensity": "Yoğunluk dönüşümü"}
    for s in order:
        d = cfg.get(s, {"method": "none"})
        if d.get("method", "none") != "none":
            steps.append((names[s], f"{d['method']}\n{kv(d)}"))
    bg = cfg.get("background", {"method": "none"})
    if bg.get("method", "none") != "none":
        steps.append(("Arka plan", f"{bg['method']}\n{kv(bg)}"))
    seg = cfg["segmentation"]
    steps.append(("Segmentasyon", f"{seg['method']}\n{kv(seg)}"))
    morph = cfg.get("morphology", [])
    if morph:
        steps.append(("Morfoloji", "\n".join(
            "doldurma (fill holes)" if o["op"] == "fill_holes" else f"{o['op']} (elips {o['size']}×{o['size']})"
            for o in morph)))
    sep = cfg.get("separation", {"method": "none"})
    if sep.get("method", "none") != "none":
        steps.append(("Ayırma", f"distance transform +\nwatershed ({kv(sep)})"))
    post = [f"alan ≥ {cfg.get('min_area', 0)} px"]
    if cfg.get("clear_border"):
        post.append("kenara değenler çıkarılır")
    steps.append(("Nesne filtresi", "CCL (8-komşuluk)\n" + "\n".join(post)))
    steps.append(("Çıktı", "Hücre sayısı\n(etiket sayısı)"))

    n = len(steps)
    per_row = int(np.ceil(n / 2))
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, per_row * 3.2)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    pos = []
    for i, (title, body) in enumerate(steps):
        row, col = divmod(i, per_row)
        if row == 1:
            col = per_row - 1 - col  # yılan düzeni
        x = 0.2 + col * 3.2
        y = 3.6 if row == 0 else 0.3
        fc = "#eaf2fc" if title not in ("Girdi", "Çıktı") else "#f4f3f0"
        ax.add_patch(FancyBboxPatch((x, y), 2.6, 2.4, boxstyle="round,pad=0.04,rounding_size=0.15",
                                    fc=fc, ec=C_FINAL if fc == "#eaf2fc" else "#8a8984", lw=1.4))
        ax.text(x + 1.3, y + 2.05, f"{i + 1}. {title}", ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=C_GT)
        ax.text(x + 1.3, y + 0.95, body, ha="center", va="center", fontsize=8, color=TXT2)
        pos.append((x, y, row))
    for (x0, y0, r0), (x1, y1, r1) in zip(pos, pos[1:]):
        if r0 == r1:
            if r0 == 0:
                a, b = (x0 + 2.6, y0 + 1.2), (x1, y1 + 1.2)
            else:
                a, b = (x0, y0 + 1.2), (x1 + 2.6, y1 + 1.2)
        else:
            a, b = (x0 + 1.3, y0), (x1 + 1.3, y1 + 2.4)
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color=TXT2, lw=1.3))
    if SHOW_TITLES:
        ax.set_title(f"Şekil 1. Nihai pipeline ({final_doc['selected_pipeline']}) akış diyagramı",
                     fontsize=11, loc="left")
    return _save(fig, config.FIG_DIR / "fig1_final_pipeline_flow.png")


# ---------------------------------------------------------------------------
# Şekil 2: Yoğunluk dönüşümleri + histogramlar
# ---------------------------------------------------------------------------
def fig2_intensity(frame: int, context: dict) -> str:
    _, lk = _exp_lookup()
    img = data.read_frame("01", frame)
    items = [("Orijinal", {"method": "none"}, "Dönüşüm yok (orijinal)"),
             ("Contrast stretching\n(p1–p99.8)", {"method": "stretch"}, "Contrast stretching"),
             ("Histogram eşitleme", {"method": "histeq"}, "Histogram equalization"),
             ("CLAHE\n(clip=2, 8×8)", {"method": "clahe"}, "CLAHE"),
             (f"Histogram eşleme\n(ref: dev t{context['match_reference_frame']:03d})", {"method": "match"}, "Histogram matching"),
             ("Gamma (γ=0.5)", {"method": "gamma", "gamma": 0.5}, "Gamma (0.5)"),
             ("Log (c·log(1+r))", {"method": "log"}, "Log dönüşümü")]
    fig, axs = plt.subplots(2, len(items), figsize=(17, 5.2), layout="constrained",
                            gridspec_kw={"height_ratios": [3, 1.3]})
    for j, (title, cfg, key) in enumerate(items):
        x = ops.apply_intensity(img, cfg, context)
        axs[0, j].imshow(x[CROP], cmap="gray", vmin=0, vmax=255)
        mae = lk[key].count_MAE if key in lk else float("nan")
        axs[0, j].set_title(f"{title}\nDev MAE={mae:.1f}", fontsize=8.5)
        axs[0, j].axis("off")
        h = np.bincount(x.ravel(), minlength=256).astype(float)
        nz = h > 0
        axs[1, j].vlines(np.arange(256)[nz], 1, h[nz], color=C_FINAL, lw=1.0)
        axs[1, j].set_yscale("log")
        axs[1, j].set_ylim(1, 2e6)
        axs[1, j].set_xlim(0, 255)
        axs[1, j].tick_params(labelsize=7)
        if j == 0:
            axs[1, j].set_ylabel("piksel sayısı (log)", fontsize=7.5)
    _suptitle(fig, f"Şekil 2. Yoğunluk dönüşümleri (geliştirme karesi t{frame:03d}, 512×512 kesit) "
                 "ve tüm karenin histogramları", x=0.01, ha="left", fontsize=11)
    return _save(fig, config.REPORT_FIG_DIR / "fig2_intensity_transforms.png")


# ---------------------------------------------------------------------------
# Filtre karşılaştırması
# ---------------------------------------------------------------------------
def fig_filters(frame: int) -> str:
    _, lk = _exp_lookup()
    img = data.read_frame("01", frame)
    items = [("Filtre yok", {"method": "none"}, "Filtre yok"),
             ("Gaussian 3×3", {"method": "gaussian", "ksize": 3}, "Gaussian 3x3"),
             ("Gaussian 5×5", {"method": "gaussian", "ksize": 5}, "Gaussian 5x5"),
             ("Gaussian 9×9", {"method": "gaussian", "ksize": 9}, "Log dönüşümü"),
             ("Gaussian 13×13", {"method": "gaussian", "ksize": 13}, "Gaussian 13x13"),
             ("Median 5×5", {"method": "median", "ksize": 5}, "Median 5x5"),
             ("Median 9×9", {"method": "median", "ksize": 9}, "Median 9x9"),
             ("Laplacian keskinleştirme", {"method": "laplacian_sharpen"}, "Laplacian keskinleştirme"),
             ("G9 + Laplacian", {"method": "gaussian_then_laplacian", "ksize": 9}, "Gaussian 9x9 + Laplacian"),
             ("G9 + unsharp", {"method": "gaussian_then_unsharp", "ksize": 9}, "Gaussian 9x9 + unsharp")]
    fig, axs = plt.subplots(2, 5, figsize=(13, 6.4), layout="constrained")
    cut = (slice(CROP[0].start, CROP[0].start + 256), slice(CROP[1].start + 100, CROP[1].start + 356))
    for ax, (title, cfg, key) in zip(axs.ravel(), items):
        x = ops.log_transform(ops.apply_filter(img, cfg))
        ax.imshow(x[cut], cmap="gray", vmin=0, vmax=255)
        r = lk.get(key)
        ax.set_title(f"{title}\nDev MAE={r.count_MAE:.1f}, FP={r.false_pos}" if r is not None else title,
                     fontsize=8.5)
        ax.axis("off")
    _suptitle(fig, f"Filtrelerin etkisi (filtre → log; t{frame:03d}, 256×256 kesit)", x=0.01,
                 ha="left", fontsize=11)
    return _save(fig, config.REPORT_FIG_DIR / "fig_filters.png")


# ---------------------------------------------------------------------------
# Şekil 3: Segmentasyon ve morfoloji çıktıları
# ---------------------------------------------------------------------------
def fig3_seg_morph(frame: int, ref_count: int, context: dict) -> str:
    exp, _ = _exp_lookup()
    defs = {e["id"]: e for e in experiments.experiment_definitions(
        int(json.loads(config.FINAL_PIPELINE_JSON.read_text(encoding="utf-8"))
            ["baseline_min_area_selected_on_dev"]))}
    erow = exp.set_index("experiment_id")
    img = data.read_frame("01", frame)
    seg_ids = [e for e in defs if defs[e]["group"] == "8-Segmentasyon"][:8]
    morph_ids = [e for e in defs if defs[e]["group"] == "9-Morfoloji"
                 and defs[e]["factor"] in ("Morfoloji yok", "Erosion 5", "Dilation 5", "Opening 7",
                                           "Closing 11", "Closing 11 -> Opening 7 (ters sıra)",
                                           "Open7 -> Close11 -> Fill", "Watershed (DT, h=3, yumuşatma yok)")]
    fig, axs = plt.subplots(2, 8, figsize=(19, 6.0), layout="constrained")
    for row, ids in enumerate((seg_ids, morph_ids)):
        for ax, eid in zip(axs[row], ids):
            out = pipeline.run(img, defs[eid]["cfg"], context, keep_stages=True)
            base = (out["stages"]["segmentation"] if row == 0 else out["stages"]["morphology"])
            vis = (base[CROP] * 255).astype(np.uint8)
            vis = overlay(vis // 2, out["labels"][CROP], color=(42, 120, 214), thickness=2)
            ax.imshow(vis)
            short = (defs[eid]["factor"].replace(" (ters sıra)", "")
                     .replace("Watershed (DT, h=3, yumuşatma yok)", "Watershed (σ=0, h=3)")
                     .replace("Open7 -> Close11 -> Fill", "Open7→Close11→Fill")
                     .replace("Closing 11 -> Opening 7", "Close11→Open7 (ters)")
                     .replace("K-means (k=4, 2 arka plan)", "K-means (k=4)")
                     .replace("Adaptif ortalama", "Adapt. ortalama")
                     .replace("Adaptif Gauss", "Adapt. Gauss"))
            ax.set_title(f"{eid}: {short}\nn={out['count']} (ref {ref_count}), "
                         f"Dev MAE={erow.loc[eid, 'count_MAE']:.1f}", fontsize=8)
            ax.axis("off")
    axs[0, 0].text(-0.08, 0.5, "Segmentasyon\n(maske)", transform=axs[0, 0].transAxes, rotation=90,
                   va="center", ha="right", fontsize=9, fontweight="bold")
    axs[1, 0].text(-0.08, 0.5, "Morfoloji\n(adaptif Gauss maskesi)", transform=axs[1, 0].transAxes,
                   rotation=90, va="center", ha="right", fontsize=9, fontweight="bold")
    _suptitle(fig, f"Şekil 3. Segmentasyon (üst) ve morfolojik işlem (alt) çıktıları — t{frame:03d}, "
                 "beyaz: maske, mavi kontur: sayılan nesneler", x=0.01, ha="left", fontsize=11)
    return _save(fig, config.REPORT_FIG_DIR / "fig3_segmentation_morphology.png")


# ---------------------------------------------------------------------------
# Şekil 4: Nihai pipeline adım adım
# ---------------------------------------------------------------------------
def fig4_steps(final_doc: dict, frame: int, context: dict) -> str:
    cfg = final_doc["final_pipeline"]
    img = data.read_frame("01", frame)
    gt = data._read_tra("01", frame)
    out = pipeline.run(img, cfg, context, keep_stages=True)
    st = out["stages"]
    panels = [("1) Ham görüntü", img, dict(cmap="gray", vmin=0, vmax=255))]
    names = {"filter": "Gaussian filtre", "intensity": "Log dönüşümü", "background": "Top-hat"}
    k = 2
    for key in ("filter", "intensity", "background"):
        if key in st:
            panels.append((f"{k}) {names[key]}", st[key], dict(cmap="gray")))
            k += 1
    panels.append((f"{k}) Segmentasyon maskesi", st["segmentation"], dict(cmap="gray")))
    panels.append((f"{k + 1}) Morfoloji sonrası", st["morphology"], dict(cmap="gray")))
    n_gt = len(np.unique(gt)) - 1
    panels.append((f"{k + 2}) Sonuç: n={out['count']} (ref {n_gt})",
                   overlay(disp(img), out["labels"], gt=gt, thickness=4), {}))
    fig, axs = plt.subplots(1, len(panels), figsize=(2.9 * len(panels), 3.6), layout="constrained")
    for ax, (t, im, kw) in zip(axs, panels):
        ax.imshow(im, **kw)
        ax.set_title(t, fontsize=9)
        ax.axis("off")
    _suptitle(fig, f"Şekil 4. Nihai pipeline'ın ara çıktıları (geliştirme karesi t{frame:03d}); "
                 "mavi: tespit, kırmızı nokta: referans hücre", x=0.01, ha="left", fontsize=11)
    return _save(fig, config.REPORT_FIG_DIR / "fig4_final_pipeline_steps.png")


# ---------------------------------------------------------------------------
# Şekil 5: Zamansal analiz
# ---------------------------------------------------------------------------
def fig5_temporal(test_df: pd.DataFrame) -> str:
    blocks = sorted(test_df.test_block.unique())
    fig, axs = plt.subplots(1, len(blocks), figsize=(13, 3.9), sharey=True, layout="constrained")
    lo = min(test_df[["ref_count", "baseline_pred", "final_pred"]].min()) - 2
    hi = max(test_df[["ref_count", "baseline_pred", "final_pred"]].max()) + 2
    for ax, b in zip(axs, blocks):
        g = test_df[test_df.test_block == b].sort_values("frame")
        ax.plot(g.frame, g.ref_count, "-o", color=C_GT, lw=2, ms=7, label="Gerçek (TRA)")
        ax.plot(g.frame, g.baseline_pred, "-s", color=C_BASE, lw=2, ms=7, label="Baseline")
        ax.plot(g.frame, g.final_pred, "-^", color=C_FINAL, lw=2, ms=8, label="Nihai pipeline")
        ax.set_title(f"Blok {b}: kare {g.frame.min()}–{g.frame.max()}")
        ax.set_xlabel("Kare numarası (Sequence 02)")
        ax.set_xticks(g.frame)
        ax.set_ylim(lo, hi)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axs[0].set_ylabel("Hücre sayısı")
    axs[-1].legend(frameon=False, loc="best")
    _suptitle(fig, "Şekil 5. Üç test bloğunda gerçek ve tahmin edilen hücre sayılarının zamansal değişimi",
                 x=0.01, ha="left", fontsize=11)
    return _save(fig, config.FIG_DIR / "fig5_temporal_counts.png")


# ---------------------------------------------------------------------------
# Şekil 6: Başarılı / hatalı / zorlayıcı örnekler (test, kilit sonrası)
# ---------------------------------------------------------------------------
def pick_examples(test_df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    d = test_df.copy()
    succ = d.sort_values(["final_abs_err", "baseline_abs_err"], ascending=[True, False]).iloc[0]
    fail = d.sort_values(["final_abs_err", "frame"], ascending=[False, True]).iloc[0]
    rest = d[~d.image_id.isin([succ.image_id, fail.image_id])]
    chal = rest.sort_values(["baseline_abs_err", "final_abs_err"], ascending=False).iloc[0]
    return [("Başarılı", succ), ("Hatalı", fail), ("Zorlayıcı", chal)]


def fig6_examples(final_doc: dict, test_df: pd.DataFrame, context: dict) -> str:
    from .testing import verify_lock
    verify_lock()
    ex = pick_examples(test_df)
    fig, axs = plt.subplots(2, 3, figsize=(13.5, 9.6), layout="constrained")
    for j, (kind, r) in enumerate(ex):
        img = data.read_frame("02", int(r.frame))
        gt = data._read_tra("02", int(r.frame))
        for i, (name, cfg, col) in enumerate((("Baseline", final_doc["baseline_pipeline"], (235, 104, 52)),
                                              ("Nihai", final_doc["final_pipeline"], (42, 120, 214)))):
            out = pipeline.run(img, cfg, context)
            axs[i, j].imshow(overlay(disp(img), out["labels"], color=col, gt=gt, thickness=4))
            axs[i, j].set_title(f"{kind} — {r.image_id} (t{int(r.frame):03d}, blok {r.test_block})\n"
                                f"{name}: tahmin {out['count']} | gerçek {int(r.ref_count)}",
                                fontsize=9)
            axs[i, j].axis("off")
    _suptitle(fig, "Şekil 6. Başarılı, hatalı ve zorlayıcı test örnekleri (üst: baseline, alt: nihai); "
                 "kırmızı nokta: referans hücre", x=0.01, ha="left", fontsize=11)
    return _save(fig, config.REPORT_FIG_DIR / "fig6_examples.png")


# ---------------------------------------------------------------------------
# Ek grafikler (GitHub'a uygun)
# ---------------------------------------------------------------------------
def fig_experiments_mae(exp: pd.DataFrame) -> str:
    groups = list(dict.fromkeys(exp.group))
    cmap = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(groups)}
    fig, ax = plt.subplots(figsize=(13, 4.2))
    x = np.arange(len(exp))
    mae = exp.count_MAE.clip(upper=15)
    ax.bar(x, mae, color=[cmap[g] for g in exp.group], width=0.75)
    for xi, v in zip(x, exp.count_MAE):
        if v > 15:
            ax.text(xi, 15.2, f"{v:.0f}", ha="center", fontsize=7, color=TXT2)
    ax.set_xticks(x)
    ax.set_xticklabels(exp.experiment_id, rotation=90, fontsize=7.5)
    ax.set_ylabel("Geliştirme Count MAE (15'te kırpıldı)")
    ax.axhline(exp.count_MAE.iloc[0], color=C_BASE, lw=1, ls="--")
    ax.text(len(exp) - 0.5, exp.count_MAE.iloc[0] + 0.3, "baseline", color=TXT2, ha="right", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[g]) for g in groups]
    ax.set_ylim(0, 16.5)
    ax.legend(handles, groups, frameon=False, ncol=len(groups), loc="lower left",
              bbox_to_anchor=(0, 1.0), fontsize=8)
    ax.set_title("Kontrollü deneylerin geliştirme kümesi sayım hatası (10 kare)", pad=24)
    return _save(fig, config.FIG_DIR / "fig_experiments_mae.png")


def fig_param_sensitivity(exp: pd.DataFrame, calib: pd.DataFrame) -> str:
    e = exp.set_index("changed_factor")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), layout="constrained")
    # (a) baseline min_area
    c = calib[calib.min_area > 0]
    axs[0].plot(c.min_area, c.MAE, "-o", color=C_BASE, lw=2, ms=6)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("min_area (px, log)")
    axs[0].set_title("(a) Baseline: minimum alan")
    # (b) adaptif C
    cs = [(-15, "Adaptif Gauss (151, C=-15)"), (-20, "Adaptif Gauss (151, C=-20)"),
          (-25, "Adaptif Gauss (151, C=-25)")]
    axs[1].plot([v for v, _ in cs], [e.loc[k, "count_MAE"] for _, k in cs], "-o", color=C_FINAL, lw=2, ms=6)
    axs[1].set_xlabel("C (blok=151)")
    axs[1].set_title("(b) Adaptif eşik: C ofseti")
    # (c) adaptif blok boyutu
    bs = [(101, "Adaptif Gauss (101, C=-20)"), (151, "Adaptif Gauss (151, C=-20)"),
          (201, "Adaptif Gauss (201, C=-20)")]
    axs[2].plot([v for v, _ in bs], [e.loc[k, "count_MAE"] for _, k in bs], "-o", color=C_FINAL, lw=2, ms=6)
    axs[2].set_xlabel("blok boyutu (C=-20)")
    axs[2].set_title("(c) Adaptif eşik: blok boyutu")
    # (d) Gaussian kernel ve opening SE
    gk = [(0, "Filtre yok"), (3, "Gaussian 3x3"), (5, "Gaussian 5x5"), (9, "Log dönüşümü"),
          (13, "Gaussian 13x13")]
    axs[3].plot([v for v, _ in gk], [e.loc[k, "count_MAE"] for _, k in gk], "-o", color="#1baf7a",
                lw=2, ms=6, label="Gaussian kernel (Otsu çerçevesi)")
    se = [(3, "Opening 3"), (7, "Opening 7"), (11, "Opening 11")]
    axs[3].plot([v for v, _ in se], [e.loc[k, "count_MAE"] for _, k in se], "-s", color="#6250d6",
                lw=2, ms=6, label="Opening SE boyutu (adaptif çerçeve)")
    axs[3].set_xlabel("kernel / SE boyutu (px)")
    axs[3].set_title("(d) Kernel ve yapılandırma elemanı")
    axs[3].legend(frameon=False, fontsize=7.5)
    for ax in axs:
        ax.set_ylabel("Dev MAE")
        ax.grid(axis="y", color=GRID, lw=0.6)
    return _save(fig, config.FIG_DIR / "fig_parameter_sensitivity.png")


def fig_validation(val: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    x = np.arange(len(val))
    ax.bar(x - 0.18, val.val_MAE, 0.34, color=C_FINAL, label="MAE")
    ax.bar(x + 0.18, val.val_RMSE, 0.34, color="#1baf7a", label="RMSE")
    for xi, a, b in zip(x, val.val_MAE, val.val_RMSE):
        ax.text(xi - 0.18, a + 0.05, f"{a:.2f}", ha="center", fontsize=7.5, color=TXT2)
        ax.text(xi + 0.18, b + 0.05, f"{b:.2f}", ha="center", fontsize=7.5, color=TXT2)
    ax.set_xticks(x)
    ax.set_xticklabels([p + (" ✓" if s == "SEÇİLDİ" else "") for p, s in zip(val.pipeline, val.selected)])
    ax.set_ylabel("Sayım hatası (hücre)")
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False)
    ax.set_title("Aday pipeline'ların doğrulama performansı (5 kare)")
    return _save(fig, config.FIG_DIR / "fig_validation.png")
