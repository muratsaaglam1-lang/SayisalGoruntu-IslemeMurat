"""Klasik görüntü işleme operatörleri.

Her fonksiyon saf (yan etkisiz) çalışır; pipeline adımları yapılandırma
sözlüklerindeki ``method`` anahtarına göre bu fonksiyonlara yönlendirilir.
Yoğunluk dönüşümleri ve filtreler uint8 (0-255) çıktı üretir, böylece sabit
eşik değerleri her konfigürasyonda aynı ölçekte yorumlanabilir.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure, segmentation
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 1) Yoğunluk dönüşümleri
# ---------------------------------------------------------------------------
def contrast_stretch(img: np.ndarray, p_low: float = 1.0, p_high: float = 99.8) -> np.ndarray:
    """Doğrusal kontrast germe: [p_low, p_high] persentilleri -> [0, 255]."""
    lo, hi = np.percentile(img, [p_low, p_high])
    hi = max(hi, lo + 1e-6)
    return _to_uint8((img.astype(np.float32) - lo) * 255.0 / (hi - lo))


def hist_equalize(img: np.ndarray) -> np.ndarray:
    """Global histogram eşitleme (cv2.equalizeHist)."""
    return cv2.equalizeHist(img)


def clahe(img: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Kontrast sınırlı adaptif histogram eşitleme."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile)).apply(img)


def hist_match(img: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Histogram eşleme; referans görüntü yalnızca geliştirme kümesinden seçilir."""
    return _to_uint8(exposure.match_histograms(img, reference))


def log_transform(img: np.ndarray, p_high: float = 99.9) -> np.ndarray:
    """s = c*log(1+r); c görüntü başına p_high persentiline göre ayarlanır.
    Düşük yoğunluklu (soluk) çekirdekleri parlak olanlara yaklaştırır."""
    lg = np.log1p(img.astype(np.float32))
    hi = max(np.percentile(lg, p_high), 1e-6)
    return _to_uint8(lg * 255.0 / hi)


def gamma_transform(img: np.ndarray, gamma: float = 0.5, p_high: float = 99.9) -> np.ndarray:
    """Güç yasası (gamma) dönüşümü, gamma<1 koyu tonları açar."""
    hi = max(np.percentile(img, p_high), 1e-6)
    x = np.clip(img.astype(np.float32) / hi, 0, 1)
    return _to_uint8(255.0 * x ** gamma)


def apply_intensity(img: np.ndarray, cfg: dict, context: dict | None = None) -> np.ndarray:
    m = cfg.get("method", "none")
    if m == "none":
        return img.copy()
    if m == "stretch":
        return contrast_stretch(img, cfg.get("p_low", 1.0), cfg.get("p_high", 99.8))
    if m == "histeq":
        return hist_equalize(img)
    if m == "clahe":
        return clahe(img, cfg.get("clip_limit", 2.0), cfg.get("tile", 8))
    if m == "match":
        if not context or "match_reference" not in context:
            raise ValueError("Histogram eşleme için context['match_reference'] gerekli")
        return hist_match(img, context["match_reference"])
    if m == "log":
        return log_transform(img, cfg.get("p_high", 99.9))
    if m == "gamma":
        return gamma_transform(img, cfg.get("gamma", 0.5), cfg.get("p_high", 99.9))
    raise ValueError(f"Bilinmeyen yoğunluk yöntemi: {m}")


# ---------------------------------------------------------------------------
# 2) Yumuşatma / keskinleştirme filtreleri
# ---------------------------------------------------------------------------
def apply_filter(img: np.ndarray, cfg: dict) -> np.ndarray:
    m = cfg.get("method", "none")
    k = int(cfg.get("ksize", 5))
    if m == "none":
        return img
    if m == "gaussian":  # sigma=0 -> OpenCV sigma'yı kernel boyutundan hesaplar
        return cv2.GaussianBlur(img, (k, k), cfg.get("sigma", 0))
    if m == "median":
        return cv2.medianBlur(img, k)
    if m == "mean":
        return cv2.blur(img, (k, k))
    if m == "bilateral":
        return cv2.bilateralFilter(img, k, cfg.get("sigma_color", 25), cfg.get("sigma_space", 5))
    if m == "laplacian_sharpen":
        # g = f - c * (Laplacian kernel * f): 3x3 konvolüsyon tabanlı keskinleştirme
        c = cfg.get("amount", 1.0)
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], np.float32)
        lap = cv2.filter2D(img.astype(np.float32), -1, kernel)
        return _to_uint8(img.astype(np.float32) + c * lap)
    if m == "unsharp":
        # g = f + a*(f - G_sigma*f): bulanık maskeleme
        a = cfg.get("amount", 1.0)
        blur = cv2.GaussianBlur(img.astype(np.float32), (k, k), 0)
        return _to_uint8(img.astype(np.float32) + a * (img.astype(np.float32) - blur))
    if m == "gaussian_then_laplacian":
        g = cv2.GaussianBlur(img, (k, k), 0)
        return apply_filter(g, {"method": "laplacian_sharpen", "amount": cfg.get("amount", 1.0)})
    if m == "gaussian_then_unsharp":
        g = cv2.GaussianBlur(img, (k, k), 0)
        return apply_filter(g, {"method": "unsharp", "ksize": cfg.get("sharp_ksize", 9),
                                "amount": cfg.get("amount", 1.0)})
    raise ValueError(f"Bilinmeyen filtre: {m}")


def apply_background(img: np.ndarray, cfg: dict) -> np.ndarray:
    """Düzensiz arka planın giderilmesi.

    tophat: beyaz top-hat = f - açma(f, B); B çekirdekten büyük elips olmalı.
    """
    m = cfg.get("method", "none")
    if m == "none":
        return img
    if m == "tophat":
        se = structuring_element(int(cfg.get("size", 81)), "ellipse")
        th = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, se)
        if cfg.get("rescale", True):
            hi = max(np.percentile(th, cfg.get("p_high", 99.9)), 1e-6)
            th = _to_uint8(th.astype(np.float32) * 255.0 / hi)
        return th
    raise ValueError(f"Bilinmeyen arka plan yöntemi: {m}")


# ---------------------------------------------------------------------------
# 3) Segmentasyon (ikili maske üretimi)
# ---------------------------------------------------------------------------
def segment(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, dict]:
    """İkili maske (bool) ve kullanılan eşik bilgisi döndürür."""
    m = cfg["method"]
    info: dict = {}
    if m == "fixed":
        t = cfg["threshold"]
        mask = img > t
        info["threshold"] = t
    elif m == "otsu":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = t * cfg.get("factor", 1.0)
        mask = img > t
        info["threshold"] = float(t)
    elif m == "triangle":
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
        mask = img > t
        info["threshold"] = float(t)
    elif m in ("adaptive_mean", "adaptive_gaussian"):
        typ = cv2.ADAPTIVE_THRESH_MEAN_C if m == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        # T(x,y) = yerel ortalama - C ; C<0 => yerel ortalamadan |C| daha parlak
        b = int(cfg.get("block_size", 151))
        out = cv2.adaptiveThreshold(img, 255, typ, cv2.THRESH_BINARY, b, cfg.get("C", -5))
        mask = out > 0
        if cfg.get("min_intensity") is not None:  # boş arka planda gürültüyü engelle
            mask &= img > cfg["min_intensity"]
        info["block_size"] = b
    elif m == "otsu_and_adaptive":
        # Global Otsu (gevşetilmiş) ile yerel eşiğin kesişimi
        t, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = t * cfg.get("factor", 1.0)
        b = int(cfg.get("block_size", 151))
        out = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, b, cfg.get("C", -5))
        mask = (img > t) & (out > 0)
        info["threshold"] = float(t)
    elif m == "kmeans":
        k = int(cfg.get("k", 3))
        z = img.reshape(-1, 1).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        cv2.setRNGSeed(0)
        _, labels, centers = cv2.kmeans(z, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        order = np.argsort(centers.ravel())
        n_bg = int(cfg.get("n_background", 1))  # en koyu n küme arka plan
        fg = set(order[n_bg:].tolist())
        mask = np.isin(labels.ravel(), list(fg)).reshape(img.shape)
        info["centers"] = sorted(float(c) for c in centers.ravel())
    else:
        raise ValueError(f"Bilinmeyen segmentasyon: {m}")
    return mask, info


# ---------------------------------------------------------------------------
# 4) Morfolojik işlemler
# ---------------------------------------------------------------------------
def structuring_element(size: int, shape: str = "ellipse") -> np.ndarray:
    shapes = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}
    return cv2.getStructuringElement(shapes[shape], (size, size))


def apply_morphology(mask: np.ndarray, ops: list[dict]) -> np.ndarray:
    m = mask.astype(np.uint8)
    codes = {"erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
             "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE}
    for op in ops:
        name = op["op"]
        if name == "fill_holes":
            m = ndi.binary_fill_holes(m).astype(np.uint8)
            continue
        se = structuring_element(int(op["size"]), op.get("shape", "ellipse"))
        m = cv2.morphologyEx(m, codes[name], se, iterations=int(op.get("iterations", 1)))
    return m.astype(bool)


# ---------------------------------------------------------------------------
# 5) Temas eden hücrelerin ayrılması ve nesne etiketleme
# ---------------------------------------------------------------------------
def label_components(mask: np.ndarray) -> np.ndarray:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    return lab


def watershed_split(mask: np.ndarray, cfg: dict) -> np.ndarray:
    """Distance transform + işaretleyici kontrollü watershed.

    İşaretleyiciler, uzaklık haritasının h-maksimumlarından (``h``) veya
    ``min_distance`` ile yerel maksimumlardan elde edilir.
    """
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if cfg.get("smooth_sigma", 0) > 0:
        dist = ndi.gaussian_filter(dist, cfg["smooth_sigma"])
    if cfg.get("marker", "hmax") == "hmax":
        peaks = h_maxima(dist, cfg.get("h", 2.0))
        markers, _ = ndi.label(peaks)
    else:
        lab0 = label_components(mask)
        coords = peak_local_max(dist, min_distance=int(cfg.get("min_distance", 15)),
                                labels=lab0, exclude_border=False)
        markers = np.zeros(mask.shape, np.int32)
        markers[tuple(coords.T)] = np.arange(1, len(coords) + 1)
    return segmentation.watershed(-dist, markers, mask=mask)


def filter_objects(labels: np.ndarray, min_area: int = 0, max_area: int | None = None,
                   clear_border: bool = False) -> np.ndarray:
    """Küçük nesneleri ve (isteğe bağlı) görüntü kenarına değen nesneleri kaldırır.

    Kenar temizleme gerekçesi: geliştirme karelerindeki TRA anotasyonlarında
    görüntü kenarına değen hiçbir hücre bulunmamaktadır (kısmen görünen hücreler
    referansa dahil edilmemiştir)."""
    ids, areas = np.unique(labels, return_counts=True)
    keep = np.zeros(int(labels.max()) + 1, bool)
    border = set()
    if clear_border:
        border = set(np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])))
    for i, a in zip(ids, areas):
        if i == 0:
            continue
        keep[i] = a >= min_area and (max_area is None or a <= max_area) and i not in border
    out = np.where(keep[labels], labels, 0)
    # ardışık yeniden numaralandırma
    _, inv = np.unique(out, return_inverse=True)
    return inv.reshape(labels.shape).astype(np.int32)
