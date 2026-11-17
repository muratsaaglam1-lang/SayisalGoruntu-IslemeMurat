"""Seed tabanlı, tekrarlanabilir geliştirme / doğrulama / test kare seçimi.

Kurallar (ödev Bölüm 3):
* Sequence 01'den 5 doğrulama + 10 geliştirme karesi, çakışmasız ve iki grup
  arasında en az 5 karelik zamansal boşluk (|t_dev - t_val| >= MIN_GAP + 1).
* Sequence 02'den rastgele 3 başlangıç noktası, her birinden ardışık 5 kare;
  bloklar çakışmaz ve aralarında en az 5 kare boşluk bulunur.
* Koşul sağlanamazsa aynı ``numpy.random.Generator`` ile (aynı seed) yeniden
  çekim yapılır: deterministik tekrar prosedürü. Deneme sayısı kaydedilir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config, data


def _select_dev_val(frames: list[int], rng: np.random.Generator):
    frames_arr = np.array(frames)
    for attempt in range(1, config.MAX_SELECTION_ATTEMPTS + 1):
        val = np.sort(rng.choice(frames_arr, size=config.N_VAL, replace=False))
        # Doğrulama karelerine en az (MIN_GAP+1) uzaklıktaki kareler aday olur
        dist = np.min(np.abs(frames_arr[:, None] - val[None, :]), axis=1)
        candidates = frames_arr[dist >= config.MIN_GAP + 1]
        if len(candidates) < config.N_DEV:
            continue  # deterministik tekrar
        dev = np.sort(rng.choice(candidates, size=config.N_DEV, replace=False))
        return dev.tolist(), val.tolist(), attempt
    raise RuntimeError("Geliştirme/doğrulama seçimi koşulları sağlanamadı.")


def _select_test_blocks(frames: list[int], rng: np.random.Generator):
    first, last = min(frames), max(frames)
    max_start = last - config.TEST_BLOCK_LEN + 1
    for attempt in range(1, config.MAX_SELECTION_ATTEMPTS + 1):
        starts = np.sort(rng.choice(np.arange(first, max_start + 1),
                                    size=config.N_TEST_BLOCKS, replace=False))
        # blok i: [s, s+4]; sonraki blok >= s + 5 + MIN_GAP olmalı
        ok = all(starts[i + 1] - (starts[i] + config.TEST_BLOCK_LEN - 1) - 1 >= config.MIN_GAP
                 for i in range(len(starts) - 1))
        if ok:
            blocks = [list(range(int(s), int(s) + config.TEST_BLOCK_LEN)) for s in starts]
            return blocks, attempt
    raise RuntimeError("Test bloğu seçimi koşulları sağlanamadı.")


def select_frames(seed: int = config.SEED) -> pd.DataFrame:
    """Seçimi yapar ve tablo olarak döndürür (CSV'ye yazmaz)."""
    data.check_dataset()
    rng = np.random.default_rng(seed)
    dev, val, att_dv = _select_dev_val(data.list_frames(config.DEV_SEQ), rng)
    blocks, att_t = _select_test_blocks(data.list_frames(config.TEST_SEQ), rng)

    rows = []
    for t in val:
        rows.append(("validation", config.DEV_SEQ, t, "", att_dv))
    for t in dev:
        rows.append(("development", config.DEV_SEQ, t, "", att_dv))
    for b, block in enumerate(blocks, start=1):
        for t in block:
            rows.append(("test", config.TEST_SEQ, t, f"B{b}", att_t))
    df = pd.DataFrame(rows, columns=["group", "sequence", "frame", "test_block", "attempts"])
    df["file_name"] = [f"{r.sequence}/t{r.frame:03d}.tif" for r in df.itertuples()]
    df["tra_file"] = [f"{r.sequence}_GT/TRA/man_track{r.frame:03d}.tif" for r in df.itertuples()]
    df["seed"] = seed
    order = {"development": 0, "validation": 1, "test": 2}
    df = df.sort_values(["group", "frame"],
                        key=lambda s: s.map(order) if s.name == "group" else s)
    df = df.reset_index(drop=True)
    df.insert(0, "image_id", _ids(df))
    return df


def _ids(df: pd.DataFrame) -> list[str]:
    prefix = {"development": "DEV", "validation": "VAL", "test": "TST"}
    counters: dict[str, int] = {}
    out = []
    for g in df["group"]:
        counters[g] = counters.get(g, 0) + 1
        out.append(f"{prefix[g]}{counters[g]:02d}")
    return out


def verify_selection(df: pd.DataFrame) -> None:
    """Seçim kurallarını doğrular; ihlal varsa AssertionError fırlatır."""
    dev = df.loc[df.group == "development", "frame"].to_numpy()
    val = df.loc[df.group == "validation", "frame"].to_numpy()
    assert len(dev) == config.N_DEV and len(val) == config.N_VAL
    assert not set(dev) & set(val)
    assert np.min(np.abs(dev[:, None] - val[None, :])) >= config.MIN_GAP + 1
    tst = df[df.group == "test"]
    blocks = [g.frame.to_numpy() for _, g in tst.groupby("test_block")]
    assert len(blocks) == config.N_TEST_BLOCKS
    for b in blocks:
        assert len(b) == config.TEST_BLOCK_LEN and np.all(np.diff(b) == 1)
    starts = sorted(blocks, key=lambda b: b[0])
    for a, b in zip(starts, starts[1:]):
        assert b[0] - a[-1] - 1 >= config.MIN_GAP


def run_selection(seed: int = config.SEED, overwrite: bool = False) -> pd.DataFrame:
    """Seçimi yapar, doğrular ve ``selected_frames.csv`` dosyasına yazar.

    Dosya zaten varsa yeniden seçim yapmak yerine mevcut dosyayla
    karşılaştırır: seçim yalnızca bir kez yapılır, sonuçlara göre değiştirilmez.
    """
    df = select_frames(seed)
    verify_selection(df)
    if config.SELECTED_FRAMES_CSV.exists() and not overwrite:
        old = pd.read_csv(config.SELECTED_FRAMES_CSV, dtype={"sequence": str}, keep_default_na=False)
        if not (old["frame"].tolist() == df["frame"].tolist()
                and old["group"].tolist() == df["group"].tolist()):
            raise RuntimeError("Mevcut selected_frames.csv yeniden üretilen seçimle uyuşmuyor!")
        return old
    df.to_csv(config.SELECTED_FRAMES_CSV, index=False)
    return df


def load_selection() -> pd.DataFrame:
    return pd.read_csv(config.SELECTED_FRAMES_CSV, dtype={"sequence": str}, keep_default_na=False)
