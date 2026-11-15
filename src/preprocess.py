"""
Hα 全面画像の前処理（Reche & Cid 2026, Acta Astronautica 249 の手順）。

1. 太陽円板のフィット（中心・半径）
2. リム暗化補正 — 動径ビン dr=0.01R で中央値強度プロファイルを取り、
   スプライン当てはめ I_fit で割る
3. ヒストグラムマッチング（円板内部のみ）— 三峰参照分布に合わせる
      D_ref(x) = 0.01·N(x;0.35,0.01²) + N(x;0.6,0.1²) + 0.01·N(x;0.85,0.01²)
   中央ピークを右寄せ(μ=0.60)することで暗い大気障害（雲）の影響を減衰させる
4. 円板外はゼロ埋め

狙い: 我々の誤差分解で最大の伸びしろだった「完全FP 324本」（本物より小さく丸い＝
黒点や雲を拾っている疑い）を、入力段階で減らす。

    python src/preprocess.py --split train --out datasets/preprocessed/train
    python src/preprocess.py --split test  --out datasets/preprocessed/test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from filament_dataset import TRAIN_IMG_DIR, TEST_IMG_DIR  # noqa: E402

H = W = 2048

# 参照分布のパラメータ（論文 §前処理）
_REF = [(0.01, 0.35, 0.01), (1.00, 0.60, 0.10), (0.01, 0.85, 0.01)]
DATASET_MEAN, DATASET_STD = 0.60, 0.10


def fit_disk(gray: np.ndarray, bright_min: int = 18):
    """明るい画素から太陽円板の中心と半径を推定する。"""
    ys, xs = np.nonzero(gray > bright_min)
    if xs.size < 500:
        return np.array([W / 2.0, H / 2.0]), 0.45 * H
    v = gray[ys, xs]
    keep = v >= np.median(v)
    xs, ys = xs[keep], ys[keep]
    cx, cy = float(xs.mean()), float(ys.mean())
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return np.array([cx, cy]), float(np.percentile(r, 96.0))


def limb_correction(gray: np.ndarray, center, R, dr: float = 0.01):
    """動径方向の中央値プロファイルで割り、リム暗化を平坦化する。"""
    yy, xx = np.mgrid[0:H, 0:W]
    rad = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / max(R, 1.0)
    disk = rad <= 1.0
    g = gray.astype(np.float32)

    edges = np.arange(0, 1.0 + dr, dr)
    centers, med = [], []
    bin_idx = np.clip((rad / dr).astype(np.int32), 0, len(edges) - 2)
    for b in range(len(edges) - 1):
        sel = disk & (bin_idx == b)
        if sel.sum() >= 50:
            centers.append((edges[b] + edges[b + 1]) / 2.0)
            med.append(float(np.median(g[sel])))
    if len(centers) < 5:
        return g, disk

    centers = np.asarray(centers); med = np.asarray(med)
    # 平滑化したプロファイルを作る（スプライン相当。SciPy 依存を避けて多項式+移動平均）
    k = min(9, len(med) // 2 * 2 + 1)
    if k >= 3:
        med_s = np.convolve(med, np.ones(k) / k, mode="same")
        med_s[: k // 2] = med[: k // 2]
        med_s[-(k // 2):] = med[-(k // 2):] if k // 2 else med_s[-1:]
    else:
        med_s = med
    prof = np.interp(rad.ravel(), centers, med_s).reshape(H, W)
    prof = np.maximum(prof, 1.0)
    out = np.where(disk, g / prof, 0.0).astype(np.float32)
    return out, disk


def _ref_cdf(n: int = 4096):
    x = np.linspace(0.0, 1.0, n)
    pdf = np.zeros(n)
    for w, mu, sd in _REF:
        pdf += w * np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    cdf = np.cumsum(pdf)
    return x, cdf / cdf[-1]


_REF_X, _REF_CDF = _ref_cdf()


def histogram_match(img: np.ndarray, disk: np.ndarray):
    """円板内部の分布を三峰参照分布に合わせる。"""
    v = img[disk]
    if v.size == 0:
        return img
    order = np.argsort(v)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(v.size)
    src_cdf = (ranks + 0.5) / v.size                    # 経験CDF
    matched = np.interp(src_cdf, _REF_CDF, _REF_X)      # 参照CDFの逆関数
    out = np.zeros_like(img, dtype=np.float32)
    out[disk] = matched.astype(np.float32)
    return out


def preprocess(gray: np.ndarray):
    """8bit グレースケール -> 前処理済み 8bit（円板外=0）。"""
    center, R = fit_disk(gray)
    corr, disk = limb_correction(gray, center, R)
    matched = histogram_match(corr, disk)
    # μ=0.6, σ=0.1 を基準に 0-1 へ収め、8bit に量子化
    z = (matched - DATASET_MEAN) / DATASET_STD
    out = np.clip(z * 0.18 + 0.5, 0.0, 1.0)
    out[~disk] = 0.0
    return (out * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    src = TRAIN_IMG_DIR if args.split == "train" else TEST_IMG_DIR
    out = Path(args.out or REPO / "datasets" / "preprocessed" / args.split)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(src).glob("*.jpeg"))
    if args.limit:
        files = files[: args.limit]
    print(f"[preprocess] {args.split}: {len(files)} 枚 -> {out}")

    t0 = time.time()
    for i, p in enumerate(files, 1):
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        cv2.imwrite(str(out / p.name), preprocess(g), [cv2.IMWRITE_JPEG_QUALITY, 95])
        if i % 50 == 0 or i == len(files):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(files)}] {el:.0f}s "
                  f"(残り約{el/i*(len(files)-i)/60:.1f}分)", flush=True)
    print(f"[done] {out}")


if __name__ == "__main__":
    main()
