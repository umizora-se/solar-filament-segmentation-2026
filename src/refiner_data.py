"""
crop refiner の学習データを作る。

MAGFiLO 作者グループのフレームワーク論文（Sci. Direct S0094576526004649）と同じ発想:
フルディスク画像ではなく「フィラメント1本ごとのクロップ」で学習することで
サンプル数を実効的に増やす（1画像に複数個体があるため）。

各サンプル:
  gray  : 元解像度から切り出した正方形クロップを 256x256 にリサイズ (uint8)
  mask  : 同じ領域の GT マスク (uint8 0/1)
  meta  : 切り出し位置とスケール（推論時に元解像度へ戻すため）
  label : 1=実在するフィラメント / 0=Stage1 の誤検出（棄却ヘッドの負例）

使い方:
  python src/refiner_data.py --fold 0 --split train
  python src/refiner_data.py --fold 0 --split val
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from filament_dataset import get_train_dicts, load_folds, SINGLECLASS_JSON, TRAIN_IMG_DIR  # noqa: E402

H = W = 2048
CROP = 256          # 学習に使うクロップの一辺
PAD = 0.40          # 箱の外側に取る余白の割合（上位解法の PAD=0.40 に合わせる）


def square_box(x0, y0, w, h, pad=PAD, img_h=H, img_w=W):
    """bbox を pad ぶん広げた正方形に変換し、画像内にクリップして返す。"""
    cx, cy = x0 + w / 2.0, y0 + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * pad)
    side = max(side, 32.0)                      # 極端に小さい箱の下限
    half = side / 2.0
    X0, Y0 = cx - half, cy - half
    # 画像外にはみ出す場合は中心をずらす（サイズは保つ）
    X0 = min(max(X0, 0.0), max(img_w - side, 0.0))
    Y0 = min(max(Y0, 0.0), max(img_h - side, 0.0))
    side = min(side, float(min(img_w, img_h)))
    return X0, Y0, side


def extract(gray_full: np.ndarray, box, mask_full=None, crop=CROP):
    """正方形領域を切り出して crop x crop にリサイズ。mask_full があれば同時に切る。"""
    X0, Y0, side = box
    x0, y0 = int(round(X0)), int(round(Y0))
    s = int(round(side))
    x1, y1 = min(x0 + s, gray_full.shape[1]), min(y0 + s, gray_full.shape[0])
    x0, y0 = max(x0, 0), max(y0, 0)
    g = gray_full[y0:y1, x0:x1]
    if g.size == 0:
        return None, None
    g = cv2.resize(g, (crop, crop), interpolation=cv2.INTER_AREA)
    m = None
    if mask_full is not None:
        m = mask_full[y0:y1, x0:x1]
        m = cv2.resize(m.astype(np.uint8), (crop, crop), interpolation=cv2.INTER_NEAREST)
    return g, m


def poly_to_mask(poly):
    rle = mask_util.merge(mask_util.frPyObjects([poly], H, W))
    return mask_util.decode(rle).astype(np.uint8)


def build_positives(fold: int, split: str):
    """GT インスタンスから正例クロップを作る。"""
    dicts = get_train_dicts(fold, split)
    by_stem = defaultdict(list)
    for d in dicts:
        for a in d["annotations"]:
            by_stem[d["stem"]].append(a)

    grays, masks, seeds, metas = [], [], [], []
    t0 = time.time()
    for i, (stem, annos) in enumerate(sorted(by_stem.items()), 1):
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for a in annos:
            seg = a["segmentation"][0]
            m_full = poly_to_mask(seg)
            box = square_box(*a["bbox"])
            g, m = extract(img, box, m_full)
            if g is None or m is None or m.sum() == 0:
                continue
            grays.append(g)
            masks.append(m)
            seeds.append(m)              # 正例の seed は GT マスク（学習時に劣化させる）
            metas.append({"stem": stem, "box": [float(v) for v in box], "label": 1})
        if i % 100 == 0 or i == len(by_stem):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(by_stem)}] 正例 {len(grays):6d} 個  "
                  f"({el:.0f}s, 残り約{el/i*(len(by_stem)-i)/60:.1f}分)", flush=True)
    return grays, masks, seeds, metas


def build_negatives(pred_json: str, fold: int, split: str, max_neg: int | None = None,
                    min_score: float = 0.30):
    """Stage1 の予測のうち GT と全く重ならないもの（完全FP）を負例にする。

    min_score: refiner が実際に見るのは後処理しきい値(約0.35)を超えた予測だけなので、
    それより低スコアの候補まで負例に含めると学習と推論で分布がズレる。
    """
    raw = json.loads(Path(pred_json).read_text())
    raw = raw["preds"] if "preds" in raw else raw
    fold_of = load_folds()
    want = {d["stem"] for d in get_train_dicts(fold, split)}
    stems = [s for s in raw if s in want]

    # GT を stem 単位で RLE 化
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt_rles = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if st in set(stems):
            seg = a.get("segmentation") or []
            if seg and seg[0] and len(seg[0]) >= 6:
                gt_rles[st].append(mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))

    grays, masks, seeds, metas = [], [], [], []
    for i, stem in enumerate(sorted(stems), 1):
        items = raw[stem]
        if not items:
            continue
        items = [it for it in items if it["score"] >= min_score]
        if not items:
            continue
        prles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
        g_list = gt_rles.get(stem, [])
        if g_list:
            iou = mask_util.iou(prles, g_list, [0] * len(g_list))   # (n_pred, n_gt)
            worst = np.asarray(iou).max(1)
        else:
            worst = np.zeros(len(prles))
        keep = [j for j in range(len(prles)) if worst[j] < 0.1]
        if not keep:
            continue
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        for j in keep:
            m_full = mask_util.decode(prles[j]).astype(np.uint8)
            ys, xs = np.nonzero(m_full)
            if ys.size == 0:
                continue
            box = square_box(float(xs.min()), float(ys.min()),
                             float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
            g, m = extract(img, box, m_full)
            if g is None:
                continue
            grays.append(g)
            masks.append(np.zeros_like(m))          # 負例の目標マスクは空
            seeds.append(m)                         # 負例の seed は Stage1 の予測マスクそのもの
            metas.append({"stem": stem, "box": [float(v) for v in box], "label": 0})
            if max_neg and len(grays) >= max_neg:
                return grays, masks, seeds, metas
        if i % 100 == 0:
            print(f"  [{i:4d}/{len(stems)}] 負例 {len(grays):5d} 個", flush=True)
    return grays, masks, seeds, metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["train", "val"], default="train")
    ap.add_argument("--neg-from", default=None,
                    help="負例を採取する Stage1 予測 JSON（省略すると正例のみ）")
    ap.add_argument("--max-neg", type=int, default=None)
    ap.add_argument("--neg-min-score", type=float, default=0.30,
                    help="負例に採る Stage1 予測のスコア下限（推論時の動作点に合わせる）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out = Path(args.out or REPO / "data" / f"refiner_f{args.fold}_{args.split}.npz")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[正例] fold={args.fold} split={args.split}")
    g, m, sd, meta = build_positives(args.fold, args.split)
    print(f"  正例 {len(g)} 個")

    if args.neg_from:
        print(f"[負例] {args.neg_from} から採取")
        gn, mn, sdn, mtn = build_negatives(args.neg_from, args.fold, args.split,
                                           args.max_neg, args.neg_min_score)
        print(f"  負例 {len(gn)} 個")
        g += gn; m += mn; sd += sdn; meta += mtn

    G = np.stack(g).astype(np.uint8)
    M = np.stack(m).astype(np.uint8)
    L = np.asarray([x["label"] for x in meta], np.uint8)
    S = np.stack(sd).astype(np.uint8)
    np.savez_compressed(out, gray=G, mask=M, seed=S, label=L, meta=json.dumps(meta))
    print(f"\n[write] {out}")
    print(f"  形状 gray={G.shape} mask={M.shape}  正例={int(L.sum())} 負例={int((L==0).sum())}")
    print(f"  サイズ {out.stat().st_size/1e6:.0f} MB")


if __name__ == "__main__":
    main()
