"""
MetaSeg 方式の予測品質推定（Rottmann et al., arXiv 1811.00648）。

Stage1 の各予測インスタンスについてセグメント単位の特徴量を作り、
「その予測が GT とどれだけ重なるか（IoU）」を勾配ブースティングで回帰する。
推定 IoU を新しいスコアとして後段の後処理に渡す。

crop refiner と違い **マスクを一切変更しない**（スコアの付け直しのみ）ため、
再セグメンテーションで TP を失う経路がない。

我々の誤差分解では「完全FP(IoU<0.1) 324本」が最大の伸びしろ(+0.065)であり、
かつ fill/aspect の手動しきい値では分離できないと実証済み。
論文の主張どおり、**学習による特徴量の集約**が必要な場面。

使い方:
  # 特徴量抽出（CPUのみ）
  python src/meta_classifier.py extract --preds work_dirs/r50_f0/val_raw_tta.json \
      --out work_dirs/r50_f0/meta_val.npz
  # 学習と評価（val を stem で2分割）
  python src/meta_classifier.py fit --feats work_dirs/r50_f0/meta_val.npz \
      --model work_dirs/r50_f0/meta_model.pkl
  # 適用（スコアを推定IoUに置換した予測 JSON を出力）
  python src/meta_classifier.py apply --preds ... --feats ... --model ... --out ...
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from filament_dataset import TRAIN_IMG_DIR, TEST_IMG_DIR, SINGLECLASS_JSON  # noqa: E402

H = W = 2048
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

FEATURE_NAMES = [
    "score", "area", "perimeter", "fractality", "bbox_w", "bbox_h", "aspect", "fill",
    "elongation", "solidity", "r_over_R", "n_pred_img", "max_iou_other",
    "in_mean", "in_std", "in_p10", "ring_mean", "contrast",
    "cl_in_mean", "cl_contrast", "log_area",
]


def _disk(gray: np.ndarray):
    """明るい画素から太陽円板の中心と半径を推定する。"""
    ys, xs = np.nonzero(gray > 18)
    if xs.size < 500:
        return np.array([W / 2, H / 2]), 0.45 * H
    v = gray[ys, xs]
    keep = v >= np.median(v)
    xs, ys = xs[keep], ys[keep]
    cx, cy = float(xs.mean()), float(ys.mean())
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return np.array([cx, cy]), float(np.percentile(r, 96.0))


def image_features(gray: np.ndarray, cl: np.ndarray, rles: list, scores: np.ndarray):
    """1画像ぶんの予測について特徴量行列 (n, len(FEATURE_NAMES)) を返す。"""
    n = len(rles)
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), np.float32)
    center, R = _disk(gray)

    # 予測どうしの最大 IoU（混雑度）
    if n > 1:
        iou = np.asarray(mask_util.iou(rles, rles, [0] * n))
        np.fill_diagonal(iou, 0.0)
        max_iou_other = iou.max(1)
    else:
        max_iou_other = np.zeros(1)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    out = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    for i, r in enumerate(rles):
        m = mask_util.decode(r).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            continue
        area = float(ys.size)
        bw = float(xs.max() - xs.min() + 1)
        bh = float(ys.max() - ys.min() + 1)
        per = float(cv2.morphologyEx(m, cv2.MORPH_GRADIENT, k3).sum())

        # 主軸の伸び（画素座標のPCA）
        p = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
        ev = np.linalg.eigvalsh(np.cov(p) + np.eye(2) * 1e-6)
        elong = float(np.sqrt(max(ev[1], 1e-6) / max(ev[0], 1e-6)))

        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull_a = sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts) if cnts else area
        solidity = float(area / max(hull_a, 1.0))

        cx, cy = float(xs.mean()), float(ys.mean())
        r_over_R = float(np.hypot(cx - center[0], cy - center[1]) / max(R, 1.0))

        ring = (cv2.dilate(m, k15) - m).astype(bool)
        mb = m.astype(bool)
        in_v = gray[mb].astype(np.float32)
        ring_v = gray[ring].astype(np.float32) if ring.any() else in_v
        cl_in = cl[mb].astype(np.float32)
        cl_ring = cl[ring].astype(np.float32) if ring.any() else cl_in

        out[i] = [
            scores[i], area, per, area / max(per, 1.0), bw, bh,
            max(bw, bh) / max(min(bw, bh), 1.0), area / max(bw * bh, 1.0),
            elong, solidity, r_over_R, float(n), float(max_iou_other[i]),
            in_v.mean(), in_v.std(), float(np.percentile(in_v, 10)),
            ring_v.mean(), ring_v.mean() - in_v.mean(),
            cl_in.mean(), cl_ring.mean() - cl_in.mean(), float(np.log1p(area)),
        ]
    return out


def find_image(stem: str) -> Path | None:
    for d in (TRAIN_IMG_DIR, TEST_IMG_DIR):
        p = Path(d) / f"{stem}.jpeg"
        if p.exists():
            return p
    return None


def load_gt_rles(stems: set):
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    out = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if st not in stems:
            continue
        seg = a.get("segmentation") or []
        if seg and seg[0] and len(seg[0]) >= 6:
            out[st].append(mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))
    return out


def cmd_extract(args):
    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    stems = sorted(preds)
    gt = load_gt_rles(set(stems)) if not args.no_labels else {}

    X, Y, S, IDX = [], [], [], []
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        items = preds[stem]
        p = find_image(stem)
        if p is None or not items:
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        cl = _CLAHE.apply(gray)
        rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
        sc = np.asarray([it["score"] for it in items], np.float32)
        F = image_features(gray, cl, rles, sc)
        X.append(F)
        S += [stem] * len(items)
        IDX += list(range(len(items)))
        if gt.get(stem):
            iou = np.asarray(mask_util.iou(rles, gt[stem], [0] * len(gt[stem])))
            Y.append(iou.max(1))                    # 全GTに対する最大IoU
        else:
            Y.append(np.zeros(len(items)))
        if i % 20 == 0 or i == len(stems):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(stems)}] {sum(len(x) for x in X):6d} 予測  "
                  f"({el:.0f}s, 残り約{el/i*(len(stems)-i)/60:.1f}分)", flush=True)

    X = np.concatenate(X).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    np.savez_compressed(args.out, X=X, Y=Y, stem=np.array(S), idx=np.array(IDX),
                        names=np.array(FEATURE_NAMES))
    print(f"\n[write] {args.out}  X={X.shape} Y={Y.shape}")
    print(f"  ラベル分布: IoU>=0.5 {int((Y>=0.5).sum())} / "
          f"0.1-0.5 {int(((Y>=0.1)&(Y<0.5)).sum())} / <0.1 {int((Y<0.1).sum())}")


def cmd_fit(args):
    d = np.load(args.feats, allow_pickle=True)
    X, Y, stem = d["X"], d["Y"], d["stem"]
    names = list(d["names"])

    # stem で2分割（同じ写真が両側に入らないように）
    uniq = sorted(set(stem.tolist()))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2])
    tr = np.array([s in half for s in stem])
    va = ~tr
    print(f"[split] 写真 {len(uniq)} -> 学習 {len(half)} / 評価 {len(uniq)-len(half)}")
    print(f"        予測 学習 {int(tr.sum())} / 評価 {int(va.sum())}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0)
    model.fit(X[tr], Y[tr])

    pred = np.clip(model.predict(X[va]), 0, 1)
    true = Y[va]
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    print(f"\n[meta regression] R^2 = {1 - ss_res/ss_tot:.4f}")
    for thr, lbl in [(0.5, "TP判定 (IoU>=0.5)"), (0.1, "実在判定 (IoU>=0.1)")]:
        y = (true >= thr).astype(int)
        if 0 < y.sum() < len(y):
            auc_meta = roc_auc_score(y, pred)
            auc_raw = roc_auc_score(y, X[va][:, 0])       # Stage1 の生スコア
            print(f"[meta classification] {lbl}: AUROC メタ {auc_meta:.4f} / "
                  f"Stage1スコア {auc_raw:.4f}  （差 {auc_meta-auc_raw:+.4f}）")

    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(model, X[va], true, n_repeats=5, random_state=0, n_jobs=4)
        imp = r.importances_mean
    order = np.argsort(-imp)[:10]
    print("\n[特徴量の寄与 上位10]")
    for j in order:
        print(f"  {names[j]:14s} {imp[j]:.4f}")

    # 本番用に全データで再学習
    final = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        min_samples_leaf=20, l2_regularization=1.0, random_state=0).fit(X, Y)
    Path(args.model).write_bytes(pickle.dumps({"model": final, "names": names,
                                               "holdout_stems": sorted(set(uniq) - half)}))
    print(f"\n[write] {args.model}")


def cmd_apply(args):
    d = np.load(args.feats, allow_pickle=True)
    X, stem, idx = d["X"], d["stem"], d["idx"]
    mdl = pickle.loads(Path(args.model).read_bytes())["model"]
    p = np.clip(mdl.predict(X), 0, 1)

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    out = {s: [dict(it) for it in v] for s, v in preds.items()}
    for s, i, v in zip(stem.tolist(), idx.tolist(), p.tolist()):
        out[s][i]["score"] = float(v)
    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"[write] {args.out}  {len(p)} 予測のスコアを推定IoUに置換")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--preds", required=True)
    e.add_argument("--out", required=True); e.add_argument("--no-labels", action="store_true")
    e.set_defaults(func=cmd_extract)
    f = sub.add_parser("fit"); f.add_argument("--feats", required=True)
    f.add_argument("--model", required=True); f.set_defaults(func=cmd_fit)
    a = sub.add_parser("apply"); a.add_argument("--preds", required=True)
    a.add_argument("--feats", required=True); a.add_argument("--model", required=True)
    a.add_argument("--out", required=True); a.set_defaults(func=cmd_apply)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
