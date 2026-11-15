"""学習ベースの再ランキング（施策2）。

現在は score' = score × √p という手作りの式。これを特徴量から学習する。

以前 MetaSeg 方式が失敗したのは MaskDINO 自身の出力しか特徴に使えなかったため
（スコアは既にその情報に対して較正済みだった）。今は独立情報源である
セマンティック検証器の確率が手元にあるので前提が変わっている。

特徴量（すべて予測1本ごと）:
  score, semprob(平均), semprob(中央値), semprob(p25), 面積log, 充填率,
  アスペクト比, 周囲semprob（マスクを膨らませた輪郭部の確率＝境界の妥当性）
学習: ロジスティック回帰（fold内をさらに2分割して過学習を検査）
出力: score' = P(本物)  として同じJSON形式

    python src/rerank.py --preds val_raw_tta.json --probs probs_val_ens.npz \
        --fit --model work_dirs/rerank.json --out val_raw_tta_rr.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, load_folds  # noqa: E402

H = W = 2048
FEATS = ["score", "sem_mean", "sem_med", "sem_p25", "log_area", "fill",
         "aspect", "sem_ring"]


def feats_for(m: np.ndarray, pm: np.ndarray, score: float) -> np.ndarray:
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return np.zeros(len(FEATS), np.float32)
    v = pm[m]
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    bw, bh = float(x1 - x0), float(y1 - y0)
    # 輪郭リング（膨張差分）の確率 = 境界の外側がどれだけフィラメントらしいか
    sub = m[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8].astype(np.uint8)
    ring = cv2.dilate(sub, np.ones((9, 9), np.uint8)) - sub
    psub = pm[max(y0 - 8, 0):y1 + 8, max(x0 - 8, 0):x1 + 8]
    sr = float(psub[ring > 0].mean()) if (ring > 0).any() else 0.0
    return np.array([score, float(v.mean()), float(np.median(v)),
                     float(np.percentile(v, 25)), float(np.log1p(xs.size)),
                     xs.size / max(bw * bh, 1), max(bw, bh) / max(min(bw, bh), 1), sr],
                    np.float32)


def build_xy(preds, probs, gt, want_y=True):
    X, Y, meta = [], [], []
    for stem, items in preds.items():
        if stem not in probs.files:
            continue
        pm = probs[stem].astype(np.float32) / 255.0
        rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
        if not rles:
            continue
        g = gt.get(stem, [])
        best = (mask_util.iou(rles, g, [0] * len(g)).max(1) if g
                else np.zeros(len(rles)))
        for it, r, b in zip(items, rles, best):
            m = mask_util.decode(r).astype(bool)
            X.append(feats_for(m, pm, it["score"]))
            Y.append(1.0 if b > 0.5 else 0.0)
            meta.append((stem, it))
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), meta


def load_gt(fold):
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    fold_of = load_folds()
    gt = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if fold_of.get(st) != fold:
            continue
        seg = a.get("segmentation") or []
        if seg and len(seg[0]) >= 6:
            gt[st].append(mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))
    return gt


def fit_logreg(X, Y, iters=600, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        gw = Z.T @ (p - Y) / len(Y) + 1e-4 * w
        gb = float((p - Y).mean())
        w -= lr * gw; b -= lr * gb
    return {"w": w.tolist(), "b": b, "mu": mu.tolist(), "sd": sd.tolist()}


def apply_model(mdl, X):
    Z = (X - np.asarray(mdl["mu"])) / np.asarray(mdl["sd"])
    return 1 / (1 + np.exp(-(Z @ np.asarray(mdl["w"]) + mdl["b"])))


def auroc(p, y):
    pos, neg = p[y > 0.5], p[y < 0.5]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); r = np.empty(len(a)); o = np.argsort(a); sa = a[o]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--cross-fit", action="store_true",
                    help="val を画像単位で2分割し、互いに学習→予測（リーク無しのPQ評価用）")
    ap.add_argument("--model", default="work_dirs/rerank.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    probs = np.load(args.probs)

    if args.fit:
        gt = load_gt(args.fold)
        X, Y, meta = build_xy(preds, probs, gt)
        print(f"[fit] {len(X)} 本  正例率={Y.mean():.3f}")
        # 画像単位で半分に割り、過学習を検査する
        stems = sorted({m[0] for m in meta})
        half = set(stems[::2])
        tr = np.array([m[0] in half for m in meta])
        m1 = fit_logreg(X[tr], Y[tr])
        a_in = auroc(apply_model(m1, X[tr]), Y[tr])
        a_out = auroc(apply_model(m1, X[~tr]), Y[~tr])
        a_score = auroc(X[~tr][:, 0], Y[~tr])
        a_sqrt = auroc(X[~tr][:, 0] * np.sqrt(X[~tr][:, 1]), Y[~tr])
        print(f"  AUROC  学習側={a_in:.4f}  検証側={a_out:.4f}")
        print(f"  比較   score単体={a_score:.4f}  score×√p={a_sqrt:.4f}")
        if a_out < a_sqrt:
            print("  ★ 手作りの式に負けている（採用すべきでない）")
        mdl = fit_logreg(X, Y)          # 最終モデルは全データで
        mdl["feats"] = FEATS
        mdl["auroc_holdout"] = float(a_out)
        mdl["auroc_sqrt"] = float(a_sqrt)
        Path(args.model).write_text(json.dumps(mdl, indent=2))
        print(f"[write] {args.model}")
        for n, w in sorted(zip(FEATS, mdl["w"]), key=lambda t: -abs(t[1])):
            print(f"    {n:10s} {w:+.3f}")
    if args.cross_fit:
        # 画像単位に2分割し、片方で学習して他方を予測する（自分自身では学習しない）
        gt = load_gt(args.fold)
        X, Y, meta = build_xy(preds, probs, gt)
        stems = sorted({m[0] for m in meta})
        half = set(stems[::2])
        sel = np.array([m[0] in half for m in meta])
        p_out = np.zeros(len(X))
        for tr_mask in (sel, ~sel):
            mdl = fit_logreg(X[tr_mask], Y[tr_mask])
            p_out[~tr_mask] = apply_model(mdl, X[~tr_mask])
        print(f"[cross-fit] AUROC={auroc(p_out, Y):.4f}  "
              f"(score単体 {auroc(X[:,0], Y):.4f} / score×√p "
              f"{auroc(X[:,0]*np.sqrt(X[:,1]), Y):.4f})")
        out = defaultdict(list)
        for (stem, it), s_ in zip(meta, p_out):
            out[stem].append({"rle": it["rle"], "score": float(s_)})
        for stem in preds:
            out.setdefault(stem, [])
        Path(args.out).write_text(json.dumps({"preds": dict(out)}))
        print(f"[write] {args.out}")
        return

    if args.out:
        mdl = json.loads(Path(args.model).read_text())
        X, _, meta = build_xy(preds, probs, {}, want_y=False)
        p = apply_model(mdl, X)
        out = defaultdict(list)
        for (stem, it), s in zip(meta, p):
            out[stem].append({"rle": it["rle"], "score": float(s)})
        for stem in preds:
            out.setdefault(stem, [])
        Path(args.out).write_text(json.dumps({"preds": dict(out)}))
        print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
