"""
val の生予測から後処理パラメータ（SCORE_THR × MIN_AREA）を公式PQで探索する。

入力: predict_submit.py --dump-raw が出した JSON、または
      学習中の FilamentPQEvaluator が出した val_predictions_f{fold}.json
出力: postproc.json（最良の score_thr / min_area と、その val-PQ）

実装メモ:
  マスクは 2048x2048 だが実体は面積中央値 1,228px しかない。そこで
  「立っている画素の平坦インデックス配列」というスパース表現で保持し、
  デコードは画像ごとに1回だけ行ってグリッドを内側で回す。
  （素朴に RLE decode/encode をグリッドごとに回すと 12通りで2分以上かかった）

例:
  python src/tune_postproc.py --preds work_dirs/r50_f0/val_raw.json --fold 0 \
      --out work_dirs/r50_f0/postproc.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from filament_dataset import load_folds, SINGLECLASS_JSON  # noqa: E402

H = W = 2048
NPIX = H * W


# --------------------------------------------------------------------------- #
# スパース表現（立っている画素の平坦インデックス、昇順）
# --------------------------------------------------------------------------- #
def rle_to_idx(rle) -> np.ndarray:
    return np.flatnonzero(mask_util.decode(rle).ravel()).astype(np.int32)


def poly_to_idx(poly) -> np.ndarray:
    rle = mask_util.merge(mask_util.frPyObjects([poly], H, W))
    return np.flatnonzero(mask_util.decode(rle).ravel()).astype(np.int32)


def iou_sparse(preds: list, gts: list) -> np.ndarray:
    """(n_gt, n_pred) の IoU 行列。公式スコアラと同じ向き（行=GT, 列=pred）。"""
    out = np.zeros((len(gts), len(preds)), np.float64)
    if not preds or not gts:
        return out
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            inter = np.intersect1d(g, p, assume_unique=True).size
            if inter:
                out[i, j] = inter / (g.size + p.size - inter)
    return out


def paint_disjoint_sparse(idxs: list, scores: np.ndarray,
                          score_thr: float, min_area: int,
                          used: np.ndarray, min_keep_frac: float = 0.0) -> list:
    """スコア降順に画素排他ペイント。used は使い回す bool ワークバッファ。
    min_keep_frac: 排他後に元マスクのこの割合未満しか残らない断片は棄却（0 で従来と同一）。"""
    order = np.argsort(-scores)
    touched, kept = [], []
    for i in order:
        if scores[i] < score_thr:
            continue
        idx = idxs[i]
        free = idx[~used[idx]]           # 既に塗られた画素は奪わない
        if free.size < min_area or free.size < min_keep_frac * idx.size:
            continue
        used[free] = True
        touched.append(free)
        kept.append(free)
    for t in touched:                     # バッファを元に戻す
        used[t] = False
    return kept


# --------------------------------------------------------------------------- #
def load_gt_idx(stems: set) -> dict:
    """stem -> [アノテーターごとの [スパースマスク, ...]]"""
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    by_rec = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if st not in stems:
            continue
        seg = a.get("segmentation") or []
        if seg and seg[0] and len(seg[0]) >= 6:
            by_rec[a["image_id"]].append(seg[0])
    out = defaultdict(list)
    for rec, polys in by_rec.items():
        out[id2stem[rec]].append([poly_to_idx(p) for p in polys])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--score-thr", default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    ap.add_argument("--min-area", default="0,100,150,200,300,500")
    ap.add_argument("--min-keep-frac", type=float, default=0.0, help="retain-fraction rule (0 = off)")
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds_raw = raw["preds"] if "preds" in raw else raw

    fold_of = load_folds()
    if args.fold < 0:
        stems = set(preds_raw)                 # -1: fold で絞らない（OOF統合など）
        print(f"[info] 全fold対象: {len(stems)} 画像")
    else:
        stems = {s for s in preds_raw if fold_of.get(s) == args.fold}
        if len(stems) != len(preds_raw):
            print(f"[warn] fold {args.fold} の val 以外を除外: {len(preds_raw)} -> {len(stems)}")
    print(f"[load] {args.preds}: {len(stems)} 画像  "
          f"平均予測数={np.mean([len(preds_raw[s]) for s in stems]):.1f}")

    t0 = time.time()
    gt = load_gt_idx(stems)
    print(f"[gt] {len(gt)} 画像  レコード数={sum(len(v) for v in gt.values())}  "
          f"GT個体数={sum(len(r) for v in gt.values() for r in v)}  ({time.time()-t0:.0f}s)")

    score_grid = [float(x) for x in args.score_thr.split(",")]
    area_grid = [int(x) for x in args.min_area.split(",")]
    cells = [(st, ma) for st in score_grid for ma in area_grid]
    stats = {c: dict(sum_iou=0.0, tp=0, fp=0, fn=0, n_pred=0) for c in cells}

    used = np.zeros(NPIX, bool)            # 使い回すワークバッファ
    t0 = time.time()
    for n, stem in enumerate(sorted(stems), 1):
        items = preds_raw[stem]
        idxs = [rle_to_idx({"size": [H, W], "counts": it["rle"].encode()}) for it in items]
        scores = np.asarray([it["score"] for it in items], np.float32)
        recs = gt.get(stem, [[]])
        for c in cells:
            kept = paint_disjoint_sparse(idxs, scores, c[0], c[1], used, args.min_keep_frac)
            s = stats[c]
            s["n_pred"] += len(kept)
            for gts in recs:
                if not gts:
                    s["fp"] += len(kept)
                    continue
                if not kept:
                    s["fn"] += len(gts)
                    continue
                iou = iou_sparse(kept, gts)          # (n_gt, n_pred)
                hit = iou > 0.5
                s["sum_iou"] += float(iou[hit].sum())
                s["tp"] += int(hit.sum())
                s["fp"] += int((hit.sum(0) == 0).sum())
                s["fn"] += int((hit.sum(1) == 0).sum())
        if n % 25 == 0 or n == len(stems):
            el = time.time() - t0
            print(f"  [{n:3d}/{len(stems)}] {el:.0f}s  (残り約{el/n*(len(stems)-n):.0f}s)")

    print(f"\n{'score_thr':>9} {'min_area':>8} {'PQ':>7} {'SQ':>6} {'RQ':>6} "
          f"{'TP':>6} {'FP':>6} {'FN':>6} {'n_pred':>7}")
    print("-" * 74)
    results, best = [], None
    for c in cells:
        s = stats[c]
        den = s["tp"] + 0.5 * s["fp"] + 0.5 * s["fn"]
        r = {
            "score_thr": c[0], "min_area": c[1],
            "PQ": s["sum_iou"] / den if den else 0.0,
            "SQ": s["sum_iou"] / s["tp"] if s["tp"] else 0.0,
            "RQ": s["tp"] / den if den else 0.0,
            "TP": s["tp"], "FP": s["fp"], "FN": s["fn"],
            "mean_n_pred": s["n_pred"] / max(len(stems), 1),
        }
        results.append(r)
        mark = ""
        if best is None or r["PQ"] > best["PQ"]:
            best = r
            mark = "  <-- best"
        print(f"{r['score_thr']:9.2f} {r['min_area']:8d} {r['PQ']:7.4f} {r['SQ']:6.3f} "
              f"{r['RQ']:6.3f} {r['TP']:6d} {r['FP']:6d} {r['FN']:6d} "
              f"{r['mean_n_pred']:7.1f}{mark}")

    print("\n=== 最良 ===")
    print(f"  score_thr={best['score_thr']}  min_area={best['min_area']}  "
          f"val-PQ={best['PQ']:.4f}  (SQ={best['SQ']:.3f} RQ={best['RQ']:.3f} "
          f"平均予測数={best['mean_n_pred']:.1f})")
    print("  参考: 古典ベースライン LB 0.46 / 人間の一致度(IAA) 0.340（天井ではない）")

    out = args.out or str(Path(args.preds).with_name("postproc.json"))
    Path(out).write_text(json.dumps({
        "score_thr": best["score_thr"], "min_area": best["min_area"], "mask_thr": 0.5, "min_keep_frac": args.min_keep_frac,
        "val_pq": best["PQ"], "val_sq": best["SQ"], "val_rq": best["RQ"],
        "mean_n_pred": best["mean_n_pred"], "fold": args.fold, "grid": results,
    }, indent=2))
    print(f"\n[write] {out}")


if __name__ == "__main__":
    main()
