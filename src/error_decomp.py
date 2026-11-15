"""OOF予測の誤差分解。どこに伸びしろがあるかを 707枚の統計で測り直す。

142枚（fold0）では標準偏差 0.0088 と揺れが大きく、伸びしろの見積もりも不安定だった。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, load_folds  # noqa: E402

H = W = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--postproc", required=True)
    args = ap.parse_args()

    pp = json.loads(Path(args.postproc).read_text())
    thr, min_area = pp["score_thr"], pp["min_area"]
    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw

    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt_by_rec = defaultdict(list)
    for a in coco["annotations"]:
        seg = a.get("segmentation") or []
        if seg and len(seg[0]) >= 6:
            gt_by_rec[a["image_id"]].append(
                mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))
    recs_by_stem = defaultdict(list)
    for rec, rles in gt_by_rec.items():
        recs_by_stem[id2stem[rec]].append(rles)

    sum_iou = tp = fp = fn = 0
    fp_complete = fp_partial = 0
    near_miss = 0
    area_ratio_tp, area_ratio_nm = [], []

    for stem in sorted(preds):
        items = sorted(preds[stem], key=lambda x: -x["score"])
        used = np.zeros((H, W), bool)
        kept = []
        for it in items:                      # 排他ペイント（提出と同じ手順）
            if it["score"] < thr:
                continue
            m = mask_util.decode({"size": [H, W], "counts": it["rle"].encode()}).astype(bool)
            m &= ~used
            if m.sum() < max(min_area, 1):
                continue
            used |= m
            kept.append(mask_util.encode(np.asfortranarray(m.astype(np.uint8))))
        for gts in recs_by_stem.get(stem, [[]]):
            if not gts:
                fp += len(kept); fp_complete += len(kept)
                continue
            if not kept:
                fn += len(gts)
                continue
            iou = mask_util.iou(kept, gts, [0] * len(gts))   # (n_pred, n_gt)
            hit = iou > 0.5
            sum_iou += float(iou[hit].sum())
            tp += int(hit.sum())
            pred_best = iou.max(1)
            gt_best = iou.max(0)
            no_hit_p = hit.sum(1) == 0
            fp += int(no_hit_p.sum())
            fp_complete += int((pred_best[no_hit_p] < 0.1).sum())
            fp_partial += int(((pred_best[no_hit_p] >= 0.1)).sum())
            no_hit_g = hit.sum(0) == 0
            fn += int(no_hit_g.sum())
            near_miss += int(((gt_best[no_hit_g] >= 0.3) & (gt_best[no_hit_g] < 0.5)).sum())
            # 面積比（予測/GT）: TP と 惜しい外れ で比べる
            for j in range(len(gts)):
                ga = mask_util.area(gts[j])
                i_best = int(np.argmax(iou[:, j]))
                pa = mask_util.area(kept[i_best])
                if iou[i_best, j] > 0.5:
                    area_ratio_tp.append(pa / max(ga, 1))
                elif 0.3 <= iou[i_best, j] < 0.5:
                    area_ratio_nm.append(pa / max(ga, 1))

    den = tp + 0.5 * fp + 0.5 * fn
    pq = sum_iou / den
    print(f"=== 誤差分解（{len(preds)}枚, thr={thr} area={min_area}）===")
    print(f"  PQ={pq:.4f}  SQ={sum_iou/max(tp,1):.4f}  RQ={tp/den:.4f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")
    print(f"\n  FP の内訳:")
    print(f"    完全FP (IoU<0.1)  {fp_complete:5d}  → 全て消せれば PQ {sum_iou/(tp+0.5*(fp-fp_complete)+0.5*fn):.4f} (+{sum_iou/(tp+0.5*(fp-fp_complete)+0.5*fn)-pq:.4f})")
    print(f"    部分的な重なり     {fp_partial:5d}")
    print(f"\n  FN の内訳:")
    nm_pq = (sum_iou + near_miss * 0.6) / (tp + near_miss + 0.5 * fp + 0.5 * (fn - near_miss))
    print(f"    惜しい外れ(IoU 0.3-0.5) {near_miss:5d}  → 全て救えれば PQ {nm_pq:.4f} (+{nm_pq-pq:.4f})")
    print(f"    その他（ラベルノイズ含む） {fn-near_miss:5d}")
    if area_ratio_tp:
        print(f"\n  面積比（予測/GT）:")
        print(f"    TP        中央値 {np.median(area_ratio_tp):.3f} (n={len(area_ratio_tp)})")
    if area_ratio_nm:
        print(f"    惜しい外れ 中央値 {np.median(area_ratio_nm):.3f} (n={len(area_ratio_nm)})")


if __name__ == "__main__":
    main()
