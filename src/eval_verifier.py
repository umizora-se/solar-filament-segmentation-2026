"""検証器の分離力を直接測る。

val-IoU は代理指標にすぎない。検証器の仕事は「本物 vs 完全FP」の判別なので、
予測を GT との IoU で TP(>0.5) / 完全FP(<0.1) にラベル付けし、
semprob の AUROC を測る。これが PQ 改善に直結する指標。

    python src/eval_verifier.py --preds work_dirs/r50_f0/val_raw_tta.json \
        --probs work_dirs/semseg_f0/probs_val.npz
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


def auroc(pos, neg):
    """Mann-Whitney U による AUROC。"""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.empty(len(a))
    order = np.argsort(a)
    sa = a[order]
    i = 0
    while i < len(sa):                       # 同値は平均順位
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    probs = np.load(args.probs)

    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    fold_of = load_folds()
    gt = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if fold_of.get(st) != args.fold:
            continue
        seg = a.get("segmentation") or []
        if seg and len(seg[0]) >= 6:
            gt[st].append(mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))

    tp, fp, mid = [], [], []
    for stem, items in preds.items():
        if stem not in probs.files:
            continue
        pm = probs[stem].astype(np.float32) / 255.0
        rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
        if not rles:
            continue
        g = gt.get(stem, [])
        ious = (mask_util.iou(rles, g, [0] * len(g)) if g
                else np.zeros((len(rles), 1)))
        best = ious.max(1) if g else np.zeros(len(rles))
        for it, r, b in zip(items, rles, best):
            m = mask_util.decode(r).astype(bool)
            sp = float(pm[m].mean()) if m.any() else 0.0
            (tp if b > 0.5 else (fp if b < 0.1 else mid)).append(sp)

    tp, fp, mid = map(np.asarray, (tp, fp, mid))
    print(f"[eval] TP={len(tp)} 完全FP={len(fp)} 中間={len(mid)}")
    print(f"  semprob   TP: med={np.median(tp):.3f} p10={np.percentile(tp,10):.3f}")
    print(f"  semprob 完全FP: med={np.median(fp):.3f} p90={np.percentile(fp,90):.3f}")
    a = auroc(tp, fp)
    print(f"  ★ AUROC (TP vs 完全FP) = {a:.4f}   [0.5=無情報, 1.0=完全分離]")
    for t in [0.05, 0.10, 0.20, 0.30]:
        print(f"    gate {t:.2f}: FP除去 {(fp<t).mean()*100:5.1f}%  "
              f"TP巻添え {(tp<t).mean()*100:5.1f}%")
    if args.out:
        Path(args.out).write_text(json.dumps({"auroc": float(a), "n_tp": len(tp),
                                              "n_fp": len(fp)}))
        print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
