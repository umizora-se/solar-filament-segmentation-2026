"""過分割された予測を統合する（惜しい外れ対策）。

診断（OOF 707枚）:
  惜しい外れ 926本のうち 169本は予測が2本以上乗っている＝1本を割っている。
  そのうち 130本は統合すれば IoU>0.5 になる（上限 +0.013）。

統合すべきペアと、そうでないペアの特徴（実測）:
                隙間px      充填率
  統合すべき     中央 28.6   中央 0.125
  統合すべきでない 中央 61.6   中央 0.067

マスクを描き直すのではなく「予測をどう束ねるか」を変えるだけなので、
これまで5回失敗した「マスク再生成」とは別の系統に属する。

    python src/merge_preds.py --preds oof_raw_tta8_sqrt.json --gap 35 --fill 0.09 \
        --out merged.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

H = W = 2048


def _bbox(rle):
    x, y, w, h = mask_util.toBbox(rle)
    return float(x), float(y), float(x + w), float(y + h)


def pair_feats(ra, rb, ba, bb, aa, ab):
    """RLE のまま特徴を出す（マスク展開はしない。展開すると O(n^2) で破綻する）。"""
    dx = max(0.0, max(ba[0] - bb[2], bb[0] - ba[2]))
    dy = max(0.0, max(ba[1] - bb[3], bb[1] - ba[3]))
    gap = float(np.hypot(dx, dy))
    inter = float(mask_util.area(mask_util.merge([ra, rb], intersect=True)))
    ua = aa + ab - inter                       # 和集合の面積
    bw = max(ba[2], bb[2]) - min(ba[0], bb[0])
    bh = max(ba[3], bb[3]) - min(ba[1], bb[1])
    return gap, ua / max(bw * bh, 1.0)


def merge_image(items, gap_thr, fill_thr, max_rounds=2):
    """スコア降順に、条件を満たすペアを統合する。統合後のスコアは最大値。"""
    rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
    scores = [float(it["score"]) for it in items]
    boxes = [_bbox(r) for r in rles]
    areas = [float(mask_util.area(r)) for r in rles]
    for _ in range(max_rounds):
        best = None
        for i in range(len(rles)):
            if rles[i] is None:
                continue
            for j in range(i + 1, len(rles)):
                if rles[j] is None:
                    continue
                g, fill = pair_feats(rles[i], rles[j], boxes[i], boxes[j],
                                     areas[i], areas[j])
                if g <= gap_thr and fill >= fill_thr:
                    key = (fill, -g)          # 隙間が小さく充填率が高いほど「1本を割った」らしい
                    if best is None or key > best[0]:
                        best = (key, i, j)
        if best is None:
            break
        _, i, j = best
        rles[i] = mask_util.merge([rles[i], rles[j]], intersect=False)
        scores[i] = max(scores[i], scores[j])
        boxes[i] = _bbox(rles[i])
        areas[i] = float(mask_util.area(rles[i]))
        rles[j] = None
    return [{"rle": r["counts"].decode() if isinstance(r["counts"], bytes)
             else r["counts"], "score": s}
            for r, s in zip(rles, scores) if r is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--gap", type=float, default=35.0)
    ap.add_argument("--fill", type=float, default=0.09)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--pre-thr", type=float, default=0.20,
                    help="この値未満のスコアの予測は統合対象にしない")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    out = {}
    n_in = n_out = 0
    for stem in sorted(preds):
        items = [it for it in preds[stem] if it["score"] >= args.pre_thr]
        rest = [it for it in preds[stem] if it["score"] < args.pre_thr]
        n_in += len(preds[stem])
        merged = merge_image(items, args.gap, args.fill, args.rounds) if len(items) > 1 else items
        out[stem] = merged + rest
        n_out += len(out[stem])
    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"[merge] gap<={args.gap} fill>={args.fill} rounds={args.rounds}: "
          f"{n_in} -> {n_out} 予測 ({n_in-n_out} 本を統合)")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
