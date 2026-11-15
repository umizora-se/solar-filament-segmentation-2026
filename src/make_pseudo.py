"""教師モデルの予測（predict_submit --dump-raw）から疑似ラベルを作る。

後処理は提出と同じ（スコア足切り → 画素排他ペイント → 最小面積）。足切りは OOF で
PQ 最大になった値を使う＝「提出するならこう出す」予測をそのまま正解とみなす。
マスクは RLE のまま保存する（ポリゴンに直すと細い個体の輪郭画素が落ち、往復 IoU が
0.91 まで下がった。学習側の mapper は疑似ラベル画像だけ RLE をそのまま読む）。

    python src/make_pseudo.py --preds work_dirs/pl_teacher_f0.json --fold 0 \\
        --out datasets/gong_unlabeled/pseudo_f0.json --score-thr 0.35 --min-area 200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from postproc import paint_disjoint  # noqa: E402

H = W = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--fold", type=int, required=True, help="教師モデルの fold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--score-thr", type=float, default=0.35)
    ap.add_argument("--min-area", type=int, default=200)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    raw = raw["preds"] if "preds" in raw else raw
    labels, n_inst = {}, 0
    for stem, items in raw.items():
        ms = [mask_util.decode({"size": [H, W], "counts": it["rle"].encode()}).astype(bool)
              for it in items]
        kept = paint_disjoint(ms, np.asarray([it["score"] for it in items]),
                              args.score_thr, args.min_area)
        labels[stem] = [mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
                        for m in kept]
        n_inst += len(kept)
    Path(args.out).write_text(json.dumps({
        "teacher_fold": args.fold, "teacher_preds": str(args.preds),
        "score_thr": args.score_thr, "min_area": args.min_area, "labels": labels}))
    print(f"[pseudo] {len(labels)} 枚  個体 {n_inst}（1枚あたり {n_inst/max(len(labels),1):.1f}）"
          f"  空の画像 {sum(not v for v in labels.values())}")
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
