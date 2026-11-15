"""
複数 fold モデルの予測を融合する（Weighted Mask Fusion）。

各モデルの予測 JSON（predict_submit.py --dump-raw の出力）を受け取り、
モデル横断で IoU クラスタリング → 票決 → 画素多数決でマスクを統合する。
出力は同じ JSON 形式なので、既存の tune_postproc.py / predict_submit.py に渡せる。

TTA のビュー間票決と同じ仕組みだが、票を投じるのが「同じモデルの別の向き」ではなく
「別データで学習した別モデル」なので、多様性が本質的に高い。

  python src/ensemble.py --preds work_dirs/r50_f0/val_raw_tta.json \
                                 work_dirs/r50_f1/val_raw_tta.json \
      --out work_dirs/ens/val_raw_ens.json --min-votes 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

H = W = 2048


def fuse_image(per_model: list, cluster_iou: float = 0.5, min_votes: int = 2,
               score_mode: str = "mean_vote"):
    """per_model: [[{rle, score}, ...] モデルごと] -> ([RLE dict], [score])。"""
    pool = []                       # (rle_dict, score, model_id)
    for mid, items in enumerate(per_model):
        for it in items:
            pool.append(({"size": [H, W], "counts": it["rle"].encode()},
                         float(it["score"]), mid))
    if not pool:
        return [], []
    n_models = len(per_model)

    order = sorted(range(len(pool)), key=lambda i: -pool[i][1])
    rles = [pool[i][0] for i in order]
    used = set()
    out_rles, out_scores = [], []

    for a in range(len(order)):
        if a in used:
            continue
        anchor_mid = pool[order[a]][2]
        members = {anchor_mid: (a, pool[order[a]][1])}
        cand = [b for b in range(a + 1, len(order))
                if b not in used and pool[order[b]][2] != anchor_mid]
        if cand:
            ious = mask_util.iou([rles[a]], [rles[b] for b in cand], [0] * len(cand))[0]
            for b, iou in zip(cand, ious):
                mid = pool[order[b]][2]
                if iou > cluster_iou and mid not in members:
                    members[mid] = (b, pool[order[b]][1])
        used.add(a)
        for b, _ in members.values():
            used.add(b)

        votes = len(members)
        if votes < min_votes:
            continue

        # 画素多数決（半数以上のモデルが立てた画素を採用）
        acc = np.zeros((H, W), np.uint8)
        for b, _ in members.values():
            acc += mask_util.decode(rles[b])
        merged = (acc >= (votes + 1) // 2).astype(np.uint8)
        if merged.sum() == 0:
            merged = mask_util.decode(rles[a])
            if merged.sum() == 0:
                continue

        sc = np.mean([s for _, s in members.values()])
        if score_mode == "mean_vote":       # 票数で重み付け（全モデル一致を優遇）
            sc *= votes / n_models
        out_rles.append(mask_util.encode(np.asfortranarray(merged)))
        out_scores.append(float(sc))
    return out_rles, out_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-votes", type=int, default=2)
    ap.add_argument("--cluster-iou", type=float, default=0.5)
    ap.add_argument("--score-mode", choices=["mean", "mean_vote"], default="mean_vote")
    args = ap.parse_args()

    models = []
    for p in args.preds:
        d = json.loads(Path(p).read_text())
        models.append(d["preds"] if "preds" in d else d)
        print(f"  読込 {p}: {len(models[-1])} 画像 / "
              f"{sum(len(v) for v in models[-1].values())} 予測")

    stems = sorted(set().union(*[set(m) for m in models]))
    print(f"[fuse] {len(models)} モデル × {len(stems)} 画像  "
          f"min_votes={args.min_votes} score_mode={args.score_mode}")

    out = {}
    n_in = n_out = 0
    t0 = time.time()
    for i, s in enumerate(stems, 1):
        per = [m.get(s, []) for m in models]
        n_in += sum(len(x) for x in per)
        r, sc = fuse_image(per, args.cluster_iou, args.min_votes, args.score_mode)
        out[s] = [{"rle": x["counts"].decode(), "score": float(v)} for x, v in zip(r, sc)]
        n_out += len(r)
        if i % 20 == 0 or i == len(stems):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(stems)}] 入力{n_in} -> 融合{n_out} "
                  f"({el:.0f}s, 残り約{el/i*(len(stems)-i)/60:.1f}分)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"\n[write] {args.out}")
    print(f"  予測 {n_in} -> {n_out}  平均 {n_out/max(len(stems),1):.1f} 個/画像")


if __name__ == "__main__":
    main()
