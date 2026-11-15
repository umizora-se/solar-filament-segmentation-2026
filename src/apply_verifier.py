"""セマンティック確率マップで予測を検証（減点/棄却）する。マスクは一切触らない。

各予測についてマスク内平均確率 semprob を測り、
  --mode mul  : score' = score * semprob
  --mode sqrt : score' = score * sqrt(semprob)
  --mode gate : semprob < --sem-thr の予測を捨てる（score は不変）
出力は同じ JSON 形式（tune_postproc に渡せる）。

    python src/apply_verifier.py --preds val_raw_tta.json \
        --probs work_dirs/semseg_f0/probs_val.npz --mode gate --sem-thr 0.10 \
        --out val_raw_tta_vgate10.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

H = W = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--mode", choices=["mul", "sqrt", "gate"], required=True)
    ap.add_argument("--sem-thr", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    probs = np.load(args.probs)

    out = {}
    sems = []
    n_drop = 0
    for stem, items in preds.items():
        if stem not in probs.files:
            out[stem] = items
            continue
        pm = probs[stem].astype(np.float32) / 255.0
        kept = []
        for it in items:
            m = mask_util.decode({"size": [H, W], "counts": it["rle"].encode()}).astype(bool)
            sp = float(pm[m].mean()) if m.any() else 0.0
            sems.append(sp)
            if args.mode == "gate":
                if sp < args.sem_thr:
                    n_drop += 1
                    continue
                kept.append(it)
            else:
                f = sp if args.mode == "mul" else float(np.sqrt(sp))
                kept.append({"rle": it["rle"], "score": it["score"] * f})
        out[stem] = kept

    sems = np.asarray(sems)
    print(f"[verifier] mode={args.mode}  semprob 分布: "
          f"p10={np.percentile(sems,10):.3f} med={np.median(sems):.3f} "
          f"p90={np.percentile(sems,90):.3f}  棄却={n_drop}")
    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
