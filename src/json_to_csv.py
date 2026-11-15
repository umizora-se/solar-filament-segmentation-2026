"""生予測 JSON + 後処理パラメータ -> 提出 CSV（排他ペイント + 自己検証）。
    python src/json_to_csv.py --preds test_raw.json --postproc postproc.json --out sub.csv
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

H = W = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--postproc", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pp = json.loads(Path(args.postproc).read_text())
    score_thr, min_area = pp["score_thr"], pp["min_area"]
    # 残存率ルール: 排他ペイント後に元マスクの min_keep_frac 未満しか残らない断片を棄却（0 なら従来と同一）
    keep_frac = float(pp.get("min_keep_frac", 0.0))
    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    print(f"[json_to_csv] {len(preds)} 画像  score_thr={score_thr} min_area={min_area} min_keep_frac={keep_frac}")

    rows = []
    n_inst = 0
    for stem in sorted(preds):
        items = sorted(preds[stem], key=lambda x: -x["score"])
        used = np.zeros((H, W), bool)
        k = 0
        for it in items:
            if it["score"] < score_thr:
                continue
            m = mask_util.decode({"size": [H, W], "counts": it["rle"].encode()}).astype(bool)
            orig = int(m.sum())
            m &= ~used
            if m.sum() < max(min_area, 1) or m.sum() < keep_frac * orig:
                continue
            used |= m
            k += 1
            rle = mask_util.encode(np.asfortranarray(m.astype(np.uint8)))["counts"].decode()
            rows.append((f"{stem}_{k}", rle))
        n_inst += k
    with open(args.out, "w") as f:
        f.write("filament_id,segmentation_rle\n")
        for fid, rle in rows:
            f.write(f'{fid},"{rle}"\n')
    print(f"[write] {args.out}  {n_inst} 予測 / {len(preds)} 画像")

    from predict_submit import verify_submission
    verify_submission(args.out)


if __name__ == "__main__":
    main()
