"""
Stage1 の予測に crop refiner を適用して、精細化＋棄却した予測を出力する。

入力  : predict_submit.py --dump-raw が出した JSON（Stage1 の生予測 RLE+score）
出力  : 同じ形式の JSON（マスクを精細化し、棄却ヘッドで落としたものを除外）
        score は Stage1 のスコアと実在確率を掛け合わせた値に更新する

これで tune_postproc.py / predict_submit.py の後段をそのまま再利用できる。

使い方:
  python src/apply_refiner.py \
      --preds work_dirs/r50_f0/val_raw_tta.json \
      --refiner work_dirs/refiner_f0/refiner_best.pth \
      --out work_dirs/r50_f0/val_raw_tta_refined.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from refiner_data import square_box, extract, CROP, PAD  # noqa: E402
from refiner_model import CropRefiner                     # noqa: E402
from train_refiner import make_seed                       # noqa: E402
from filament_dataset import TRAIN_IMG_DIR, TEST_IMG_DIR  # noqa: E402

H = W = 2048
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def find_image(stem: str) -> Path | None:
    for d in (TRAIN_IMG_DIR, TEST_IMG_DIR):
        p = Path(d) / f"{stem}.jpeg"
        if p.exists():
            return p
    return None


@torch.no_grad()
def refine_image(model, img_gray: np.ndarray, rles: list, scores: np.ndarray,
                 device="cuda", batch=32, keep_thr: float = 0.5,
                 mask_thr: float = 0.5, blend: str = "replace",
                 score_mult: bool = True):
    """1枚ぶんの予測を精細化する。戻り値: (RLE list, score list)。"""
    boxes, crops, seeds = [], [], []
    for r in rles:
        m = mask_util.decode(r).astype(np.uint8)
        ys, xs = np.nonzero(m)
        if ys.size == 0:
            boxes.append(None); continue
        box = square_box(float(xs.min()), float(ys.min()),
                         float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
        # seed は Stage1 の予測マスクそのもの（箱ではない）。
        # 箱だけ渡すと refiner が Stage1 のマスク情報を捨ててしまう。
        g, sd = extract(img_gray, box, m)
        if g is None:
            boxes.append(None); continue
        boxes.append(box)
        crops.append(g)
        seeds.append(sd.astype(np.float32))

    if not crops:
        return [], []

    X = np.stack([
        np.stack([c.astype(np.float32) / 255.0,
                  _CLAHE.apply(c).astype(np.float32) / 255.0,
                  sd], 0)
        for c, sd in zip(crops, seeds)
    ])
    segs, clss = [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        with torch.amp.autocast("cuda"):
            sg, cl = model(xb)
        segs.append(torch.sigmoid(sg).float().cpu().numpy())
        clss.append(torch.sigmoid(cl).float().cpu().numpy())
    segs = np.concatenate(segs); clss = np.concatenate(clss)

    out_rles, out_scores = [], []
    k = 0
    for j, box in enumerate(boxes):
        if box is None:
            continue
        prob, real = segs[k], float(clss[k]); k += 1
        if real < keep_thr:                       # 棄却ヘッドで落とす
            continue
        X0, Y0, side = box
        s = int(round(side))
        x0, y0 = int(round(X0)), int(round(Y0))
        x1, y1 = min(x0 + s, W), min(y0 + s, H)
        x0, y0 = max(x0, 0), max(y0, 0)
        pm = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        full = np.zeros((H, W), np.uint8)
        full[y0:y1, x0:x1] = (pm > mask_thr).astype(np.uint8)
        if blend == "union":                      # 元マスクと和を取る保守的な設定
            full |= mask_util.decode(rles[j]).astype(np.uint8)
        if full.sum() == 0:
            continue
        out_rles.append(mask_util.encode(np.asfortranarray(full)))
        # score_mult=False なら Stage1 のスコアをそのまま使う。
        # 実在確率を掛けるとスコア分布が下方シフトし、後段の閾値と噛み合わなくなる。
        out_scores.append(float(scores[j]) * real if score_mult else float(scores[j]))
    return out_rles, out_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--refiner", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-thr", type=float, default=0.5, help="棄却ヘッドのしきい値")
    ap.add_argument("--mask-thr", type=float, default=0.5)
    ap.add_argument("--blend", choices=["replace", "union"], default="replace")
    ap.add_argument("--no-score-mult", action="store_true",
                    help="スコアに実在確率を掛けない（Stage1のスコアを保つ）")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.85)
    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    stems = sorted(preds)
    if args.limit:
        stems = stems[: args.limit]

    ck = torch.load(args.refiner, map_location="cpu", weights_only=False)
    model = CropRefiner(None).cuda()
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[refiner] epoch={ck.get('epoch')} val_iou={ck.get('val_iou'):.4f}")

    out = {}
    n_in = n_out = 0
    t0 = time.time()
    for i, stem in enumerate(stems, 1):
        items = preds[stem]
        p = find_image(stem)
        if p is None or not items:
            out[stem] = items
            continue
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
        sc = np.asarray([it["score"] for it in items], np.float32)
        r2, s2 = refine_image(model, img, rles, sc, keep_thr=args.keep_thr,
                              mask_thr=args.mask_thr, blend=args.blend,
                              score_mult=not args.no_score_mult)
        out[stem] = [{"rle": r["counts"].decode(), "score": float(s)}
                     for r, s in zip(r2, s2)]
        n_in += len(items); n_out += len(r2)
        if i % 20 == 0 or i == len(stems):
            el = time.time() - t0
            print(f"  [{i:4d}/{len(stems)}] 入力{n_in} -> 出力{n_out} "
                  f"(棄却 {n_in-n_out}, {el:.0f}s, 残り約{el/i*(len(stems)-i)/60:.1f}分)", flush=True)

    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"\n[write] {args.out}")
    print(f"  予測 {n_in} -> {n_out} （棄却 {n_in-n_out} = {(n_in-n_out)/max(n_in,1)*100:.1f}%）")


if __name__ == "__main__":
    main()
