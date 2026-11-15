"""RF-DETR-Seg でフル 2048 をタイル推論し、MaskDINO と同じ JSON 形式で出す。

学習は 1024 クロップなので推論も 1024 タイル（stride 512 で重ね、境界の取りこぼしを防ぐ）。
タイル間の重複検出は IoU>0.5 の貪欲クラスタリングで統合する。
出力は {"preds": {stem: [{"rle":..., "score":...}]}} で、ensemble.py / tune_postproc.py に渡せる。

    python src/rfdetr_infer.py --weights work_dirs/rfdetr_f0/checkpoint_best_ema.pth \
        --split val --fold 0 --out work_dirs/rfdetr_f0/val_raw.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pycocotools.mask as mask_util
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TRAIN_IMG_DIR, TEST_IMG_DIR, load_folds  # noqa: E402

H = W = 2048
TILE, STRIDE = 1024, 512


def cluster(rles, scores, thr=0.5):
    """スコア降順の貪欲クラスタリング。重複タイル由来の二重検出をまとめる。"""
    if not rles:
        return [], []
    order = np.argsort(-np.asarray(scores))
    keep_r, keep_s = [], []
    for i in order:
        r, s = rles[i], scores[i]
        if keep_r:
            iou = mask_util.iou([r], keep_r, [0] * len(keep_r))[0]
            if iou.max() > thr:
                continue
        keep_r.append(r)
        keep_s.append(float(s))
    return keep_r, keep_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.10, help="収集時の低め閾値")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.85)
    from rfdetr import RFDETRSegPreview
    model = RFDETRSegPreview(resolution=1008, num_classes=1,
                             pretrain_weights=args.weights)
    model.optimize_for_inference()
    print(f"[rfdetr] {args.weights}")

    if args.split == "val":
        fold_of = load_folds()
        stems = sorted(s for s, f in fold_of.items() if f == args.fold)
        img_dir = Path(TRAIN_IMG_DIR)
    else:
        img_dir = Path(TEST_IMG_DIR)
        stems = sorted(p.stem for p in img_dir.glob("*.jpeg"))
    print(f"[infer] {args.split}: {len(stems)} 枚  タイル{TILE}/stride{STRIDE}")

    from PIL import Image
    out = {}
    t0 = time.time()
    for n, stem in enumerate(stems, 1):
        gray = cv2.imread(str(img_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        rles, scores = [], []
        for y0 in range(0, H - TILE + 1, STRIDE):
            for x0 in range(0, W - TILE + 1, STRIDE):
                tile = gray[y0:y0 + TILE, x0:x0 + TILE]
                det = model.predict(Image.fromarray(np.stack([tile] * 3, -1)),
                                    threshold=args.thr)
                if det.mask is None or len(det) == 0:
                    continue
                for m, s in zip(det.mask, det.confidence):
                    full = np.zeros((H, W), np.uint8)
                    mm = m.astype(np.uint8)
                    if mm.shape != (TILE, TILE):
                        mm = cv2.resize(mm, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
                    full[y0:y0 + TILE, x0:x0 + TILE] = mm
                    if full.sum() == 0:
                        continue
                    rles.append(mask_util.encode(np.asfortranarray(full)))
                    scores.append(float(s))
        kr, ks = cluster(rles, scores)
        out[stem] = [{"rle": r["counts"].decode(), "score": s} for r, s in zip(kr, ks)]
        if n % 10 == 0 or n == len(stems):
            el = time.time() - t0
            print(f"  [{n}/{len(stems)}] {stem}: {len(rles)}->{len(kr)} "
                  f"({el:.0f}s 残り{el/n*(len(stems)-n)/60:.1f}分)", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"preds": out}))
    print(f"[write] {args.out}  平均 {np.mean([len(v) for v in out.values()]):.1f} 個/画像")


if __name__ == "__main__":
    main()
