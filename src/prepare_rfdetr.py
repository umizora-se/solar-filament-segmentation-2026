"""RF-DETR 用の COCO データセットを作る（1024 クロップ、縮小なし）。

我々の実測: フィラメントは面積中央値 1,228px（画像の 0.03%）、短辺 p05=20px。
2048 を 640 などに縮小すると消えるため、原寸から 1024 を切り出してそのまま使う。
RF-DETR の resolution は 56 の倍数を要求するので、学習時は 1008 を指定する。

出力: datasets/rfdetr/{train,valid}/ に画像と _annotations.coco.json

    python src/prepare_rfdetr.py --fold 0 --out datasets/rfdetr
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, TRAIN_IMG_DIR, load_folds  # noqa: E402

H = W = 2048
CROP = 1024


def clip_poly(poly, x0, y0):
    """ポリゴンをクロップ座標へ移し、範囲内に収める。外に出るものは None。"""
    p = np.asarray(poly, np.float64).reshape(-1, 2)
    p[:, 0] -= x0
    p[:, 1] -= y0
    inside = ((p[:, 0] >= 0) & (p[:, 0] < CROP) & (p[:, 1] >= 0) & (p[:, 1] < CROP))
    if inside.mean() < 0.5:                # 半分以上が外なら捨てる（半端な教師を作らない）
        return None
    p[:, 0] = np.clip(p[:, 0], 0, CROP - 1)
    p[:, 1] = np.clip(p[:, 1], 0, CROP - 1)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="datasets/rfdetr")
    ap.add_argument("--crops-per-img", type=int, default=4)
    args = ap.parse_args()

    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    fold_of = load_folds()
    by_stem = defaultdict(list)
    for a in coco["annotations"]:
        seg = a.get("segmentation") or []
        if seg and len(seg[0]) >= 6:
            by_stem[id2stem[a["image_id"]]].append(seg[0])

    out = Path(args.out)
    rng = np.random.RandomState(0)
    for split in ["train", "valid"]:
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        stems = sorted(s for s in by_stem
                       if (fold_of.get(s) != args.fold) == (split == "train"))
        images, anns = [], []
        iid = aid = 1
        for si, stem in enumerate(stems, 1):
            gray = cv2.imread(str(Path(TRAIN_IMG_DIR) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            polys = by_stem[stem]
            cents = [(np.asarray(p[0::2]).mean(), np.asarray(p[1::2]).mean()) for p in polys]
            n = args.crops_per_img if split == "train" else 4
            boxes = []
            if split == "valid":               # val は 2x2 タイルで全面を覆う
                boxes = [(x, y) for y in (0, CROP) for x in (0, CROP)]
            else:
                for _ in range(n):             # train は個体中心まわりをランダムに
                    if cents and rng.rand() < 0.8:
                        cx, cy = cents[rng.randint(len(cents))]
                        x0 = int(cx - CROP / 2 + rng.randint(-256, 257))
                        y0 = int(cy - CROP / 2 + rng.randint(-256, 257))
                    else:
                        x0, y0 = rng.randint(0, W - CROP + 1), rng.randint(0, H - CROP + 1)
                    boxes.append((int(np.clip(x0, 0, W - CROP)),
                                  int(np.clip(y0, 0, H - CROP))))
            for x0, y0 in boxes:
                kept = []
                for poly in polys:
                    p = clip_poly(poly, x0, y0)
                    if p is None:
                        continue
                    xs, ys = p[:, 0], p[:, 1]
                    w, h = xs.max() - xs.min(), ys.max() - ys.min()
                    if w < 2 or h < 2:
                        continue
                    kept.append((p, [float(xs.min()), float(ys.min()), float(w), float(h)]))
                if not kept:                    # 空クロップは学習に入れない
                    continue
                fn = f"{stem}_{x0}_{y0}.jpg"
                cv2.imwrite(str(d / fn), gray[y0:y0 + CROP, x0:x0 + CROP],
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                images.append({"id": iid, "file_name": fn, "width": CROP, "height": CROP})
                for p, bbox in kept:
                    anns.append({"id": aid, "image_id": iid, "category_id": 1,
                                 "segmentation": [p.reshape(-1).tolist()],
                                 "bbox": bbox, "area": float(bbox[2] * bbox[3]),
                                 "iscrowd": 0})
                    aid += 1
                iid += 1
            if si % 100 == 0:
                print(f"  {split} [{si}/{len(stems)}] 画像{len(images)} 個体{len(anns)}",
                      flush=True)
        (d / "_annotations.coco.json").write_text(json.dumps({
            "images": images, "annotations": anns,
            "categories": [{"id": 1, "name": "filament", "supercategory": "none"}]}))
        print(f"[{split}] クロップ {len(images)} 枚 / 個体 {len(anns)} 件 -> {d}")


if __name__ == "__main__":
    main()
