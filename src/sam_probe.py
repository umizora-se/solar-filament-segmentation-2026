"""SAM 適合性テストとマスク描き直し。

mode=gt   : GT の箱をプロンプトに SAM でマスクを描かせ、GT との IoU を測る。
            「箱が完璧なら SAM は Hα フィラメントをどこまで描けるか」= ドメイン適合の判定。
            参考: MaskDINO の TP 平均 IoU ≒ 0.686（これを大きく超えるなら本物）
mode=preds: 予測 JSON（dump-raw 形式）の各マスクの箱 → SAM マスクに置換。
            *_samrep.json（常に置換）と *_samgd.json（IoU ガード付き置換）を出力。

クロップ方式: 箱の正方形拡張（pad 1.8 倍、最小192px）を切り出して SAM に与える。
フル画像 2048→1024 縮小では細い構造が潰れるため。
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
from filament_dataset import (SINGLECLASS_JSON, TRAIN_IMG_DIR, TEST_IMG_DIR,  # noqa: E402
                              load_folds)

H = W = 2048
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def load_predictor(ckpt: str):
    torch.cuda.set_per_process_memory_fraction(0.85)
    if "sam2" in Path(ckpt).name:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        for cfg in ["configs/sam2.1/sam2.1_hiera_l.yaml", "sam2.1_hiera_l.yaml"]:
            try:
                model = build_sam2(cfg, ckpt, device="cuda")
                break
            except Exception as e:
                err = e
        else:
            raise err
        print(f"[sam] SAM2.1 hiera_large 読込: {ckpt}")
        return SAM2ImagePredictor(model)
    if "sam_hq" in Path(ckpt).name:
        # SAM-HQ: HQ出力トークン + 初期層特徴の融合で細い構造の境界を改善した版
        from segment_anything_hq import sam_model_registry as reg_hq, SamPredictor as SP_hq
        name = "vit_h" if "vit_h" in ckpt else ("vit_l" if "vit_l" in ckpt else "vit_b")
        sam = reg_hq[name](checkpoint=ckpt).cuda().eval()
        print(f"[sam] SAM-HQ {name} 読込: {ckpt}")
        return SP_hq(sam)
    from segment_anything import sam_model_registry, SamPredictor
    name = "vit_h" if "vit_h" in ckpt else ("vit_l" if "vit_l" in ckpt else "vit_b")
    sam = sam_model_registry[name](checkpoint=ckpt).cuda().eval()
    print(f"[sam] {name} 読込: {ckpt}")
    return SamPredictor(sam)


def square_crop(x0, y0, x1, y1, pad=1.8, min_side=192):
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = int(round(min(max(max(x1 - x0, y1 - y0) * pad, min_side), W)))
    a = int(round(cx - side / 2)); b = int(round(cy - side / 2))
    a = max(0, min(a, W - side)); b = max(0, min(b, H - side))
    return a, b, side


def to_rgb(gray_crop, img_mode):
    g = _CLAHE.apply(gray_crop) if img_mode == "clahe" else gray_crop
    return np.stack([g, g, g], -1)


def sam_mask(pred, gray, m, img_mode):
    """m: (H,W) bool の元マスク。箱→クロップ→SAM。返り値: (H,W) bool（失敗時 None）"""
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    a, b, s = square_crop(x0, y0, x1, y1)
    pred.set_image(to_rgb(gray[b:b + s, a:a + s], img_mode))
    box = np.array([x0 - a, y0 - b, x1 - a, y1 - b], np.float32)
    masks, scores, _ = pred.predict(box=box, multimask_output=True)
    out = np.zeros((H, W), bool)
    out[b:b + s, a:a + s] = masks[int(np.argmax(scores))]
    return out


def iou_bool(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


# --------------------------------------------------------------------------- #
def run_gt(args, pred):
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    fold_of = load_folds()
    anns = [a for a in coco["annotations"]
            if fold_of.get(id2stem[a["image_id"]]) == args.fold
            and a.get("segmentation") and len(a["segmentation"][0]) >= 6]
    rng = np.random.RandomState(0)
    rng.shuffle(anns)
    anns = anns[: args.limit] if args.limit else anns
    print(f"[gt] fold{args.fold} val インスタンス {len(anns)} 件  img_mode={args.img_mode}")

    cache = {}
    stat = defaultdict(list)
    t0 = time.time()
    for i, a in enumerate(anns, 1):
        stem = id2stem[a["image_id"]]
        if stem not in cache:
            cache.clear()
            cache[stem] = cv2.imread(str(Path(TRAIN_IMG_DIR) / f"{stem}.jpeg"),
                                     cv2.IMREAD_GRAYSCALE)
        gray = cache[stem]
        gm = mask_util.decode(mask_util.merge(
            mask_util.frPyObjects(a["segmentation"], H, W))).astype(bool)
        sm = sam_mask(pred, gray, gm, args.img_mode)
        if sm is None:
            continue
        iou = iou_bool(sm, gm)
        area = gm.sum()
        bucket = "small<500" if area < 500 else ("mid" if area < 2000 else "large>2000")
        stat["all"].append(iou); stat[bucket].append(iou)
        if i % 50 == 0 or i == len(anns):
            el = time.time() - t0
            print(f"  [{i}/{len(anns)}] 平均IoU={np.mean(stat['all']):.4f} "
                  f"({el:.0f}s 残り{el/i*(len(anns)-i)/60:.1f}分)", flush=True)
    print(f"\n=== SAM 適合性（GT箱プロンプト, img_mode={args.img_mode}）===")
    for k in ["all", "small<500", "mid", "large>2000"]:
        if stat[k]:
            v = np.asarray(stat[k])
            print(f"  {k:>10}: 平均IoU={v.mean():.4f}  IoU>0.5率={(v>.5).mean():.3f}  n={len(v)}")
    print("  参考: MaskDINO の TP 平均 IoU ≒ 0.686")
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"img_mode": args.img_mode, "mean_iou": float(np.mean(stat["all"])),
             "hit_rate": float((np.asarray(stat["all"]) > .5).mean()),
             "n": len(stat["all"])}))
        print(f"[write] {args.out}")


def run_preds(args, pred):
    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    rep, gd = {}, {}
    t0 = time.time()
    stems = sorted(preds)
    n_rep = n_keep = 0
    for i, stem in enumerate(stems, 1):
        p = Path(TRAIN_IMG_DIR) / f"{stem}.jpeg"
        if not p.exists():
            p = Path(TEST_IMG_DIR) / f"{stem}.jpeg"
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        rep[stem], gd[stem] = [], []
        for it in preds[stem]:
            m = mask_util.decode({"size": [H, W], "counts": it["rle"].encode()}).astype(bool)
            sm = sam_mask(pred, gray, m, args.img_mode)
            if sm is None or sm.sum() == 0:
                sm = m
            enc = mask_util.encode(np.asfortranarray(sm.astype(np.uint8)))["counts"].decode()
            rep[stem].append({"rle": enc, "score": it["score"]})
            if iou_bool(sm, m) >= args.guard_iou:
                gd[stem].append({"rle": enc, "score": it["score"]}); n_rep += 1
            else:
                gd[stem].append(it); n_keep += 1
        if i % 10 == 0 or i == len(stems):
            el = time.time() - t0
            print(f"  [{i}/{len(stems)}] 置換={n_rep} 保持={n_keep} "
                  f"({el:.0f}s 残り{el/i*(len(stems)-i)/60:.1f}分)", flush=True)
    base = Path(args.preds)
    for tag, d in [("samrep", rep), ("samgd", gd)]:
        o = base.with_name(base.stem + f"_{tag}.json")
        o.write_text(json.dumps({"preds": d}))
        print(f"[write] {o}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["gt", "preds"], required=True)
    ap.add_argument("--ckpt", default=str(REPO / "weights" / "sam_vit_h_4b8939.pth"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--img-mode", choices=["raw", "clahe"], default="raw")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--preds", default=None)
    ap.add_argument("--guard-iou", type=float, default=0.25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pred = load_predictor(args.ckpt)
    with torch.inference_mode():
        (run_gt if args.mode == "gt" else run_preds)(args, pred)


if __name__ == "__main__":
    main()
