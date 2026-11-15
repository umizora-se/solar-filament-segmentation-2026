"""セマンティック U-Net のタイル推論（1024 タイル、stride 512、平均合成）。
出力: npz {stem: (2048,2048) uint8 確率×255}
    python src/semseg_infer.py --weights work_dirs/semseg_f0/semseg_best.pth \
        --fold 0 --split val --out work_dirs/semseg_f0/probs_val.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TRAIN_IMG_DIR, TEST_IMG_DIR, load_folds  # noqa: E402
from train_semseg import SemsegUNet, build_arch, _CLAHE, CROP  # noqa: E402

H = W = 2048


def infer_image(model, gray):
    x = np.stack([gray / 255.0, _CLAHE.apply(gray) / 255.0, gray / 255.0]).astype(np.float32)
    x = torch.from_numpy(x).cuda()
    acc = torch.zeros((H, W), device="cuda")
    cnt = torch.zeros((H, W), device="cuda")
    for y0 in range(0, H - CROP + 1, 512):
        for x0 in range(0, W - CROP + 1, 512):
            with torch.amp.autocast("cuda"):
                p = torch.sigmoid(model(x[None, :, y0:y0 + CROP, x0:x0 + CROP]))[0].float()
            acc[y0:y0 + CROP, x0:x0 + CROP] += p
            cnt[y0:y0 + CROP, x0:x0 + CROP] += 1
    return (acc / cnt * 255).byte().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.85)
    ck = torch.load(args.weights, map_location="cuda")
    if isinstance(ck, dict) and "state" in ck:      # 新形式（arch を記録）
        model = (SemsegUNet() if ck["arch"] == "baseline"
                 else build_arch(ck["arch"], ck["encoder"])).cuda().eval()
        model.load_state_dict(ck["state"])
        print(f"[model] arch={ck['arch']} encoder={ck['encoder']}")
    else:                                            # 旧形式（baseline のみ）
        model = SemsegUNet().cuda().eval()
        model.load_state_dict(ck)

    if args.split == "val":
        fold_of = load_folds()
        stems = sorted(s for s, f in fold_of.items() if f == args.fold)
        img_dir = Path(TRAIN_IMG_DIR)
    else:
        img_dir = Path(TEST_IMG_DIR)
        stems = sorted(p.stem for p in img_dir.glob("*.jpeg"))
    print(f"[semseg_infer] {args.split}: {len(stems)} 枚")

    out = {}
    t0 = time.time()
    with torch.inference_mode():
        for i, st in enumerate(stems, 1):
            gray = cv2.imread(str(img_dir / f"{st}.jpeg"), cv2.IMREAD_GRAYSCALE)
            out[st] = infer_image(model, gray)
            if i % 20 == 0 or i == len(stems):
                el = time.time() - t0
                print(f"  [{i}/{len(stems)}] {el:.0f}s 残り{el/i*(len(stems)-i)/60:.1f}分",
                      flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
