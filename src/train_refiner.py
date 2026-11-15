"""
crop refiner の学習。

入力3ch = [gray, CLAHE(gray), seed]
  seed は「検出箱を塗りつぶした矩形」。学習時はジッタさせ、推論時は Stage1 の予測箱を使う
  （上位解法 YOLO->crop U-Net と同じ表現）。

損失
  segmentation : Dice + 0.5*BCE（正例サンプルのみ。細長い対象で正例画素が1-3%のため Dice 主体）
  rejection    : BCE（全サンプル。正例=1 / Stage1の完全FP=0）

使い方:
  python src/train_refiner.py --train data/refiner_f0_train.npz --val data/refiner_f0_val.npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from refiner_model import CropRefiner, dice_bce_loss  # noqa: E402

CROP = 256
PAD = 0.40
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def make_seed(jitter_px: int = 0, crop: int = CROP, pad: float = PAD) -> np.ndarray:
    """（旧）クロップ中央に置かれた検出箱を塗りつぶした矩形。

    上位解法は Stage1 が YOLO（箱のみ）だったためこの表現を使っていたが、
    我々の Stage1 は MaskDINO で完全なマスクを持つ。箱を渡すとマスク情報を
    捨ててしまい、refiner がゼロから再セグメンテーションしてしまうため使わない。
    """
    half = crop / (2.0 * (1.0 + 2.0 * pad))
    cx = cy = crop / 2.0
    if jitter_px:
        cx += np.random.uniform(-jitter_px, jitter_px)
        cy += np.random.uniform(-jitter_px, jitter_px)
        half *= np.random.uniform(0.85, 1.15)
    s = np.zeros((crop, crop), np.float32)
    y0, y1 = int(max(0, cy - half)), int(min(crop, cy + half))
    x0, x1 = int(max(0, cx - half)), int(min(crop, cx + half))
    s[y0:y1, x0:x1] = 1.0
    return s


def degrade_seed(seed: np.ndarray) -> np.ndarray:
    """学習時、GT マスクを Stage1 の不完全なマスクらしく劣化させる。

    推論時に渡るのは Stage1 の予測マスクなので、学習でも「少しずれた・
    太った/痩せたマスク」を見せておく（上位解法の dilate/erode + shift と同じ発想）。
    """
    s = seed.astype(np.uint8)
    r = np.random.rand()
    k = np.random.randint(3, 10) | 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if r < 0.35:
        s = cv2.dilate(s, ker)
    elif r < 0.70:
        s = cv2.erode(s, ker)
    dx, dy = np.random.randint(-8, 9, 2)
    s = np.roll(np.roll(s, dy, 0), dx, 1)
    return s.astype(np.float32)


class RefinerDataset(Dataset):
    def __init__(self, npz_path: str, train: bool = True, jitter_px: int = 12):
        d = np.load(npz_path)
        self.gray = d["gray"]          # (N,256,256) uint8
        self.mask = d["mask"]          # (N,256,256) uint8 目標マスク
        self.seed = d["seed"]          # (N,256,256) uint8 入力の粗マスク
        self.label = d["label"].astype(np.float32)
        self.train = train
        self.jitter = jitter_px if train else 0

    def __len__(self):
        return len(self.gray)

    def __getitem__(self, i):
        g = self.gray[i]
        m = self.mask[i].astype(np.float32)
        sd = self.seed[i].astype(np.float32)
        lab = self.label[i]

        if self.train:                                   # 幾何拡張は3枚に同じ変換をかける
            k = np.random.randint(4)
            if k:
                g, m, sd = np.rot90(g, k), np.rot90(m, k), np.rot90(sd, k)
            if np.random.rand() < 0.5:
                g, m, sd = g[:, ::-1], m[:, ::-1], sd[:, ::-1]
            if np.random.rand() < 0.5:
                g, m, sd = g[::-1], m[::-1], sd[::-1]
            g, m, sd = (np.ascontiguousarray(g), np.ascontiguousarray(m),
                        np.ascontiguousarray(sd))
            if lab > 0.5:                                # 正例の seed は GT なので劣化させる
                sd = degrade_seed(sd)
            if np.random.rand() < 0.5:                   # 観測所間の輝度差に対応
                a = np.random.uniform(0.85, 1.15)
                b = np.random.uniform(-15, 15)
                g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        cl = _CLAHE.apply(g)
        x = np.stack([g.astype(np.float32) / 255.0,
                      cl.astype(np.float32) / 255.0,
                      sd], 0)
        return torch.from_numpy(x), torch.from_numpy(m), torch.tensor(lab)


def evaluate(model, loader, device):
    model.eval()
    inter = union = 0.0
    n_seg = 0
    cls_correct = cls_n = 0
    probs, labs = [], []
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for x, m, lab in loader:
            x, m, lab = x.to(device), m.to(device), lab.to(device)
            seg, cls = model(x)
            p = (torch.sigmoid(seg) > 0.5).float()
            pos = lab > 0.5
            if pos.any():
                pp, mm = p[pos], m[pos]
                inter += (pp * mm).sum().item()
                union += ((pp + mm) > 0).float().sum().item()
                n_seg += int(pos.sum())
            pred = (torch.sigmoid(cls) > 0.5).float()
            cls_correct += (pred == lab).sum().item()
            cls_n += lab.numel()
            probs.append(torch.sigmoid(cls).float().cpu()); labs.append(lab.cpu())
    iou = inter / max(union, 1)
    probs = torch.cat(probs).numpy(); labs = torch.cat(labs).numpy()
    # 棄却ヘッドの分離能: 正例と負例の平均確率
    p_pos = probs[labs > 0.5].mean() if (labs > 0.5).any() else float("nan")
    p_neg = probs[labs < 0.5].mean() if (labs < 0.5).any() else float("nan")
    model.train()
    return iou, cls_correct / max(cls_n, 1), p_pos, p_neg, n_seg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--ckpt", default=str(REPO / "work_dirs/pretrained/"
                    "maskdino_r50_50ep_300q_hid1024_3sd1_instance_maskenhanced_mask46.1ap_box51.5ap.pth"))
    ap.add_argument("--out", default=str(REPO / "work_dirs/refiner_f0"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cls-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(0.85)     # WSL2 フリーズ防止

    tr = RefinerDataset(args.train, train=True)
    va = RefinerDataset(args.val, train=False)
    print(f"[data] train {len(tr)} (正例 {int(tr.label.sum())} / 負例 {int((tr.label==0).sum())})")
    print(f"[data] val   {len(va)} (正例 {int(va.label.sum())} / 負例 {int((va.label==0).sum())})")

    dl_tr = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    n_pos, n_neg = float(tr.label.sum()), float((tr.label == 0).sum())
    pw = torch.tensor(max(n_neg / max(n_pos, 1), 1e-3), device=dev)
    print(f"[loss] 棄却ヘッドの pos_weight = {pw.item():.3f}  (正例{n_pos:.0f} / 負例{n_neg:.0f})")

    model = CropRefiner(args.ckpt).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")

    best = -1.0
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); agg = {}
        for x, m, lab in dl_tr:
            x, m, lab = x.to(dev, non_blocking=True), m.to(dev, non_blocking=True), lab.to(dev)
            with torch.amp.autocast("cuda"):
                seg, cls = model(x)
                pos = lab > 0.5
                if pos.any():
                    l_seg, d, b = dice_bce_loss(seg[pos], m[pos])
                else:
                    l_seg = seg.sum() * 0.0; d = b = 0.0
                # 正例が負例の数倍あるため、負例側を重く見る（pos_weight<1 で正例を割り引く）
                l_cls = F.binary_cross_entropy_with_logits(cls, lab, pos_weight=pw)
                loss = l_seg + args.cls_weight * l_cls
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in (("loss", loss.item()), ("dice", d), ("bce", b), ("cls", l_cls.item())):
                agg[k] = agg.get(k, 0.0) + v
        n = len(dl_tr)
        iou, acc, p_pos, p_neg, n_seg = evaluate(model, dl_va, dev)
        print(f"ep {ep:3d}/{args.epochs} | loss {agg['loss']/n:6.3f} "
              f"(dice {agg['dice']/n:.3f} bce {agg['bce']/n:.3f} cls {agg['cls']/n:.3f}) | "
              f"val IoU {iou:.4f} 判定acc {acc:.3f} P(pos) {p_pos:.3f} P(neg) {p_neg:.3f} | "
              f"{time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "epoch": ep, "val_iou": iou},
                   out / "refiner_last.pth")
        if iou > best:
            best = iou
            torch.save({"model": model.state_dict(), "epoch": ep, "val_iou": iou},
                       out / "refiner_best.pth")
    print(f"\n[done] 最良 val IoU {best:.4f} -> {out/'refiner_best.pth'}")


if __name__ == "__main__":
    main()
