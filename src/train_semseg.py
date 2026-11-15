"""独立セマンティック U-Net（FP 検証器用）。

- 教師: 全アノテーターの和集合（union）。「誰かがフィラメントと言った画素」を学ぶ
- エンコーダ: torchvision ResNet-50（ImageNet）。MaskDINO と独立の初期値・学習過程
  → 誤り方の相関を下げる（5-fold 失敗 = 同一アーキ同一初期値、の教訓）
- 用途: MaskDINO 予測のマスク内平均確率で completely-wrong FP（黒点・雲）を減点

    python src/train_semseg.py --fold 0 --out work_dirs/semseg_f0
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, TRAIN_IMG_DIR, load_folds  # noqa: E402

H = W = 2048
CROP = 1024
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# --------------------------------------------------------------------------- #
def build_union_rles(fold: int, target: str = "union"):
    """stem -> (教師RLE, [インスタンス bbox 中心]) を train/val 別に返す。

    target="union"    : 誰か1人でも引いた画素（従来）
    target="majority" : 過半数のアノテーターが引いた画素

    PQ が報いるのは多数派の形である（検出率は合意度で決まる: 全員一致92.8% /
    1人だけ33%）。教師を多数派に合わせると、検証器の判断が採点基準に近づく。
    1人しか引いていない写真では両者は一致する。
    """
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    fold_of = load_folds()
    per_rec, centers = {}, {}          # per_rec: stem -> {record_id: [poly RLE, ...]}
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        seg = a.get("segmentation") or []
        if not seg or len(seg[0]) < 6:
            continue
        per_rec.setdefault(st, {}).setdefault(a["image_id"], []).extend(
            mask_util.frPyObjects([seg[0]], H, W))
        xs = np.asarray(seg[0][0::2]); ys = np.asarray(seg[0][1::2])
        centers.setdefault(st, []).append((float(xs.mean()), float(ys.mean())))

    tr, va = {}, {}
    for st, recs in per_rec.items():
        if target == "union" or len(recs) == 1:
            rle = mask_util.merge([r for pl in recs.values() for r in pl])
        else:
            # アノテーターごとの和集合を作り、画素ごとに票を数えて過半数を採る
            acc = np.zeros((H, W), np.uint8)
            for pl in recs.values():
                acc += mask_util.decode(mask_util.merge(pl))
            need = (len(recs) + 1) // 2                 # 2人なら1、3人なら2
            m = (acc >= need).astype(np.uint8)
            if m.sum() == 0:                            # 念のため空を避ける
                m = (acc >= 1).astype(np.uint8)
            rle = mask_util.encode(np.asfortranarray(m))
        (tr if fold_of.get(st) != fold else va)[st] = (rle, centers[st])
    return tr, va


class SemsegDataset(Dataset):
    def __init__(self, table: dict, train: bool, crops_per_img: int = 2):
        self.stems = sorted(table)
        self.table = table
        self.train = train
        self.cpi = crops_per_img if train else 1

    def __len__(self):
        return len(self.stems) * self.cpi

    def __getitem__(self, idx):
        stem = self.stems[idx // self.cpi]
        rle, centers = self.table[stem]
        gray = cv2.imread(str(Path(TRAIN_IMG_DIR) / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        target = mask_util.decode(rle)
        rng = np.random
        if self.train:
            if centers and rng.rand() < 0.7:      # 70% は個体中心の近くを切る
                cx, cy = centers[rng.randint(len(centers))]
                x0 = int(cx - CROP // 2 + rng.randint(-256, 257))
                y0 = int(cy - CROP // 2 + rng.randint(-256, 257))
            else:
                x0 = rng.randint(0, W - CROP + 1); y0 = rng.randint(0, H - CROP + 1)
            x0 = np.clip(x0, 0, W - CROP); y0 = np.clip(y0, 0, H - CROP)
        else:
            x0 = y0 = (W - CROP) // 2
        g = gray[y0:y0 + CROP, x0:x0 + CROP]
        t = target[y0:y0 + CROP, x0:x0 + CROP]
        if self.train:
            k = rng.randint(4)
            g = np.rot90(g, k); t = np.rot90(t, k)
            if rng.rand() < 0.5:
                g = g[:, ::-1]; t = t[:, ::-1]
            if rng.rand() < 0.5:
                g = g[::-1]; t = t[::-1]
            g = np.clip(g.astype(np.float32) * rng.uniform(0.9, 1.1)
                        + rng.uniform(-10, 10), 0, 255).astype(np.uint8)
        g = np.ascontiguousarray(g); t = np.ascontiguousarray(t)
        x = np.stack([g / 255.0, _CLAHE.apply(g) / 255.0, g / 255.0]).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(t.astype(np.float32))


# --------------------------------------------------------------------------- #
class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


def build_arch(arch: str, encoder: str = "resnet50"):
    """smp の各アーキを同一 API で構築する。arch='baseline' は自前実装。

    比較対象の系統:
      unet      : 素の U-Net（skip は単純結合）
      unetpp    : U-Net++（入れ子の密な skip。細い構造の境界に効くとされる）
      manet     : MA-Net（位置/チャネル注意。Attention U-Net の後継系）
      fpn       : FPN（マルチスケール特徴を明示的に統合）
      deeplabv3p: DeepLabV3+（ASPP で広い受容野。雲など大域文脈の判定向き）
      segformer : Transformer 系（CNN と誤り方が根本的に異なる＝票決の多様性）
    """
    import segmentation_models_pytorch as smp
    m = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "manet": smp.MAnet,
         "fpn": smp.FPN, "deeplabv3p": smp.DeepLabV3Plus, "segformer": smp.Segformer,
         "upernet": smp.UPerNet}[arch]
    net = m(encoder_name=encoder, encoder_weights="imagenet",
            in_channels=3, classes=1)

    class Wrap(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            return self.net(x)[:, 0]
    return Wrap(net)


class SemsegUNet(nn.Module):
    """torchvision ResNet-50 (ImageNet) エンコーダ + U-Net デコーダ。出力 stride 2 → x2。"""

    def __init__(self):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet50(weights="IMAGENET1K_V2")
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)        # /2, 64
        self.pool = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.up4 = ConvBlock(2048 + 1024, 256)
        self.up3 = ConvBlock(256 + 512, 128)
        self.up2 = ConvBlock(128 + 256, 64)
        self.up1 = ConvBlock(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        s = self.stem(x)                       # /2
        c1 = self.l1(self.pool(s))             # /4
        c2 = self.l2(c1)                       # /8
        c3 = self.l3(c2)                       # /16
        c4 = self.l4(c3)                       # /32
        u = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[-2:], mode="bilinear",
                                                  align_corners=False), b], 1)
        d = self.up4(u(c4, c3))
        d = self.up3(u(d, c2))
        d = self.up2(u(d, c1))
        d = self.up1(u(d, s))                  # /2
        return F.interpolate(self.head(d), scale_factor=2, mode="bilinear",
                             align_corners=False)[:, 0]


def dice_bce(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((1, 2))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2)) + target.sum((1, 2)) + 1)
    return dice.mean() + 0.5 * F.binary_cross_entropy_with_logits(logit, target)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="work_dirs/semseg_f0")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--arch", default="baseline",
                    choices=["baseline", "unet", "unetpp", "manet", "fpn",
                             "deeplabv3p", "segformer", "upernet"])
    ap.add_argument("--encoder", default="resnet50")
    ap.add_argument("--target", choices=["union", "majority"], default="union",
                    help="教師の作り方。majority は過半数が引いた画素のみ")
    args = ap.parse_args()

    torch.cuda.set_per_process_memory_fraction(0.85)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tr_tab, va_tab = build_union_rles(args.fold, args.target)
    print(f"[data] train {len(tr_tab)} 枚 / val {len(va_tab)} 枚（{args.target} 教師）")

    tr = DataLoader(SemsegDataset(tr_tab, True), batch_size=args.batch, shuffle=True,
                    num_workers=4, pin_memory=True, drop_last=True)
    va_stems = sorted(va_tab)[:24]
    va = DataLoader(SemsegDataset({s: va_tab[s] for s in va_stems}, False),
                    batch_size=args.batch, shuffle=False, num_workers=2)

    model = (SemsegUNet() if args.arch == "baseline"
             else build_arch(args.arch, args.encoder)).cuda()
    print(f"[model] arch={args.arch} encoder={args.encoder} "
          f"params={sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = len(tr) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    scaler = torch.amp.GradScaler("cuda")
    best = 0.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train(); tl = []
        for x, t in tr:
            x, t = x.cuda(non_blocking=True), t.cuda(non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = dice_bce(model(x), t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tl.append(loss.item())
        model.eval(); inter = union = 0
        with torch.inference_mode():
            for x, t in va:
                with torch.amp.autocast("cuda"):
                    p = torch.sigmoid(model(x.cuda())) > 0.5
                t = t.cuda() > 0.5
                inter += (p & t).sum().item(); union += (p | t).sum().item()
        iou = inter / max(union, 1)
        el = time.time() - t0
        mark = ""
        if iou > best:
            best = iou
            torch.save({"arch": args.arch, "encoder": args.encoder,
                        "state": model.state_dict()}, out / "semseg_best.pth")
            mark = "  <-- best保存"
        print(f"[ep {ep:2d}/{args.epochs}] loss={np.mean(tl):.4f} val-IoU={iou:.4f} "
              f"({el/60:.0f}分 残り{el/ep*(args.epochs-ep)/60:.0f}分){mark}", flush=True)
    print(f"[done] best val-IoU={best:.4f} -> {out}/semseg_best.pth")


if __name__ == "__main__":
    main()
