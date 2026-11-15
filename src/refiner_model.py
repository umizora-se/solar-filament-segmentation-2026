"""
crop refiner: U-Net + 2つのヘッド。

  (a) segmentation head : クロップ内のフィラメント輪郭を精細化する
      -> 惜しい外れ（IoU 0.3-0.5）を 0.5 の上へ押し上げる
  (b) rejection head    : そもそもフィラメントか否かを判定する
      -> 完全FP（IoU<0.1、val で324本）を棄却する

棄却ヘッドを持つのがこの設計の要点。誤差分解で「完全FPは本物より丸く小さいが、
fill/aspect のグリッド探索では分離できない」と実証されたため、
幾何ルールではなく画素の見た目から判断する学習器が必要になった。

エンコーダはローカルにある MaskDINO チェックポイント内の COCO 事前学習 ResNet-50 を流用する
（ネットワークが使えないため torchvision/timm の重みをダウンロードできない）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# --------------------------------------------------------------------------- #
def load_maskdino_backbone(ckpt_path: str) -> dict:
    """MaskDINO チェックポイントから ResNet-50 backbone の重みを取り出し、
    torchvision の resnet50 が受け取れるキー名に変換する。"""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)

    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        n = k[len("backbone."):]
        if isinstance(v, torch.Tensor):
            v = v.clone()
        # detectron2 ResNet -> torchvision ResNet のキー対応
        if n.startswith("stem.conv1.norm."):
            out["bn1." + n.split("stem.conv1.norm.")[1]] = v
        elif n.startswith("stem.conv1.weight"):
            out["conv1.weight"] = v
        elif n.startswith("res"):
            # res2.0.conv1.weight -> layer1.0.conv1.weight
            stage = int(n[3]) - 1                      # res2 -> layer1
            rest = n[5:]                               # "0.conv1.weight"
            rest = rest.replace("shortcut.norm.", "downsample.1.")
            rest = rest.replace("shortcut.", "downsample.0.")
            rest = rest.replace("conv1.norm.", "bn1.")
            rest = rest.replace("conv2.norm.", "bn2.")
            rest = rest.replace("conv3.norm.", "bn3.")
            out[f"layer{stage}.{rest}"] = v
    return out


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.b(x)


class CropRefiner(nn.Module):
    """入力 3ch = [gray, CLAHE(gray), seed] -> マスク + 実在確率。"""

    def __init__(self, maskdino_ckpt: str | None = None, freeze_stem: bool = True):
        super().__init__()
        enc = torchvision.models.resnet50(weights=None)
        if maskdino_ckpt:
            sd = load_maskdino_backbone(maskdino_ckpt)
            missing, unexpected = enc.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(unexpected)
            print(f"[refiner] MaskDINO backbone から {loaded}/{len(sd)} テンソルをロード "
                  f"(未使用 {len(unexpected)}, 未充足 {len(missing)})")

        # 入力を 3ch のまま使う（[gray, CLAHE, seed]）。conv1 は ImageNet の 3ch 重みをそのまま活かす
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)     # /2, 64ch
        self.pool = enc.maxpool                                      # /4
        self.layer1, self.layer2 = enc.layer1, enc.layer2            # /4 256ch, /8 512ch
        self.layer3, self.layer4 = enc.layer3, enc.layer4            # /16 1024ch, /32 2048ch
        if freeze_stem:
            for p in self.stem.parameters():
                p.requires_grad = False

        self.dec4 = ConvBlock(2048 + 1024, 512)
        self.dec3 = ConvBlock(512 + 512, 256)
        self.dec2 = ConvBlock(256 + 256, 128)
        self.dec1 = ConvBlock(128 + 64, 64)
        self.head_seg = nn.Conv2d(64, 1, 1)

        # 棄却ヘッド: 最深特徴を大域プールして 2値判定
        self.head_cls = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    @staticmethod
    def _up_cat(x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        s0 = self.stem(x)            # /2   64
        p = self.pool(s0)            # /4
        s1 = self.layer1(p)          # /4  256
        s2 = self.layer2(s1)         # /8  512
        s3 = self.layer3(s2)         # /16 1024
        s4 = self.layer4(s3)         # /32 2048

        d = self.dec4(self._up_cat(s4, s3))
        d = self.dec3(self._up_cat(d, s2))
        d = self.dec2(self._up_cat(d, s1))
        d = self.dec1(self._up_cat(d, s0))
        logit = self.head_seg(F.interpolate(d, scale_factor=2, mode="bilinear",
                                            align_corners=False))
        return logit.squeeze(1), self.head_cls(s4).squeeze(1)


# --------------------------------------------------------------------------- #
def dice_bce_loss(logit, target, eps=1.0):
    """細長い対象は正例画素が 1-3% しかないので Dice を主、BCE を従にする。"""
    bce = F.binary_cross_entropy_with_logits(logit, target)
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum((1, 2)) + eps
    den = p.sum((1, 2)) + target.sum((1, 2)) + eps
    dice = 1 - (num / den).mean()
    return dice + 0.5 * bce, dice.item(), bce.item()
