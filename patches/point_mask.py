"""PATCHED (filament baseline): マスク損失を「点だけ計算」で求める仕組み。

背景:
  デコーダーの各層（10層 + 補助出力 + DN）が全クエリ分のマスクを最終解像度で出力し、
  損失計算まで保持する。マスクを stride2（原寸2048で 1024^2）にするとメモリに収まらない
  （実測 OOM、クエリ50でも不可）。
  一方、マッチングと損失は各マスクから点を抜き出して（点サンプリング）計算しており、
  マスク全面は使っていない。

方法:
  マスク = クエリ埋め込み · マスク特徴（チャンネル方向の内積）、点の抜き出し = 双線形補間。
  どちらも線形なので
      point_sample(einsum(e, F), p) == e · point_sample(F, p)
  が厳密に成り立つ。学習時はマスクの代わりにクエリ埋め込み (B, Q, C) を受け渡し、
  点が必要になった所でマスク特徴から点を抜き出して内積を取る。

  FILAMENT_POINT_MASK=1 かつ学習時のみ有効。推論（eval）は従来通り全面マスクを作る。
"""
from __future__ import annotations

import os

import torch
from detectron2.projects.point_rend.point_features import point_sample

_CTX: dict = {"mask_features": None}


def enabled(training: bool) -> bool:
    return training and os.environ.get("FILAMENT_POINT_MASK") == "1"


def set_mask_features(feat: torch.Tensor) -> None:
    _CTX["mask_features"] = feat


def mask_features() -> torch.Tensor:
    f = _CTX["mask_features"]
    if f is None:
        raise RuntimeError("point_mask: マスク特徴が設定されていない（デコーダーの forward 前に呼ばれた）")
    return f


def logits_at(embed: torch.Tensor, b: int, coords: torch.Tensor) -> torch.Tensor:
    """embed (n, C) の各クエリについて、画像 b の座標 coords (P, 2)（[0,1] 正規化）での
    マスク logit (n, P) を返す。point_sample(全面マスク, coords) と厳密に一致する。"""
    feats = point_sample(mask_features()[b:b + 1].float(), coords[None].float(),
                         align_corners=False)[0]          # (C, P)
    return embed.float() @ feats
