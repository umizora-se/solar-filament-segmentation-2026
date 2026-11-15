"""推論後処理: スコア順の画素排他ペイント + 最小面積フィルタ。

提出制約（1画像内でマスクが重なってはいけない）をここで担保する。
学習時の PQ 評価・推論提出・後処理チューニングで共通利用。
"""
from __future__ import annotations

import numpy as np


def paint_disjoint(masks: list[np.ndarray], scores: np.ndarray,
                   score_thr: float, min_area: int,
                   mask_thr: float = 0.5, min_keep_frac: float = 0.0) -> list[np.ndarray]:
    """
    masks  : list of float/bool 2D array（同一 H,W）。float なら mask_thr で二値化。
    scores : 各マスクのスコア（同順）
    min_keep_frac : 排他後に元マスクのこの割合未満しか残らない断片は棄却（0 で従来と同一）
    戻り値 : 画素が互いに排他な bool マスクの list（スコア降順）
    """
    if len(masks) == 0:
        return []
    order = np.argsort(-np.asarray(scores))
    H, Wd = masks[0].shape
    canvas = np.zeros((H, Wd), np.uint8)
    out = []
    for k in order:
        if scores[k] < score_thr:
            continue
        m = masks[k]
        m = (m >= mask_thr) if m.dtype != bool else m
        orig = int(m.sum())
        m = m & (canvas == 0)          # 既に塗られた画素は奪わない
        if int(m.sum()) < min_area or int(m.sum()) < min_keep_frac * orig:
            continue
        canvas[m] = 1
        out.append(m)
    return out
