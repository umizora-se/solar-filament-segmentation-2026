"""cfg 構築ヘルパ。MaskDINO / deeplab の config に FILAMENT 名前空間を足す。"""
from __future__ import annotations

import sys
from pathlib import Path

from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "external" / "MaskDINO"))
from maskdino import add_maskdino_config  # noqa: E402


def add_filament_config(cfg):
    # 勾配累積（VRAM を増やさず実効バッチを稼ぐ）
    cfg.SOLVER.ACCUM_STEPS = 1
    cfg.FILAMENT = CN()
    # VRAM 安全弁: WSL2 は GPU メモリ溢れを例外にせずホストRAMへページングし
    # システム全体をフリーズさせる。上限を切って clean な CUDA OOM にする。
    cfg.FILAMENT.CUDA_MEM_FRACTION = 0.85
    # deformable attention エンコーダの勾配チェックポイント（活性化メモリを大幅削減）
    cfg.INPUT.NATIVE_CROP = False   # True で原寸クロップ（縮小しない）
    cfg.FILAMENT.USE_NEIGHBORS = False   # True で近傍GONGフレームへのラベル転写データを学習に加える
    cfg.FILAMENT.USE_PSEUDO = False      # True で外部 GONG フレームの疑似ラベルを学習に加える
    cfg.FILAMENT.GRAD_CHECKPOINT = True
    cfg.FILAMENT.PRETRAINED_KEY = "r50_3s"   # train_filament.PRETRAINED のキー
    cfg.FILAMENT.SCORE_THR = 0.25      # 予測スコア足切り
    cfg.FILAMENT.MIN_AREA = 150        # 最小マスク面積 (2048 基準 px)
    cfg.FILAMENT.MASK_THR = 0.5        # マスク二値化しきい値（sigmoid 後）
    cfg.FILAMENT.DUMP_PRED = False     # val 推論の生予測を JSON 保存
    cfg.FILAMENT.EVAL_MAX_DET = 100    # PQ 評価時に使う最大インスタンス数
    # 学習終了時の自動評価。detectron2 は EVAL_PERIOD=0 でも最終評価を必ず走らせる
    # （val 142枚をフル2048推論で約11分）。スモーク時は False にする。
    cfg.FILAMENT.EVAL_AFTER_TRAIN = True
    # 学習中の評価で使う val 画像数の上限（0=全部）。決定的にサンプリングする。
    # 最終的な数値は --eval-only + FILAMENT.EVAL_MAX_IMAGES 0 で取り直すこと。
    cfg.FILAMENT.EVAL_MAX_IMAGES = 0
    # マスク損失を Dice から Focal-Tversky に切り替える（FNを重く罰する）
    cfg.FILAMENT.TVERSKY_LOSS = False
    # マスクヘッドを stride4->stride2 に高解像度化（惜しい外れをTP化）
    cfg.FILAMENT.HIRES_MASK = False
    cfg.FILAMENT.HIRES_SKIP = False   # True で stem の stride2 特徴を skip 接続（HIRES_MASK 前提）
    cfg.FILAMENT.POINT_MASK = False   # True で学習時のマスク損失・マッチングを点だけで計算（全面マスクを作らない）
    # True で分類ターゲットを「一致した予測のマスク IoU」にする（Stable-DINO / Align-DETR 式の IoU 考慮型）
    cfg.FILAMENT.IOU_CLS = False


def setup_cfg(config_file: str, opts=None, output_dir: str | None = None):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskdino_config(cfg)
    add_filament_config(cfg)
    cfg.merge_from_file(config_file)
    if opts:
        cfg.merge_from_list(opts)
    if output_dir:
        cfg.OUTPUT_DIR = output_dir
    cfg.freeze()
    return cfg
