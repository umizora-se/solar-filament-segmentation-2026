"""
テスト画像を推論して submission.csv を作る。

フォーマットは data/SUBMISSION_FORMAT.md（公式セルフ評価ノートから確定）:
  列: filament_id, segmentation_rle
  filament_id = {stem}_{k}   (k は 1 始まり)
  segmentation_rle = COCO 圧縮RLE の counts 文字列（size 2048x2048 は暗黙）
  予測ゼロの画像は行を出さない

例:
  # test 180枚 -> submission.csv
  python src/predict_submit.py --config configs/maskdino_filament_r50_1024crop.yaml \
      --weights work_dirs/r50_f0/model_final.pth --out submission.csv

  # 後処理チューニング結果を適用
  python src/predict_submit.py ... --postproc work_dirs/r50_f0/postproc.json

  # val で生予測を保存（tune_postproc.py 用）
  python src/predict_submit.py ... --split val --fold 0 --dump-raw work_dirs/r50_f0/val_raw.json

  # 提出前の検証だけ
  python src/predict_submit.py --verify submission.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "external" / "MaskDINO"))

import pycocotools.mask as mask_util  # noqa: E402
from detectron2.checkpoint import DetectionCheckpointer  # noqa: E402
from detectron2.data import detection_utils as d2_utils  # noqa: E402

from filament_config import setup_cfg  # noqa: E402
from filament_dataset import register_filament, get_test_dicts, get_val_photo_dicts  # noqa: E402
from postproc import paint_disjoint  # noqa: E402

H = W = 2048


# --------------------------------------------------------------------------- #
# RLE
# --------------------------------------------------------------------------- #
def mask_to_rle(mask: np.ndarray) -> str:
    """bool マスク -> COCO 圧縮RLE の counts 文字列（採点側の decode と対になる形式）。"""
    r = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return r["counts"].decode("utf-8")


def rle_to_mask(counts: str) -> np.ndarray:
    return mask_util.decode({"size": [H, W], "counts": counts.encode("utf-8")}).astype(bool)


# --------------------------------------------------------------------------- #
# 推論
# --------------------------------------------------------------------------- #
def load_model(cfg, weights: str):
    import train_net as maskdino_train_net
    model = maskdino_train_net.Trainer.build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    model.eval()
    return model


def read_image(path: str, fmt: str = "RGB") -> torch.Tensor:
    img = d2_utils.read_image(path, format=fmt)          # (H, W, 3) uint8
    return torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))


@torch.no_grad()
def infer_full(model, path: str, size: int = H):
    """フル画像1発推論。size=2048 なら学習時の val 評価と同じ土俵。"""
    img = read_image(path)
    if size != H:
        img = F.interpolate(img[None].float(), size=(size, size),
                            mode="bilinear", align_corners=False)[0].to(torch.uint8)
    with torch.amp.autocast("cuda"):
        out = model([{"image": img, "height": size, "width": size}])
    inst = out[0]["instances"]
    return inst.pred_masks, inst.scores          # GPU 上、(N, size, size)


@torch.no_grad()
def infer_tiled(model, path: str, tile: int = 1024, overlap: int = 256, merge_iou: float = 0.5):
    """タイル推論（学習クロップと解像度・トークン数を揃えたい場合）。

    タイルを重ねて切り、各タイルの予測を 2048 キャンバスへ戻し、
    タイル跨ぎで IoU>merge_iou の予測を統合する。
    """
    img = read_image(path)
    step = tile - overlap
    starts = sorted(set(list(range(0, H - tile + 1, step)) + [H - tile]))

    masks, scores = [], []
    for y0 in starts:
        for x0 in starts:
            crop = img[:, y0:y0 + tile, x0:x0 + tile]
            with torch.amp.autocast("cuda"):
                out = model([{"image": crop, "height": tile, "width": tile}])
            inst = out[0]["instances"]
            if len(inst) == 0:
                continue
            m = (inst.pred_masks > 0.5).cpu().numpy()
            s = inst.scores.cpu().numpy()
            for mi, si in zip(m, s):
                full = np.zeros((H, W), bool)
                full[y0:y0 + tile, x0:x0 + tile] = mi
                masks.append(full)
                scores.append(float(si))

    if not masks:
        return np.zeros((0, H, W), bool), np.zeros(0, np.float32)

    # --- タイル跨ぎの重複を統合（スコア降順に貪欲マージ）---
    order = np.argsort(-np.asarray(scores))
    rles = [mask_util.encode(np.asfortranarray(masks[i].astype(np.uint8))) for i in order]
    keep, kept_rles = [], []
    for rank, i in enumerate(order):
        if kept_rles:
            ious = mask_util.iou([rles[rank]], kept_rles, [0] * len(kept_rles))[0]
            if ious.max() > merge_iou:
                continue          # 既に採用した予測と同一物体 -> 捨てる
        keep.append(i)
        kept_rles.append(rles[rank])
    return np.stack([masks[i] for i in keep]), np.asarray([scores[i] for i in keep], np.float32)



# --------------------------------------------------------------------------- #
# Flip TTA（上位解法が共通採用。Fususu の topology-safe flip-4 を参考に実装）
# --------------------------------------------------------------------------- #
# 太陽全面画像は円板なので D4（90°回転×4 × 左右反転×2 = 8通り）が全て妥当な変換。
# (k_rot, hflip): x を 90°*k 回転してから必要なら左右反転する。
TF_D4 = [(k, hf) for k in range(4) for hf in (False, True)]      # 8方向
TF_FLIP4 = [(0, False), (0, True), (2, False), (2, True)]        # 従来の4方向
# (k=2,hf=False) は 180°回転 = 上下左右反転、(k=2,hf=True) は上下反転に等しい


def _apply_tf(x, k, hf):
    """x: (C,H,W) または (N,H,W)。空間次元は末尾2つ。"""
    if k:
        x = torch.rot90(x, k, dims=(-2, -1))
    if hf:
        x = torch.flip(x, dims=[-1])
    return x


def _invert_tf(x, k, hf):
    """_apply_tf の逆変換（順序も逆にする）。"""
    if hf:
        x = torch.flip(x, dims=[-1])
    if k:
        x = torch.rot90(x, -k, dims=(-2, -1))
    return x


@torch.no_grad()
def infer_flip_tta(model, path: str, size: int = H,
                   collect_thr: float = 0.15, cluster_iou: float = 0.5,
                   min_views: int = 2, topology_guard: bool = True,
                   n_views: int = 4, mask_bin_thr: float = 0.5,
                   merge: str = "vote", prob_margin: int = 48, return_votes: bool = False):
    """D4対称群によるTTA。model にモデルのリストを渡すとモデル間でも同じ統合を行う。

    各ビューで推論 -> マスクを元向きに戻す -> ビュー横断でIoUクラスタリング
    -> 票数 min_views 以上のクラスタだけ採用 -> クラスタ内マスクを統合。

    merge="vote": 画素ごとの多数決（半数以上のビューが立てた画素を採用）。
        実測の問題: フィラメントの端は細く淡いため方向間で一致せず、系統的に削られる。
        長さ比が TTA無し 0.754 -> 8方向TTA 0.664 に短縮することを OOF で確認した。
    merge="prob": 各ビューの確率を平均してから一度だけ2値化する。
        弱い証拠も累積されるため端が残る。モデル側は既に確率を返すようにパッチ済み。
        中間値が生まれるので mask_bin_thr のチューニングが初めて意味を持つ。
        確率は bbox + prob_margin の範囲だけ保持する（マスクは画像の0.03%しかないため）。

    複数モデル: 「モデル × 向き」を1つのビューとして扱い、確率もモデル横断で平均する。
        予測マスクを 0/1 にしてから束ねる src/ensemble.py と違い、弱い証拠を捨てない。
        min_views はモデル数倍する（1モデル8向きで2票 = 2モデル16ビューで4票、と割合を揃える）。

    return_votes: 各出力に賛成したビュー数と総ビュー数も返す（masks, scores, votes, n_views）。
        スコアは賛成ビューのスコア平均なので、2ビューだけが見つけた個体と全ビューが
        見つけた個体が同じ値になりうる。票数はその区別に使う独立の手がかり。

    topology_guard: 統合でマスクが分裂した場合（連結成分が増えた場合）は
    元のアンカーマスクに戻す。TTA が形状を壊すのを防ぐ。
    """
    img = read_image(path)
    tfs = TF_D4 if n_views == 8 else TF_FLIP4
    models = model if isinstance(model, (list, tuple)) else [model]
    min_views = min_views * len(models)
    views = [(mdl, k, hf) for mdl in models for k, hf in tfs]
    pool = []            # (mask bool (H,W), score, view_id, prob_patch|None)
    for vid, (mdl, k, hf) in enumerate(views):
        x = _apply_tf(img, k, hf)
        with torch.amp.autocast("cuda"):
            out = mdl([{"image": x, "height": size, "width": size}])
        inst = out[0]["instances"]
        if len(inst) == 0:
            continue
        sc = inst.scores
        keep = sc >= collect_thr
        if keep.sum() == 0:
            continue
        prob = _invert_tf(inst.pred_masks[keep], k, hf)   # 元の向きに戻す（確率のまま）
        # mask_bin_thr < 0.5 にすると確率場に沿ってマスクが育つ（惜しい外れ対策）。
        # 一律膨張と違い、モデルが自信のある方向にだけ広がる。
        m = (prob > mask_bin_thr).cpu().numpy()
        pr = prob.float().cpu().numpy() if merge == "prob" else None
        for idx, (mi, si) in enumerate(zip(m, sc[keep].float().cpu().numpy())):
            # merge="prob" のときは bbox+余白の確率パッチだけ持つ（全面 float は 16MB/枚）
            patch = None
            if pr is not None and mi.any():
                ys, xs = np.nonzero(mi)
                y0 = max(int(ys.min()) - prob_margin, 0); y1 = min(int(ys.max()) + 1 + prob_margin, H)
                x0 = max(int(xs.min()) - prob_margin, 0); x1 = min(int(xs.max()) + 1 + prob_margin, W)
                patch = (y0, y1, x0, x1, pr[idx][y0:y1, x0:x1].astype(np.float16))
            pool.append((mi, float(si), vid, patch))
        del m, prob, pr, inst, out
        torch.cuda.empty_cache()

    def _ret(m, s, v):
        return (m, s, np.asarray(v, np.int32), len(views)) if return_votes else (m, s)

    if not pool:
        return _ret(np.zeros((0, H, W), bool), np.zeros(0, np.float32), [])

    # スコア降順に貪欲クラスタリング（IoU は RLE で軽量に）
    order = sorted(range(len(pool)), key=lambda i: -pool[i][1])
    rles = {i: mask_util.encode(np.asfortranarray(pool[i][0].astype(np.uint8))) for i in order}
    used = set()
    masks_out, scores_out, votes_out = [], [], []
    for i in order:
        if i in used:
            continue
        anchor_mask, anchor_score, anchor_view = pool[i][0], pool[i][1], pool[i][2]
        members = {anchor_view: (i, anchor_score)}
        cand = [j for j in order if j not in used and j != i and pool[j][2] != anchor_view]
        if cand:
            ious = mask_util.iou([rles[i]], [rles[j] for j in cand], [0] * len(cand))[0]
            for j, iou in zip(cand, ious):
                v = pool[j][2]
                if iou > cluster_iou and v not in members:
                    members[v] = (j, pool[j][1])
        used.add(i)
        for j, _ in members.values():
            used.add(j)

        if len(members) < min_views:          # 票が足りない -> 捨てる
            continue

        if merge == "prob":
            # 確率を平均してから一度だけ2値化する（端の弱い証拠を捨てない）
            acc = np.zeros((H, W), np.float32)
            n_contrib = 0
            for j, _ in members.values():
                patch = pool[j][3]
                if patch is None:
                    acc += pool[j][0].astype(np.float32)
                else:
                    y0, y1, x0, x1, pv = patch
                    acc[y0:y1, x0:x1] += pv.astype(np.float32)
                n_contrib += 1
            merged = (acc / max(n_contrib, 1)) > mask_bin_thr
        else:
            # 画素ごとの多数決（>=半数のビューが立てた画素を採用）
            acc = np.zeros((H, W), np.uint8)
            for j, _ in members.values():
                acc += pool[j][0]
            merged = acc >= (len(members) + 1) // 2

        if topology_guard and merged.any():
            # 平均化で分裂したらアンカーに戻す（TTA が形状を壊すのを防ぐ）
            import cv2
            n_a = cv2.connectedComponents(anchor_mask.astype(np.uint8), 8)[0]
            n_m = cv2.connectedComponents(merged.astype(np.uint8), 8)[0]
            area_ratio = merged.sum() / max(anchor_mask.sum(), 1)
            if n_m > n_a or not (0.5 < area_ratio < 2.0):
                merged = anchor_mask
        if not merged.any():
            continue
        masks_out.append(merged)
        scores_out.append(float(np.mean([sc for _, sc in members.values()])))
        votes_out.append(len(members))

    if not masks_out:
        return _ret(np.zeros((0, H, W), bool), np.zeros(0, np.float32), [])
    return _ret(np.stack(masks_out), np.asarray(scores_out, np.float32), votes_out)


def to_numpy_masks(pred_masks: torch.Tensor, scores: torch.Tensor,
                   score_thr: float, mask_thr: float = 0.5):
    """GPU 上の予測を、しきい値で絞ってから CPU の bool 配列にする（メモリ節約）。"""
    if len(scores) == 0:
        return np.zeros((0, H, W), bool), np.zeros(0, np.float32)
    keep = scores >= score_thr
    if keep.sum() == 0:
        return np.zeros((0, H, W), bool), np.zeros(0, np.float32)
    m = pred_masks[keep]
    s = scores[keep].float().cpu().numpy()
    if m.shape[-2:] != (H, W):
        m = F.interpolate(m[:, None].float(), size=(H, W),
                          mode="bilinear", align_corners=False)[:, 0]
    m = (m > mask_thr).cpu().numpy()
    return m, s


# --------------------------------------------------------------------------- #
# 検証
# --------------------------------------------------------------------------- #
def verify_submission(csv_path: str, expect_stems: set[str] | None = None) -> bool:
    """提出前チェック: 列名 / ID 一意性 / RLE デコード可能性 / 画素の重なり0。"""
    import collections
    ok = True
    rows = list(csv.DictReader(open(csv_path)))
    print(f"[verify] {csv_path}: {len(rows)} 行")

    if not rows:
        print("  ★ 行が0件")
        return False
    cols = list(rows[0].keys())
    if cols != ["filament_id", "segmentation_rle"]:
        print(f"  ★ 列名が不正: {cols} (期待: ['filament_id','segmentation_rle'])"); ok = False
    else:
        print("  ✓ 列名 OK")

    ids = [r["filament_id"] for r in rows]
    dup = [k for k, v in collections.Counter(ids).items() if v > 1]
    if dup:
        print(f"  ★ filament_id が重複: {dup[:5]}"); ok = False
    else:
        print("  ✓ filament_id は全行ユニーク")

    by_img = collections.defaultdict(list)
    for r in rows:
        by_img[r["filament_id"].split("_", 1)[0]].append(r["segmentation_rle"])

    bad_overlap, bad_decode, total_px = [], [], 0
    for stem, rles in by_img.items():
        canvas = np.zeros((H, W), np.int32)
        for rle in rles:
            try:
                m = rle_to_mask(rle)
            except Exception as e:
                bad_decode.append((stem, str(e)[:60])); continue
            canvas += m
            total_px += int(m.sum())
        if canvas.max() > 1:
            bad_overlap.append((stem, int(canvas.max()), int((canvas > 1).sum())))

    if bad_decode:
        print(f"  ★ RLE デコード失敗 {len(bad_decode)} 件: {bad_decode[:3]}"); ok = False
    else:
        print("  ✓ 全 RLE がデコード可能")

    if bad_overlap:
        print(f"  ★ 画素の重なりあり {len(bad_overlap)} 画像: {bad_overlap[:3]}"); ok = False
    else:
        print("  ✓ 画素の重なり 0")

    print(f"  画像数={len(by_img)}  平均予測数={len(rows)/max(len(by_img),1):.1f}  "
          f"平均マスク面積={total_px/max(len(rows),1):.0f}px")
    if expect_stems is not None:
        missing = expect_stems - set(by_img)
        print(f"  予測ゼロの画像: {len(missing)} 枚"
              f"{'（フォーマット上は問題なし）' if missing else ''}")
    print(f"[verify] {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "configs/maskdino_filament_r50_1024crop.yaml"))
    ap.add_argument("--weights", nargs="+",
                    help="複数渡すとモデル間で確率平均して融合する（--tta 必須）")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--split", choices=["test", "val"], default="test")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--infer-size", type=int, default=H,
                    help="フル推論の入力サイズ。2048=ネイティブ(既定)")
    ap.add_argument("--tta", action="store_true", help="4方向フリップTTAを使う")
    ap.add_argument("--tta-views", type=int, default=4, choices=[4, 8],
                    help="TTAの方向数。4=フリップのみ / 8=D4(90度回転を含む)")
    ap.add_argument("--tta-min-views", type=int, default=2, help="採用に必要な票数")
    ap.add_argument("--tta-merge", choices=["vote", "prob"], default="vote",
                    help="ビュー統合の方式。prob は確率平均（端の侵食を防ぐ）")
    ap.add_argument("--mask-bin-thr", type=float, default=0.5,
                    help="TTA各ビューのマスク2値化閾値（下げると確率場に沿ってマスクが育つ）")
    ap.add_argument("--tta-collect-thr", type=float, default=0.15,
                    help="各ビューで候補として集めるスコア下限")
    ap.add_argument("--tile", type=int, default=0,
                    help=">0 でタイル推論（例 1024）。学習クロップと揃えたい場合")
    ap.add_argument("--tile-overlap", type=int, default=256)
    ap.add_argument("--score-thr", type=float, default=None)
    ap.add_argument("--min-area", type=int, default=None)
    ap.add_argument("--mask-thr", type=float, default=None)
    ap.add_argument("--postproc", help="tune_postproc.py が出した postproc.json")
    ap.add_argument("--dump-raw", help="後処理前の予測(RLE+score)を保存するパス")
    ap.add_argument("--dump-score-thr", type=float, default=0.05,
                    help="dump 時の足切り。チューニング余地を残すため低め")
    ap.add_argument("--limit", type=int, default=0, help="先頭N枚だけ（動作確認用）")
    ap.add_argument("--stems-file", default=None,
                    help="対象 stem の JSON 配列。train 画像で推論したいとき用")
    ap.add_argument("--image-dir", default=None,
                    help="このフォルダの *.jpeg 全部を推論する（外部フレームへの疑似ラベル付け用）")
    ap.add_argument("--verify", help="CSV を検証するだけ（推論しない）")
    args = ap.parse_args()

    if args.verify:
        sys.exit(0 if verify_submission(args.verify) else 1)

    assert args.weights, "--weights が必要です"

    cfg = setup_cfg(args.config, [], output_dir=str(REPO / "work_dirs" / "predict"))
    # VRAM 安全弁（推論でも必ず適用）
    from train_filament import apply_vram_guards
    apply_vram_guards(cfg)

    score_thr = args.score_thr if args.score_thr is not None else cfg.FILAMENT.SCORE_THR
    min_area = args.min_area if args.min_area is not None else cfg.FILAMENT.MIN_AREA
    mask_thr = args.mask_thr if args.mask_thr is not None else cfg.FILAMENT.MASK_THR
    if args.postproc:
        pp = json.loads(Path(args.postproc).read_text())
        score_thr = pp.get("score_thr", score_thr)
        min_area = pp.get("min_area", min_area)
        mask_thr = pp.get("mask_thr", mask_thr)
        print(f"[postproc] {args.postproc} を適用")
    print(f"[postproc] score_thr={score_thr}  min_area={min_area}  mask_thr={mask_thr}")

    register_filament(args.fold)
    dicts = get_test_dicts() if args.split == "test" else get_val_photo_dicts(args.fold)
    if args.stems_file:
        import json as _j
        want = set(_j.loads(Path(args.stems_file).read_text()))
        from filament_dataset import TRAIN_IMG_DIR as _TD
        dicts = [{"file_name": str(_TD / f"{s}.jpeg"), "image_id": s, "stem": s,
                  "height": 2048, "width": 2048, "annotations": []} for s in sorted(want)]
    if args.image_dir:
        dicts = [{"file_name": str(p), "image_id": p.stem, "stem": p.stem,
                  "height": 2048, "width": 2048, "annotations": []}
                 for p in sorted(Path(args.image_dir).glob("*.jpeg"))]
    if args.limit:
        dicts = dicts[: args.limit]
    print(f"[data] split={args.split} 枚数={len(dicts)}")

    assert len(args.weights) == 1 or args.tta, "複数モデルの融合は --tta が必要です"
    models = [load_model(cfg, w).cuda() for w in args.weights]
    model = models if len(models) > 1 else models[0]
    print(f"[model] {len(models)} モデル: {', '.join(Path(w).parent.name for w in args.weights)}")

    rows, raw = [], {}
    t0 = time.time()
    for i, d in enumerate(dicts, 1):
        stem = d["stem"]
        votes, nv = None, None
        if args.tta:
            masks, scores, votes, nv = infer_flip_tta(
                model, d["file_name"], args.infer_size,
                collect_thr=args.tta_collect_thr, min_views=args.tta_min_views,
                n_views=args.tta_views, mask_bin_thr=args.mask_bin_thr,
                merge=args.tta_merge, return_votes=True)
            sel = scores >= (args.dump_score_thr if args.dump_raw else score_thr)
            masks, scores, votes = masks[sel], scores[sel], votes[sel]
        elif args.tile:
            masks, scores = infer_tiled(model, d["file_name"], args.tile, args.tile_overlap)
            sel = scores >= (args.dump_score_thr if args.dump_raw else score_thr)
            masks, scores = masks[sel], scores[sel]
        else:
            pm, sc = infer_full(model, d["file_name"], args.infer_size)
            masks, scores = to_numpy_masks(
                pm, sc, args.dump_score_thr if args.dump_raw else score_thr, mask_thr)
            del pm, sc
            torch.cuda.empty_cache()

        if args.dump_raw:
            raw[stem] = [{"rle": mask_to_rle(m), "score": float(s)}
                         for m, s in zip(masks, scores)]
            if votes is not None:              # TTA の票数（賛成ビュー数 / 総ビュー数）
                for it, v in zip(raw[stem], votes):
                    it["votes"], it["nv"] = int(v), int(nv)

        kept = paint_disjoint(list(masks), scores, score_thr, min_area, mask_thr)
        for k, m in enumerate(kept, start=1):          # ★ k は 1 始まり
            rows.append((f"{stem}_{k}", mask_to_rle(m)))

        if i <= 3 or i % 20 == 0 or i == len(dicts):
            el = time.time() - t0
            print(f"  [{i:3d}/{len(dicts)}] {stem}: pred={len(masks)} -> kept={len(kept)}  "
                  f"({el:.0f}s, {el/i:.2f}s/枚, 残り約{el/i*(len(dicts)-i)/60:.1f}分)",
                  flush=True)   # リダイレクト時もリアルタイムに見えるように

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        w.writerows(rows)
    print(f"[write] {args.out}: {len(rows)} 行 / {len(dicts)} 画像")

    if args.dump_raw:
        Path(args.dump_raw).write_text(json.dumps({"preds": raw}))
        print(f"[write] {args.dump_raw}（後処理前、score>={args.dump_score_thr}）")

    print()
    ok = verify_submission(args.out, expect_stems={d["stem"] for d in dicts})
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
