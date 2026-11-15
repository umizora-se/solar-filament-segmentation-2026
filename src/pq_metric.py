"""
公式 Panoptic Quality（LB 指標）のローカル実装。

EDA ノート セル17-18 / azimahmadzadeh/self-evaluation-notebook と同じ定義:
  - 予測1インスタンス集合 vs 各アノテーターレコードを総当たり IoU
  - hit = IoU > 0.5 （1対1マッチではなく「全ペア」）
  - TP = hit したペア全部（IoU を SQ 用に蓄積）
  - FP = どの GT にも hit しなかった予測
  - FN = どの予測にも hit されなかった GT
  - 1枚の写真は「その写真の全アノテーターレコード」に対して採点し、TP/FP/FN をプール
  - PQ = ΣIoU(TP) / (|TP| + 0.5|FP| + 0.5|FN|)

診断用に greedy 1対1マッチ版 PQ も返す。

マスクは全て 2048x2048（あるいは指定 H,W）でラスタライズして比較する。
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

H = W = 2048
IOU_THR = 0.5


# --------------------------------------------------------------------------- #
# ラスタライズ
# --------------------------------------------------------------------------- #
def poly_to_mask(poly_flat, h=H, w=W) -> np.ndarray:
    """COCO ポリゴン(フラット list) -> bool マスク。"""
    if len(poly_flat) < 6:
        return np.zeros((h, w), bool)
    rle = mask_util.frPyObjects([poly_flat], h, w)
    m = mask_util.decode(mask_util.merge(rle))
    return m.astype(bool)


def rle_to_mask(rle, h=H, w=W) -> np.ndarray:
    """{'size','counts'} または counts 文字列 -> bool マスク。"""
    if isinstance(rle, (str, bytes)):
        rle = {"size": [h, w], "counts": rle if isinstance(rle, bytes) else rle.encode()}
    elif isinstance(rle.get("counts"), str):
        rle = {"size": rle["size"], "counts": rle["counts"].encode()}
    return mask_util.decode(rle).astype(bool)


# --------------------------------------------------------------------------- #
# IoU
# --------------------------------------------------------------------------- #
def _to_rle(m):
    """bool マスク or 既に RLE dict -> RLE dict。"""
    if isinstance(m, dict):
        return m
    return mask_util.encode(np.asfortranarray(np.asarray(m).astype(np.uint8)))


def iou_matrix(pred_masks: list, gt_masks: list) -> np.ndarray:
    """(nP, nG) の IoU 行列。

    RLE ベースで計算する。密行列版は float64 の (N, 4.19M) を実体化するため
    20個体で約670MB を消費してホストRAMを圧迫していた。pycocotools の
    mask_util.iou は RLE のまま交差を数えるので桁違いに軽い。
    """
    if len(pred_masks) == 0 or len(gt_masks) == 0:
        return np.zeros((len(pred_masks), len(gt_masks)), np.float64)
    P = [_to_rle(m) for m in pred_masks]
    G = [_to_rle(m) for m in gt_masks]
    return np.asarray(mask_util.iou(P, G, [0] * len(G)), dtype=np.float64)


# --------------------------------------------------------------------------- #
# 集計器
# --------------------------------------------------------------------------- #
class PQAccumulator:
    def __init__(self, iou_thr: float = IOU_THR):
        self.iou_thr = iou_thr
        self.reset()

    def reset(self):
        self.sum_iou_tp = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        # greedy 診断用
        self.g_sum_iou = 0.0
        self.g_tp = 0
        self.g_fp = 0
        self.g_fn = 0
        self.n_pred_total = 0
        self.n_photos = 0

    def add_photo(self, pred_masks: list[np.ndarray], gt_records: list[list[np.ndarray]]):
        """
        pred_masks : そのモデルの予測インスタンス（1集合）
        gt_records : [[gt_mask, ...](annotator A), [gt_mask, ...](annotator B), ...]
        """
        self.n_photos += 1
        self.n_pred_total += len(pred_masks)
        for gts in gt_records:
            iou = iou_matrix(pred_masks, gts)  # (nP, nG)
            hit = iou > self.iou_thr
            # --- official: 全ペア ---
            self.sum_iou_tp += float(iou[hit].sum())
            self.tp += int(hit.sum())
            if iou.size:
                self.fp += int((hit.sum(axis=1) == 0).sum())
                self.fn += int((hit.sum(axis=0) == 0).sum())
            else:
                self.fp += len(pred_masks)
                self.fn += len(gts)
            # --- greedy 1対1 ---
            gt, gp, gs = self._greedy(iou)
            self.g_sum_iou += gs
            self.g_tp += gt
            self.g_fp += (len(pred_masks) - gt)
            self.g_fn += (len(gts) - gt)

    def _greedy(self, iou: np.ndarray):
        if iou.size == 0:
            return 0, 0, 0.0
        order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
        used_p, used_g = set(), set()
        s = 0.0
        tp = 0
        for i, j in order:
            v = iou[i, j]
            if v <= self.iou_thr:
                break
            if i in used_p or j in used_g:
                continue
            used_p.add(i); used_g.add(j); s += float(v); tp += 1
        return tp, 0, s

    def result(self) -> dict:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        pq = self.sum_iou_tp / den if den > 0 else 0.0
        sq = self.sum_iou_tp / self.tp if self.tp > 0 else 0.0
        rq = self.tp / den if den > 0 else 0.0
        g_den = self.g_tp + 0.5 * self.g_fp + 0.5 * self.g_fn
        g_pq = self.g_sum_iou / g_den if g_den > 0 else 0.0
        return {
            "PQ": pq, "SQ": sq, "RQ": rq,
            "TP": self.tp, "FP": self.fp, "FN": self.fn,
            "PQ_greedy": g_pq,
            "greedy_TP": self.g_tp, "greedy_FP": self.g_fp, "greedy_FN": self.g_fn,
            "mean_n_pred": self.n_pred_total / max(self.n_photos, 1),
            "n_photos": self.n_photos,
        }


# --------------------------------------------------------------------------- #
# GT の読み込み（magfilo_singleclass.json -> stem -> [annotatorごとのマスク list]）
# --------------------------------------------------------------------------- #
def load_gt_by_stem(singleclass_json: str, stems: set[str] | None = None,
                    h=H, w=W) -> dict[str, list[list[np.ndarray]]]:
    coco = json.loads(Path(singleclass_json).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    by_record = defaultdict(list)  # image_id(record) -> [mask,...]
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if stems is not None and st not in stems:
            continue
        seg = a.get("segmentation") or []
        if seg and seg[0] and len(seg[0]) >= 6:
            by_record[a["image_id"]].append(poly_to_mask(seg[0], h, w))
    # record -> stem でまとめ直す
    out = defaultdict(list)
    rec2stem = {rid: id2stem[rid] for rid in by_record}
    for rid, masks in by_record.items():
        out[rec2stem[rid]].append(masks)
    return out


def load_gt_rle_by_stem(singleclass_json: str, stems: set | None = None,
                        h=H, w=W) -> dict:
    """load_gt_by_stem の RLE 版。

    密なマスクを返す load_gt_by_stem は 1個体あたり 4.2MB を確保するため、
    val 142枚（約1,600個体）で 6GB 超のホストRAMを食う。
    iou_matrix は RLE dict をそのまま受けられるので、こちらを使う。
    """
    coco = json.loads(Path(singleclass_json).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    by_record = defaultdict(list)
    for a in coco["annotations"]:
        st = id2stem[a["image_id"]]
        if stems is not None and st not in stems:
            continue
        seg = a.get("segmentation") or []
        if seg and seg[0] and len(seg[0]) >= 6:
            by_record[a["image_id"]].append(
                mask_util.merge(mask_util.frPyObjects([seg[0]], h, w)))
    out = defaultdict(list)
    for rid, rles in by_record.items():
        out[id2stem[rid]].append(rles)
    return out


# --------------------------------------------------------------------------- #
# CLI: 自己テスト（GT vs GT で PQ=1.0、A vs B で IAA を実測）
# --------------------------------------------------------------------------- #
def _self_test(singleclass_json: str, limit: int = 40):
    coco = json.loads(Path(singleclass_json).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    stem_recs = defaultdict(list)
    for im in coco["images"]:
        stem_recs[id2stem[im["id"]]].append(im["id"])
    multi = [s for s, r in stem_recs.items() if len(r) >= 2][:limit]

    gt = load_gt_by_stem(singleclass_json, set(multi))

    # (1) GT vs 自分自身
    acc = PQAccumulator()
    for s in multi:
        acc.add_photo(gt[s][0], [gt[s][0]])
    print(f"[self-test] pred==GT(annot0):  PQ={acc.result()['PQ']:.4f}  (期待 1.0000)")

    # (2) A vs B（IAA = 人間の天井）
    acc = PQAccumulator()
    for s in multi:
        recs = gt[s]
        acc.add_photo(recs[0], recs[1:])   # annotator0 を予測とみなし、残りを GT
    r = acc.result()
    print(f"[self-test] IAA (annot0 vs others) over {len(multi)} photos: "
          f"PQ={r['PQ']:.4f}  PQ_greedy={r['PQ_greedy']:.4f}  "
          f"SQ={r['SQ']:.3f}  RQ={r['RQ']:.3f}  TP/FP/FN={r['TP']}/{r['FP']}/{r['FN']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(Path(__file__).resolve().parents[1] / "data/magfilo_singleclass.json"))
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()
    _self_test(args.json, args.limit)
