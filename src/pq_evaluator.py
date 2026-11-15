"""detectron2 の test() から呼ばれる公式PQ Evaluator。

- 予測 Instances を stem ごとに集約
- score_thr / min_area / mask_thr で後処理（画素排他ペイント）
- magfilo_singleclass.json から stem の全アノテーターレコードを GT として引き、PQ をプール計算
- cfg.FILAMENT.DUMP_PRED なら生予測(RLE+score, 後処理前)を OUTPUT_DIR/val_predictions_f{fold}.json に保存
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn.functional as F
from detectron2.evaluation import DatasetEvaluator
from detectron2.utils import comm

from pq_metric import PQAccumulator, load_gt_rle_by_stem
from postproc import paint_disjoint


def _resize_masks(pred_masks: torch.Tensor, h: int, w: int) -> np.ndarray:
    """(N, ph, pw) float/bool -> (N, h, w) float [0,1]."""
    if pred_masks.numel() == 0:
        return np.zeros((0, h, w), np.float32)
    x = pred_masks.float().unsqueeze(1)              # N,1,ph,pw
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x.squeeze(1).clamp(0, 1).cpu().numpy()


class FilamentPQEvaluator(DatasetEvaluator):
    def __init__(self, dataset_name, cfg, output_dir: str | None = None):
        from detectron2.data import MetadataCatalog
        md = MetadataCatalog.get(dataset_name)
        self.singleclass_json = md.singleclass_json
        self.fold = getattr(md, "fold", 0)
        self.score_thr = cfg.FILAMENT.SCORE_THR
        self.min_area = cfg.FILAMENT.MIN_AREA
        self.mask_thr = cfg.FILAMENT.MASK_THR
        self.max_det = cfg.FILAMENT.EVAL_MAX_DET
        self.dump = cfg.FILAMENT.DUMP_PRED
        self.output_dir = output_dir or cfg.OUTPUT_DIR
        self._preds = {}       # stem -> list[RLE dict]   (後処理済み。密マスクは保持しない)
        self._raw = {}         # stem -> [{rle, score}]   (後処理前、DUMP用)

    def reset(self):
        self._preds = {}
        self._raw = {}

    def process(self, inputs, outputs):
        for inp, out in zip(inputs, outputs):
            stem = inp.get("stem") or Path(inp["file_name"]).stem
            h = int(inp.get("height", 2048))
            w = int(inp.get("width", 2048))
            inst = out["instances"].to("cpu")
            if len(inst) == 0:
                self._preds[stem] = []
                self._raw[stem] = []
                continue
            scores = inst.scores.numpy()
            order = np.argsort(-scores)[: self.max_det]
            scores = scores[order]
            masks = _resize_masks(inst.pred_masks[order], h, w)   # (n,h,w) float

            if self.dump:
                self._raw[stem] = [
                    {"rle": _enc(m >= self.mask_thr), "score": float(s)}
                    for m, s in zip(masks, scores)
                ]
            kept = paint_disjoint(list(masks), scores, self.score_thr,
                                  self.min_area, self.mask_thr)
            # 密マスク(2048^2 bool = 4.2MB/個)を全画像ぶん保持するとホストRAMが枯れる。
            # 直ちに RLE 化して捨てる。
            self._preds[stem] = [
                mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in kept
            ]
            del masks, kept

    def evaluate(self):
        # 複数プロセス集約
        if comm.get_world_size() > 1:
            comm.synchronize()
            preds_list = comm.gather(self._preds, dst=0)
            raw_list = comm.gather(self._raw, dst=0)
            if not comm.is_main_process():
                return {}
            self._preds = {k: v for d in preds_list for k, v in d.items()}
            self._raw = {k: v for d in raw_list for k, v in d.items()}

        stems = set(self._preds)
        gt = load_gt_rle_by_stem(self.singleclass_json, stems)

        acc = PQAccumulator()
        for stem in stems:
            acc.add_photo(self._preds.get(stem, []), gt.get(stem, [[]]))
        res = acc.result()

        if self.dump:
            outp = Path(self.output_dir) / f"val_predictions_f{self.fold}.json"
            outp.parent.mkdir(parents=True, exist_ok=True)   # ← 無いと FileNotFoundError
            outp.write_text(json.dumps({
                "score_thr_applied": None, "note": "raw (pre-postproc) masks + scores",
                "preds": self._raw,
            }))
            print(f"[FilamentPQEvaluator] dumped raw val preds -> {outp}")

        from collections import OrderedDict
        pretty = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}
        print(f"[FilamentPQEvaluator] fold {self.fold}  {pretty}")
        return OrderedDict({"filament_pq": {
            "PQ": res["PQ"], "SQ": res["SQ"], "RQ": res["RQ"],
            "PQ_greedy": res["PQ_greedy"], "mean_n_pred": res["mean_n_pred"],
        }})


def _enc(mask_bool: np.ndarray) -> str:
    r = mask_util.encode(np.asfortranarray(mask_bool.astype(np.uint8)))
    return r["counts"].decode()
