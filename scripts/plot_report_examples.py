"""Report figure: three OOF segmentations (single-annotator photos at the 80th/50th/20th per-image PQ percentile).

    python3 scripts/plot_report_examples.py   ->  docs/report/fig_examples.png
Predictions are the final OOF (SWA + seed 2, alpha 0.75, thr 0.32, area 200, retain fraction 0.8); the photos were not seen in training.
"""
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pycocotools.mask as mu

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TRAIN_IMG_DIR  # noqa: E402
from postproc import paint_disjoint  # noqa: E402
from pq_metric import iou_matrix, load_gt_rle_by_stem, rle_to_mask  # noqa: E402

THR, AREA, KEEP = 0.32, 200, 0.8


def kept_masks(items):
    ms = [rle_to_mask(it["rle"]) for it in items]
    sc = np.array([it["score"] for it in items])
    return paint_disjoint(ms, sc, THR, AREA, min_keep_frac=KEEP)


def image_pq(pred, gt_records):
    tp = fp = fn = 0.0
    sio = 0.0
    pr = [mu.encode(np.asfortranarray(m.astype(np.uint8))) for m in pred]
    for rec in gt_records:
        iou = iou_matrix(pr, rec) if pr and rec else np.zeros((len(pr), len(rec)))
        hit = iou > 0.5
        tp += hit.sum(); sio += iou[hit].sum()
        fp += (~hit.any(1)).sum(); fn += (~hit.any(0)).sum()
    return sio / (tp + .5 * fp + .5 * fn) if tp + fp + fn else float("nan")


def main():
    oof = json.loads((REPO / "work_dirs/oof_swa2_a0.75.json").read_text())["preds"]
    gt = load_gt_rle_by_stem(str(REPO / "data/magfilo_singleclass.json"))
    rows = []
    for stem, recs in gt.items():
        if len(recs) == 1 and stem in oof and len(recs[0]) >= 4:
            pred = kept_masks(oof[stem])
            rows.append((image_pq(pred, recs), stem, pred, recs[0]))
    rows.sort(key=lambda r: r[0])
    print(f"{len(rows)} single-annotator photos with >=4 filaments")
    picks = [rows[int(len(rows) * q)] for q in (0.8, 0.5, 0.2)]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.55))
    for ax, (pq, stem, pred, gts), name in zip(axs, picks, ("high", "median", "low")):
        img = cv2.imread(str(TRAIN_IMG_DIR / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        s = 1024 / img.shape[0]
        ax.imshow(cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), cmap="gray")
        for masks, col in (([rle_to_mask(g) for g in gts], "#1fd655"), (pred, "#ff3b30")):
            for m in masks:
                m = cv2.resize(m.astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                for c in cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                    ax.plot(c[:, 0, 0], c[:, 0, 1], color=col, lw=0.9)
        ax.set_title(f"{name}: image PQ {pq:.2f}", fontsize=8, pad=2)
        ax.axis("off"); print(name, stem, round(pq, 3), len(pred), len(gts))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.005, wspace=0.02)
    out = REPO / "docs/report/fig_examples.png"
    fig.savefig(out, dpi=220); print("[write]", out)


if __name__ == "__main__":
    main()
