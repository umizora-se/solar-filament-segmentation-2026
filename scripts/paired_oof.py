"""2つの OOF 予測を、後処理を固定して fold 別の差と画像ブートストラップで比較する（DECISION_RULES ルール1）。

例: python3 -u scripts/paired_oof.py work_dirs/oof_votes_a0.0.json work_dirs/oof_swaA.json
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import load_folds  # noqa: E402
from tune_postproc import NPIX, H, W, iou_sparse, load_gt_idx, paint_disjoint_sparse, rle_to_idx  # noqa: E402

THRS = [0.30, 0.35, 0.40]
AREA = 200


def per_image_stats(preds, stems, gt):
    used = np.zeros(NPIX, bool)
    out = {t: np.zeros((len(stems), 4)) for t in THRS}
    for n, st in enumerate(stems):
        items = preds[st]
        idxs = [rle_to_idx({"size": [H, W], "counts": it["rle"].encode()}) for it in items]
        scores = np.asarray([it["score"] for it in items], np.float32)
        for t in THRS:
            kept = paint_disjoint_sparse(idxs, scores, t, AREA, used)
            s = out[t][n]
            for gts in gt.get(st, [[]]):
                if not gts:
                    s[2] += len(kept)
                elif not kept:
                    s[3] += len(gts)
                else:
                    iou = iou_sparse(kept, gts)
                    hit = iou > 0.5
                    s += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return out


def pq(a):
    a = a.sum(0)
    return a[0] / (a[1] + 0.5 * a[2] + 0.5 * a[3])


def main():
    base_p, new_p = sys.argv[1], sys.argv[2]
    base = json.loads(Path(base_p).read_text())["preds"]
    new = json.loads(Path(new_p).read_text())["preds"]
    stems = sorted(base)
    assert set(new) == set(stems), "画像の集合が違う"
    gt = load_gt_idx(set(stems))
    sb, sn = per_image_stats(base, stems, gt), per_image_stats(new, stems, gt)
    fold_of = load_folds()
    folds = np.array([fold_of[s] for s in stems])
    boot = np.random.default_rng(0).integers(0, len(stems), (5000, len(stems)))
    print(f"base={Path(base_p).name}  new={Path(new_p).name}  ({len(stems)}枚, area={AREA})")
    for t in THRS:
        b, c = sb[t], sn[t]
        per = [pq(c[folds == k]) - pq(b[folds == k]) for k in range(5)]
        ds = np.array([pq(c[i]) - pq(b[i]) for i in boot])
        print(f"thr={t:.2f} base={pq(b):.4f} new={pq(c):.4f} d={pq(c) - pq(b):+.4f} "
              f"folds={' '.join(f'{x:+.4f}' for x in per)} pos={sum(x > 0 for x in per)}/5 "
              f"95%CI=[{np.percentile(ds, 2.5):+.4f},{np.percentile(ds, 97.5):+.4f}] "
              f"P(d>0)={np.mean(ds > 0):.3f}", flush=True)


if __name__ == "__main__":
    main()
