"""予測 × 検証器モデル の semprob 行列を一度だけ作る（貪欲選択の高速化）。

鍵となる性質: マスク内平均は線形なので

    mean_pixels( mean_models(p_m) ) = mean_models( mean_pixels(p_m) )

つまり「アンサンブルの semprob」は「各モデルの semprob の平均」で正確に求まる。
よって (予測 × モデル) の行列を一度作れば、どの組み合わせの評価も numpy 演算だけで済む。
素朴に組み合わせごとに確率マップを平均して測ると 707枚 × 20通り = 12時間かかる。

出力: npz {X: (n_pred, n_model) float32, y: (n_pred,) int8, model_names, stems, idx}
       y は 1=TP(IoU>0.5) / 0=完全FP(IoU<0.1) / -1=中間
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, load_folds  # noqa: E402

H = W = 2048


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="OOF 生予測 JSON")
    ap.add_argument("--spec", required=True,
                    help='JSON: {"arch名": {"0": probs_f0.npz, "1": ...}}')
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.preds).read_text())
    preds = raw["preds"] if "preds" in raw else raw
    spec = json.loads(Path(args.spec).read_text())
    names = sorted(spec)
    fold_of = load_folds()

    # GT（全fold）
    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    id2stem = {im["id"]: Path(im["file_name"]).stem for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        seg = a.get("segmentation") or []
        if seg and len(seg[0]) >= 6:
            gt[id2stem[a["image_id"]]].append(
                mask_util.merge(mask_util.frPyObjects([seg[0]], H, W)))

    # fold ごとに確率マップを開く（メモリ節約のため fold 単位で処理）
    stems_by_fold = defaultdict(list)
    for s in preds:
        stems_by_fold[fold_of.get(s, -1)].append(s)

    rows_X, rows_y, rows_stem, rows_i = [], [], [], []
    for k in sorted(stems_by_fold):
        if str(k) not in {kk for n in names for kk in spec[n]}:
            print(f"  fold{k}: 確率マップが無い。スキップ")
            continue
        opened = {}
        for n in names:
            p = spec[n].get(str(k))
            if p and Path(p).exists():
                opened[n] = np.load(p)
        if len(opened) != len(names):
            print(f"  fold{k}: {len(opened)}/{len(names)} モデルのみ。スキップ")
            continue
        print(f"  fold{k}: {len(stems_by_fold[k])} 画像 × {len(names)} モデル", flush=True)
        for stem in sorted(stems_by_fold[k]):
            items = preds[stem]
            if not items:
                continue
            rles = [{"size": [H, W], "counts": it["rle"].encode()} for it in items]
            g = gt.get(stem, [])
            best = (mask_util.iou(rles, g, [0] * len(g)).max(1) if g
                    else np.zeros(len(rles)))
            masks = [mask_util.decode(r).astype(bool) for r in rles]
            pmaps = {n: opened[n][stem].astype(np.float32) / 255.0
                     for n in names if stem in opened[n].files}
            if len(pmaps) != len(names):
                continue
            for i, (m, b) in enumerate(zip(masks, best)):
                if not m.any():
                    continue
                rows_X.append([float(pmaps[n][m].mean()) for n in names])
                rows_y.append(1 if b > 0.5 else (0 if b < 0.1 else -1))
                rows_stem.append(stem)
                rows_i.append(i)

    X = np.asarray(rows_X, np.float32)
    y = np.asarray(rows_y, np.int8)
    print(f"[matrix] {X.shape[0]} 予測 × {X.shape[1]} モデル  "
          f"TP={int((y==1).sum())} 完全FP={int((y==0).sum())} 中間={int((y==-1).sum())}")
    np.savez_compressed(args.out, X=X, y=y, model_names=np.array(names),
                        stems=np.array(rows_stem), idx=np.array(rows_i, np.int32))
    print(f"[write] {args.out}")


if __name__ == "__main__":
    main()
