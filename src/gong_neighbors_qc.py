"""近傍 GONG フレームの品質管理。元画像と比べて、ラベルを写してよいかを判定する。

判定:
  ・位置ずれ（位相相関、円板内のみ）が max_shift 画素以下
      → 1画素未満なら補正なしでラベルを写せる（実測 0.14〜0.16px）
  ・円板内の画像相関が min_corr 以上
      → 雲・極端なシーイング悪化で像が崩れたフレームを落とす（実測 0.85〜0.89）

出力: datasets/gong_neighbors/qc.csv（neighbor, shift_px, corr, pass）
filament_dataset.get_train_dicts_with_neighbors はこれが存在すれば pass のみ使う。

    python src/gong_neighbors_qc.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TRAIN_IMG_DIR, NEIGHBOR_DIR  # noqa: E402
from preprocess import fit_disk  # noqa: E402


def disk_mask(img: np.ndarray, frac: float = 0.95) -> np.ndarray:
    c, r = fit_disk(img)
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    return np.hypot(xx - c[0], yy - c[1]) < r * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-shift", type=float, default=1.0)
    ap.add_argument("--min-corr", type=float, default=0.6)
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(NEIGHBOR_DIR / "manifest.csv"))
            if r.get("status") == "ok" and not r.get("excluded")]
    print(f"[qc] 対象 {len(rows)} 枚  基準: ずれ≦{args.max_shift}px, 相関≧{args.min_corr}")
    cache: dict[str, tuple] = {}
    out = []
    for i, r in enumerate(sorted(rows, key=lambda x: x["source"]), 1):
        src = r["source"]
        if src not in cache:
            cache.clear()
            s_img = cv2.imread(str(Path(TRAIN_IMG_DIR) / f"{src}.jpeg"), cv2.IMREAD_GRAYSCALE)
            m = disk_mask(s_img)
            cache[src] = (s_img, m, np.float32(s_img) * m)
        s_img, m, s_f = cache[src]
        n_img = cv2.imread(str(NEIGHBOR_DIR / "jpeg" / f"{r['neighbor']}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if n_img is None or n_img.shape != s_img.shape:
            out.append({"neighbor": r["neighbor"], "source": src, "shift_px": "", "corr": "",
                        "pass": 0, "reason": "読込不可"})
            continue
        (dx, dy), _ = cv2.phaseCorrelate(s_f, np.float32(n_img) * m)
        shift = float(np.hypot(dx, dy))
        corr = float(np.corrcoef(s_img[m], n_img[m])[0, 1])
        ok = shift <= args.max_shift and corr >= args.min_corr
        reason = "" if ok else ("ずれ大" if shift > args.max_shift else "相関低（雲など）")
        out.append({"neighbor": r["neighbor"], "source": src, "shift_px": f"{shift:.3f}",
                    "corr": f"{corr:.4f}", "pass": int(ok), "reason": reason})
        if i % 200 == 0:
            print(f"  [{i}/{len(rows)}]", flush=True)
    with open(NEIGHBOR_DIR / "qc.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["neighbor", "source", "shift_px", "corr", "pass", "reason"])
        w.writeheader(); w.writerows(out)
    sh = np.array([float(o["shift_px"]) for o in out if o["shift_px"]])
    cr = np.array([float(o["corr"]) for o in out if o["corr"]])
    n_ok = sum(o["pass"] for o in out)
    print(f"[qc] 合格 {n_ok} / {len(out)}")
    if len(sh):
        print(f"  ずれ(px)  p50={np.median(sh):.3f} p95={np.percentile(sh,95):.3f} 最大={sh.max():.2f}")
        print(f"  相関      p05={np.percentile(cr,5):.3f} p50={np.median(cr):.3f}")
    from collections import Counter
    print(f"  不合格の理由: {dict(Counter(o['reason'] for o in out if not o['pass']))}")


if __name__ == "__main__":
    main()
