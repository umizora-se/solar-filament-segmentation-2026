"""後処理の足切りを「画像・予測の性質で変える」方式を OOF 707枚で探索し、入れ子の交差検証で採点する。

現行: 全画像共通の score_thr / min_area（OOF で thr=0.35 area=200）。
試す方式（いずれも予測ごとに「残す候補か」を決め、その後は現行と同じ画素排他ペイント + min_area）:
  F0 global   : score >= t                                   （現行。比較の基準）
  F1 rel_up   : score >= max(t, rho * 画像内最大スコア)       自信のある画像では厳しく
  F1b rel_down: score >= t  または score >= rho * 画像内最大   自信のない画像でも上位は残す
  F2 size     : 面積 < A0 なら t_small、以上なら t_large      小さい予測ほど厳しく
  F3 limb     : 円盤中心からの距離 r/R >= r0 なら t_out        MAGFiLO はリム付近のフィラメントが少ない
  F4 topk     : score >= t のうち上位 k 個まで

採点: 方式ごとに「4 fold でパラメータを選び、残り 1 fold で評価」を 5 回（入れ子 CV）。
現行（F0）も同じ手順で採点し直して比較する。最後に最良方式と F0 を画像ブートストラップで比べる。

例: python3 -u scripts/pp_search.py --preds work_dirs/oof_votes2_a0.0.json --out work_dirs/pp_search.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TRAIN_IMG_DIR, load_folds  # noqa: E402
from tune_postproc import H, W, NPIX, iou_sparse, load_gt_idx, rle_to_idx  # noqa: E402

T = [0.25, 0.30, 0.35, 0.40, 0.45]
A = [100, 200, 300]

G: dict = {}   # fork で子プロセスと共有するデータ


def disk_geometry(stem: str):
    """円盤の中心 (x, y) と半径 R（2048 基準 px）。512 に縮小して明るい画素の重心と面積から求める。"""
    im = Image.open(TRAIN_IMG_DIR / f"{stem}.jpeg").convert("L").resize((512, 512))
    a = np.asarray(im, np.float32)
    m = a > 0.25 * np.percentile(a, 99)
    ys, xs = np.nonzero(m)
    if len(xs) < 1000:
        return 1024.0, 1024.0, 900.0
    return xs.mean() * 4 + 2, ys.mean() * 4 + 2, np.sqrt(m.sum() / np.pi) * 4


def configs():
    out = []
    for t, a in itertools.product(T, A):
        out.append(("F0_global", dict(t=t, a=a)))
    for t, a, rho in itertools.product(T, A, [0.3, 0.4, 0.5, 0.6, 0.7]):
        out.append(("F1_rel_up", dict(t=t, a=a, rho=rho)))
    for t, a, rho in itertools.product(T, A, [0.7, 0.8, 0.9]):
        out.append(("F1b_rel_down", dict(t=t, a=a, rho=rho)))
    for ts, tl, a0 in itertools.product(T, T, [600, 1200, 2400]):
        out.append(("F2_size", dict(ts=ts, tl=tl, a0=a0, a=200)))
    for ti, to, r0 in itertools.product(T, T + [0.5, 0.6], [0.8, 0.9, 0.95]):
        out.append(("F3_limb", dict(ti=ti, to=to, r0=r0, a=200)))
    for t, k in itertools.product(T, [4, 6, 8, 10, 12]):
        out.append(("F4_topk", dict(t=t, k=k, a=200)))
    return out


def eligible(fam, p, s, area, r):
    """予測ごとの「残す候補か」(bool 配列)。s はスコア降順に並んでいる。"""
    if fam == "F0_global":
        return s >= p["t"]
    if fam == "F1_rel_up":
        return s >= max(p["t"], p["rho"] * (s[0] if len(s) else 0))
    if fam == "F1b_rel_down":
        return (s >= p["t"]) | (s >= p["rho"] * (s[0] if len(s) else 0))
    if fam == "F2_size":
        return s >= np.where(area < p["a0"], p["ts"], p["tl"])
    if fam == "F3_limb":
        return s >= np.where(r >= p["r0"], p["to"], p["ti"])
    if fam == "F4_topk":
        e = s >= p["t"]
        e[np.cumsum(e) > p["k"]] = False
        return e
    raise ValueError(fam)


def eval_config(ci):
    fam, p = G["cfgs"][ci]
    used = np.zeros(NPIX, bool)
    res = np.zeros((len(G["stems"]), 4))
    for n, st in enumerate(G["stems"]):
        idxs, s, area, r = G["img"][st]
        el = eligible(fam, p, s, area, r)
        kept, touched = [], []
        for i in np.nonzero(el)[0]:          # スコア降順に画素排他ペイント（tune_postproc と同じ）
            free = idxs[i][~used[idxs[i]]]
            if free.size < p["a"]:
                continue
            used[free] = True
            touched.append(free)
            kept.append(free)
        for t_ in touched:
            used[t_] = False
        acc = res[n]
        for gts in G["gt"].get(st, [[]]):
            if not gts:
                acc[2] += len(kept)
            elif not kept:
                acc[3] += len(gts)
            else:
                iou = iou_sparse(kept, gts)
                hit = iou > 0.5
                acc += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return ci, res


def pq(a):
    a = a.sum(0)
    d = a[1] + 0.5 * a[2] + 0.5 * a[3]
    return a[0] / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    t0 = time.time()
    raw = json.loads(Path(args.preds).read_text())["preds"]
    fold_of = load_folds()
    stems = sorted(raw)
    G["stems"] = stems
    G["gt"] = load_gt_idx(set(stems))
    G["img"] = {}
    for st in stems:
        items = sorted(raw[st], key=lambda it: -it["score"])
        idxs = [rle_to_idx({"size": [H, W], "counts": it["rle"].encode()}) for it in items]
        s = np.asarray([it["score"] for it in items], np.float32)
        area = np.asarray([len(i) for i in idxs], np.float32)
        cx, cy, R = disk_geometry(st)
        r = np.asarray([np.hypot(i % W - cx, i // W - cy).mean() / R if len(i) else 0 for i in idxs],
                       np.float32)
        G["img"][st] = (idxs, s, area, r)
    G["cfgs"] = configs()
    print(f"[load] {len(stems)} 画像 / 設定 {len(G['cfgs'])} 通り ({time.time() - t0:.0f}s)", flush=True)

    res = [None] * len(G["cfgs"])
    with mp.get_context("fork").Pool(args.workers) as pool:
        for done, (ci, r) in enumerate(pool.imap_unordered(eval_config, range(len(G["cfgs"]))), 1):
            res[ci] = r
            if done % 25 == 0 or done == len(res):
                print(f"  [{done}/{len(res)}] {time.time() - t0:.0f}s", flush=True)
    res = np.stack(res)                                    # (n_cfg, n_img, 4)
    folds = np.array([fold_of[s] for s in stems])
    fams = sorted({f for f, _ in G["cfgs"]})

    def nested(fam):
        ids = [i for i, (f, _) in enumerate(G["cfgs"]) if f == fam]
        held = np.zeros((len(stems), 4))
        picks = []
        for k in range(5):
            tr, te = folds != k, folds == k
            best = max(ids, key=lambda i: pq(res[i][tr]))
            picks.append(best)
            held[te] = res[best][te]
        return held, picks

    summary = {}
    base_held, _ = nested("F0_global")
    print("\n=== 入れ子 CV（4 fold で選び 1 fold で評価）===")
    for fam in fams:
        held, picks = nested(fam)
        full = max((i for i, (f, _) in enumerate(G["cfgs"]) if f == fam), key=lambda i: pq(res[i]))
        per = [pq(held[folds == k]) - pq(base_held[folds == k]) for k in range(5)]
        rng = np.random.default_rng(0)
        bi = rng.integers(0, len(stems), (2000, len(stems)))
        ds = np.array([pq(held[i]) - pq(base_held[i]) for i in bi])
        summary[fam] = dict(nested_pq=pq(held), delta=pq(held) - pq(base_held), folds=per,
                            ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                            full_best=G["cfgs"][full][1], full_best_pq=pq(res[full]),
                            picks=[G["cfgs"][i][1] for i in picks])
        print(f"{fam:13s} 入れ子 {pq(held):.4f}  Δ={pq(held) - pq(base_held):+.4f}  "
              f"fold別 {' '.join(f'{x:+.4f}' for x in per)}  95%CI [{summary[fam]['ci'][0]:+.4f},"
              f"{summary[fam]['ci'][1]:+.4f}]  | 全体最良 {pq(res[full]):.4f} {G['cfgs'][full][1]}", flush=True)
    Path(args.out).write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[write] {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
