"""SWA 版本命の後処理（票数補正 α・score_thr・min_area）を細かいグリッドで同時に探索し、入れ子 CV で採点する。

現行: α=0.5 / thr=0.35 / area=200（粗いグリッド: thr 0.05 刻み, area 100/200/300 で決定）。
細かくするほど OOF に過適合しやすいので、「4 fold で選び 1 fold で評価」の入れ子 CV で現行と比べる。
現行の固定設定も同じ 707 枚で採点し、画像ブートストラップで差を見る。

例: python3 -u scripts/pp_fine.py --out work_dirs/pp_fine.json
"""
import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import pp_search as P  # noqa: E402  採点関数（画素排他ペイント + 公式 PQ）を共用
from filament_dataset import load_folds  # noqa: E402

WD = REPO / "work_dirs"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
THRS = [round(0.30 + 0.01 * i, 2) for i in range(11)]
AREAS = [100, 150, 200, 250, 300]
CURRENT = (0.5, 0.35, 200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    args = ap.parse_args()
    t0 = time.time()
    raw = {}
    for k in range(5):
        raw.update(json.loads((WD / f"r50_full2048_seed2_f{k}" / "val_raw_prob2m_swa_votes.json").read_text())["preds"])
    stems = sorted(raw)
    fold_of = load_folds()
    folds = np.array([fold_of[s] for s in stems])
    P.G["stems"] = stems
    P.G["gt"] = P.load_gt_idx(set(stems))
    dec = {}
    for st in stems:
        items = raw[st]
        dec[st] = ([P.rle_to_idx({"size": [P.H, P.W], "counts": it["rle"].encode()}) for it in items],
                   np.asarray([it["score"] for it in items], np.float32),
                   np.asarray([it["votes"] / it["nv"] for it in items], np.float32))
    print(f"[load] {len(stems)} 画像 ({time.time() - t0:.0f}s)", flush=True)

    keys, res = [], []
    for a in ALPHAS:
        P.G["img"] = {}
        for st, (idxs, s, v) in dec.items():
            sc = s * v ** a
            o = np.argsort(-sc)
            P.G["img"][st] = ([idxs[i] for i in o], sc[o], np.zeros(len(o), np.float32), np.zeros(len(o), np.float32))
        P.G["cfgs"] = [("F0_global", dict(t=t, a=ar)) for t, ar in itertools.product(THRS, AREAS)]
        with mp.get_context("fork").Pool(args.workers) as pool:
            out = dict(pool.imap_unordered(P.eval_config, range(len(P.G["cfgs"]))))
        for ci, (_, p) in enumerate(P.G["cfgs"]):
            keys.append((a, p["t"], p["a"]))
            res.append(out[ci])
        print(f"  α={a}: {len(P.G['cfgs'])} 通り完了 ({time.time() - t0:.0f}s)", flush=True)
    res = np.stack(res)
    np.savez_compressed(Path(args.out).with_suffix(".npz"), res=res, keys=np.array(keys), folds=folds)  # 再解析用
    cur = keys.index(CURRENT)

    held = np.zeros((len(stems), 4))
    picks = []
    for k in range(5):
        tr, te = folds != k, folds == k
        best = max(range(len(keys)), key=lambda i: P.pq(res[i][tr]))
        picks.append(keys[best])
        held[te] = res[best][te]
    full = max(range(len(keys)), key=lambda i: P.pq(res[i]))
    base = res[cur]
    per = [P.pq(held[folds == k]) - P.pq(base[folds == k]) for k in range(5)]
    bi = np.random.default_rng(0).integers(0, len(stems), (2000, len(stems)))
    ds = np.array([P.pq(held[i]) - P.pq(base[i]) for i in bi])
    summ = dict(current=CURRENT, current_pq=P.pq(base), nested_pq=P.pq(held), delta=P.pq(held) - P.pq(base),
                folds=per, ci=[float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))],
                p_pos=float(np.mean(ds > 0)), picks=picks, full_best=keys[full], full_best_pq=P.pq(res[full]))
    print("\n=== 細かいグリッドの入れ子 CV（α × thr × area = %d 通り）===" % len(keys))
    print(f"  現行 α={CURRENT[0]} thr={CURRENT[1]} area={CURRENT[2]}  {P.pq(base):.4f}")
    print(f"  入れ子 CV  {P.pq(held):.4f}  Δ={P.pq(held) - P.pq(base):+.4f}  fold別 "
          f"{' '.join(f'{x:+.4f}' for x in per)}  95%CI [{summ['ci'][0]:+.4f},{summ['ci'][1]:+.4f}] "
          f"P(Δ>0)={summ['p_pos']:.3f}")
    print(f"  fold ごとの選択: {picks}")
    print(f"  全体最良 {keys[full]}  {P.pq(res[full]):.4f}（参考。過適合を含む）")
    Path(args.out).write_text(json.dumps(summ, indent=2, default=float))
    print(f"[write] {args.out} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
