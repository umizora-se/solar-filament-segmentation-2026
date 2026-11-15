"""OOF（約0.47）と public LB（0.41）の差の原因を、学習データの正解と test の画像・予測だけで検証する。
test の正解（公開版 MAGFiLO に含まれる）は規約で使用禁止のため、一切使わない。

採点は公式セルフ評価ノートと同じ（注釈者×画像ごとに全ペア IoU>0.5、全体でプール）。
  1. public 部分集合のぶれ: OOF の写真から public 相当の枚数を無作為に抜き、PQ の分布を見る
  2. 注釈者の人数（1/2/3人）別の OOF PQ と、train/test の比較はできない（test の人数は不明）ので train 側のみ
  3. 観測年・観測所（GONG サイト）の分布: train と test の割合、年・サイト別 OOF PQ、
     OOF を test の年・サイト構成で重み付けし直した PQ
  4. 予測の性質: 1画像あたりの予測数・面積・スコアを OOF と test で比較（年・サイト別）

例: python3 -u scripts/lb_gap.py --oof work_dirs/oof_swa2_a0.75.json \
        --test work_dirs/test_ens5x2p_swa_votes_a0.75_mv2.json --thr 0.34 --area 200
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import TEST_IMG_DIR  # noqa: E402
from tune_postproc import H, W, NPIX, iou_sparse, load_gt_idx, paint_disjoint_sparse, rle_to_idx  # noqa: E402


def pq(a):
    a = np.asarray(a).sum(0)
    d = a[1] + 0.5 * a[2] + 0.5 * a[3]
    return a[0] / d if d else 0.0


def kept_masks(items, thr, area, used):
    idxs = [rle_to_idx({"size": [H, W], "counts": it["rle"].encode()}) for it in items]
    s = np.asarray([it["score"] for it in items], np.float32)
    return paint_disjoint_sparse(idxs, s, thr, area, used)


def site(st):
    return st[-2]


def year(st):
    return st[:4]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--thr", type=float, default=0.34)
    ap.add_argument("--area", type=int, default=200)
    a = ap.parse_args()
    oof = json.loads(Path(a.oof).read_text())["preds"]
    tst = json.loads(Path(a.test).read_text())["preds"]
    stems = sorted(oof)
    gt = load_gt_idx(set(stems))
    used = np.zeros(NPIX, bool)

    # 写真ごとの (sum_iou, tp, fp, fn)、注釈者数、予測の性質
    st_stats, n_ann, feat = {}, {}, {}
    for st in stems:
        kept = kept_masks(oof[st], a.thr, a.area, used)
        acc = np.zeros(4)
        recs = gt.get(st, [[]])
        for gts in recs:
            if not gts:
                acc[2] += len(kept)
            elif not kept:
                acc[3] += len(gts)
            else:
                iou = iou_sparse(kept, gts)
                hit = iou > 0.5
                acc += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
        st_stats[st], n_ann[st] = acc, len(recs)
        feat[st] = (len(kept), np.mean([len(k) for k in kept]) if kept else 0.0,
                    np.mean([len(g) for g in recs]))
    tfeat = {}
    for st, items in tst.items():
        kept = kept_masks(items, a.thr, a.area, used)
        tfeat[st] = (len(kept), np.mean([len(k) for k in kept]) if kept else 0.0)
    test_stems = sorted(p.stem for p in TEST_IMG_DIR.glob("*.jpeg"))
    assert set(tfeat) <= set(test_stems)

    allv = np.stack([st_stats[s] for s in stems])
    print(f"OOF 全体 PQ {pq(allv):.4f}  （{len(stems)}写真, 注釈者レコード {sum(n_ann.values())}）\n")

    # ---- 1. public 部分集合のぶれ ----
    rng = np.random.default_rng(0)
    print("=== 1. public 部分集合のぶれ（OOF から無作為抽出 2万回）===")
    for frac in (0.2, 0.3, 0.5):
        m = round(180 * frac)
        sims = np.array([pq(allv[rng.choice(len(stems), m, replace=False)]) for _ in range(20000)])
        print(f"  test の {int(frac * 100)}%（{m}枚）: 平均 {sims.mean():.4f}  標準偏差 {sims.std():.4f}  "
              f"5%点 {np.percentile(sims, 5):.4f}  1%点 {np.percentile(sims, 1):.4f}  "
              f"P(PQ<0.415) {np.mean(sims < 0.415):.4f}")

    # ---- 2. 注釈者の人数別 ----
    print("\n=== 2. 注釈者の人数別 OOF PQ ===")
    for k in (1, 2, 3):
        g = [s for s in stems if n_ann[s] == k]
        v = np.stack([st_stats[s] for s in g])
        print(f"  {k}人: {len(g):3d}写真  PQ {pq(v):.4f}  (SQ {v[:, 0].sum() / v[:, 1].sum():.3f}  "
              f"RQ {v[:, 1].sum() / (v[:, 1].sum() + .5 * v[:, 2].sum() + .5 * v[:, 3].sum()):.3f})")

    # ---- 3. 年・サイト ----
    for name, key in (("観測所（GONG サイト）", site), ("観測年", year)):
        tr = collections.Counter(key(s) for s in stems)
        te = collections.Counter(key(s) for s in test_stems)
        print(f"\n=== 3. {name}: train/test の割合と OOF PQ ===")
        rows = []
        for g in sorted(set(tr) | set(te)):
            gs = [s for s in stems if key(s) == g]
            p = pq(np.stack([st_stats[s] for s in gs])) if gs else float("nan")
            rows.append((g, tr[g] / len(stems), te[g] / len(test_stems), p, len(gs)))
            print(f"  {g}: train {tr[g] / len(stems):5.1%} ({tr[g]:3d})  test {te[g] / len(test_stems):5.1%} "
                  f"({te[g]:3d})  OOF PQ {p:.4f}")
        # test の構成で重み付け（写真単位の統計をグループごとに test 比率/ train 比率 で重み付け）
        w = np.array([(te[key(s)] / len(test_stems)) / (tr[key(s)] / len(stems)) for s in stems])
        print(f"  → OOF を test の{name}構成で重み付け: PQ {pq(allv * w[:, None]):.4f}（元 {pq(allv):.4f}）")

    # ---- 4. 予測の性質 ----
    print("\n=== 4. 予測の性質（後処理後）OOF vs test ===")
    o = np.array([feat[s][:2] for s in stems])
    t = np.array([tfeat[s] for s in test_stems if s in tfeat])
    print(f"  1画像あたり予測数  OOF {o[:, 0].mean():.2f}  test {t[:, 0].mean():.2f}")
    print(f"  予測の平均面積     OOF {o[o[:, 0] > 0, 1].mean():.0f}px  test {t[t[:, 0] > 0, 1].mean():.0f}px")
    print(f"  予測ゼロの画像     OOF {np.mean(o[:, 0] == 0):.1%}  test {np.mean(t[:, 0] == 0):.1%}")
    print(f"  参考: OOF の正解の個数（注釈者平均）{np.mean([feat[s][2] for s in stems]):.2f}/画像")
    for g in sorted(set(site(s) for s in test_stems)):
        oo = [feat[s][0] for s in stems if site(s) == g]
        tt = [tfeat[s][0] for s in test_stems if site(s) == g and s in tfeat]
        print(f"  サイト {g}: 予測数 OOF {np.mean(oo) if oo else float('nan'):.2f} ({len(oo)})  "
              f"test {np.mean(tt) if tt else float('nan'):.2f} ({len(tt)})")


if __name__ == "__main__":
    main()
