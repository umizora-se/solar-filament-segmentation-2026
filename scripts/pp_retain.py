"""Retain-fraction rule search (thr x area x keep), nested 5-fold CV on OOF-707. Usage: python3 scripts/pp_retain.py <workers> <out.npz>"""
import itertools, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "scripts"))
import pp_search as P
from filament_dataset import load_folds
from tune_postproc import NPIX, iou_sparse
WD = REPO / "work_dirs"; ALPHA = 0.75
THRS = [0.32, 0.33, 0.34, 0.35, 0.36]; AREAS = [150, 200, 250]
FRACS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
CUR = (0.34, 200, 0.0)
CFG = list(itertools.product(THRS, AREAS, FRACS))

def ev(ci):
    t, a, f = CFG[ci]
    used = np.zeros(NPIX, bool); res = np.zeros((len(G["stems"]), 4))
    for n, st in enumerate(G["stems"]):
        idxs, s = G["img"][st]
        kept = []
        for i in range(len(s)):
            if s[i] < t: break
            free = idxs[i][~used[idxs[i]]]
            if free.size < a or free.size < f * idxs[i].size: continue
            used[free] = True; kept.append(free)
        for k_ in kept: used[k_] = False
        acc = res[n]
        for gts in G["gt"].get(st, [[]]):
            if not gts: acc[2] += len(kept)
            elif not kept: acc[3] += len(gts)
            else:
                iou = iou_sparse(kept, gts); hit = iou > 0.5
                acc += [iou[hit].sum(), hit.sum(), (hit.sum(0) == 0).sum(), (hit.sum(1) == 0).sum()]
    return ci, res

G = {}
if __name__ == "__main__":
    t0 = time.time(); raw = {}
    for k in range(5):
        raw.update(json.loads((WD / f"r50_full2048_seed2_f{k}" / "val_raw_prob2m_swa_votes.json").read_text())["preds"])
    stems = sorted(raw); fo = load_folds(); folds = np.array([fo[s] for s in stems])
    G["stems"] = stems; G["gt"] = P.load_gt_idx(set(stems)); G["img"] = {}
    for st in stems:
        it = raw[st]
        idxs = [P.rle_to_idx({"size": [P.H, P.W], "counts": x["rle"].encode()}) for x in it]
        sc = np.array([x["score"] for x in it], np.float32) * np.array([x["votes"] / x["nv"] for x in it], np.float32) ** ALPHA
        o = np.argsort(-sc); G["img"][st] = ([idxs[i] for i in o], sc[o])
    print(f"[load] {time.time()-t0:.0f}s", flush=True)
    with mp.get_context("fork").Pool(int(sys.argv[1])) as pool:
        out = dict(pool.imap_unordered(ev, range(len(CFG))))
    res = np.stack([out[i] for i in range(len(CFG))]); cur = CFG.index(CUR); base = res[cur]
    print(f"current {CUR} PQ={P.pq(base):.4f}")
    for f in FRACS:
        b = max((i for i, c in enumerate(CFG) if c[2] == f), key=lambda i: P.pq(res[i]))
        r = res[b].sum(0); print(f"  f={f}: best {CFG[b]} PQ={P.pq(res[b]):.4f}  TP={r[1]:.0f} FP={r[2]:.0f} FN={r[3]:.0f}")
    held = np.zeros_like(base); picks = []
    for k in range(5):
        tr = folds != k; b = max(range(len(CFG)), key=lambda i: P.pq(res[i][tr])); picks.append(CFG[b]); held[folds == k] = res[b][folds == k]
    per = [P.pq(held[folds == k]) - P.pq(base[folds == k]) for k in range(5)]
    bi = np.random.default_rng(0).integers(0, len(stems), (2000, len(stems)))
    ds = np.array([P.pq(held[i]) - P.pq(base[i]) for i in bi])
    print(f"nested {P.pq(held):.4f} Δ={P.pq(held)-P.pq(base):+.4f} folds {' '.join(f'{x:+.4f}' for x in per)} CI[{np.percentile(ds,2.5):+.4f},{np.percentile(ds,97.5):+.4f}] P>0={np.mean(ds>0):.3f}")
    print("picks", picks, f"({time.time()-t0:.0f}s)")
    np.savez_compressed(Path(sys.argv[2]), res=res, cfg=np.array(CFG), folds=folds)
