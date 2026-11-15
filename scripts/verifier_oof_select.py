"""検証器メンバーを OOF 707枚で選び直す。

142枚での貪欲前進選択は val 過適合を起こし、LB を 0.40→0.39 に下げた
（docs/DECISION_RULES.md）。同じ選択を 707枚でやり直す。

手順:
  1. 有望な3アーキ（fold0 単体AUROCで現メンバーを上回った sernx/segf_b2/eff_b3）を
     fold1-4 で学習 → 計6アーキが全fold揃う
  2. 予測×モデルの semprob 行列を一度だけ作る（線形性により組み合わせ評価が即座に済む）
  3. OOF 707枚の AUROC で貪欲前進選択
  4. 選ばれた組で OOF-PQ を測り、0.4228 に対し +0.005 以上なら test 側を作って提出
"""
import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "veroofsel_summary.txt"
OOF = WD / "oof_raw_tta8.json"
TEST8 = WD / "r50_f0" / "test_raw_tta8.json"
BASE = 0.4228                     # 現行3種・sqrt・OOF707枚
NEW = [("sernx", "unet", "se_resnext50_32x4d"),
       ("segf_b2", "segformer", "mit_b2"),
       ("eff_b3", "unet", "efficientnet-b3")]
HAVE = ["baseline", "deeplabv3p", "segformer"]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=8 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def fold0_dir(tag):
    return {"baseline": WD / "semseg_f0", "deeplabv3p": WD / "semseg_deeplabv3p",
            "segformer": WD / "semseg_segformer", "sernx": WD / "semseg_sernx",
            "segf_b2": WD / "semseg_segf_b2", "eff_b3": WD / "semseg_eff_b3"}[tag]


def auroc(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg]); r = np.empty(len(a)); o = np.argsort(a); sa = a[o]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 検証器メンバーのOOF再選択 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: 現行3種(sqrt) OOF-PQ {BASE}。判定は +0.005 以上")

    # ---- 1. 新3アーキを fold1-4 で学習 ----
    for tag, arch, enc in NEW:
        for k in [1, 2, 3, 4]:
            out = WD / f"semseg_f{k}_{tag}"
            if (out / "semseg_best.pth").exists():
                continue
            ok = False
            for bs in [4, 2, 1]:
                log(f"学習: fold{k}/{tag} (batch={bs})")
                if run(["python3", "-u", "src/train_semseg.py", "--fold", str(k),
                        "--arch", arch, "--encoder", enc, "--out", str(out),
                        "--epochs", "20", "--batch", str(bs)],
                       f"vs_train_f{k}_{tag}.log"):
                    ok = True
                    break
                if "OutOfMemory" not in (WD / f"vs_train_f{k}_{tag}.log").read_text(
                        errors="ignore"):
                    break
            if not ok:
                log(f"★ fold{k}/{tag} 学習失敗")

    # ---- 推論（全6アーキ × 全5fold）----
    spec = {}
    for tag in HAVE + [t for t, _, _ in NEW]:
        spec[tag] = {}
        for k in range(5):
            d = fold0_dir(tag) if k == 0 else WD / f"semseg_f{k}_{tag}"
            ck, pv = d / "semseg_best.pth", d / "probs_val.npz"
            if not ck.exists():
                continue
            if not pv.exists():
                log(f"推論: fold{k}/{tag}")
                if not run(["python3", "-u", "src/semseg_infer.py", "--weights", str(ck),
                            "--fold", str(k), "--split", "val", "--out", str(pv)],
                           f"vs_inf_f{k}_{tag}.log"):
                    continue
            spec[tag][str(k)] = str(pv)
    full = [t for t in spec if len(spec[t]) == 5]
    log(f"全fold揃ったアーキ: {full}")
    if len(full) < 4:
        log("★ 揃ったアーキが少なすぎる。終了")
        return
    spec = {t: spec[t] for t in full}
    sp = WD / "veroofsel_spec.json"
    sp.write_text(json.dumps(spec, indent=2))

    # ---- 2. semprob 行列 ----
    mat = WD / "semprob_matrix.npz"
    if not mat.exists():
        log("semprob 行列を作成（予測 × モデル）")
        if not run(["python3", "-u", "src/semprob_matrix.py", "--preds", str(OOF),
                    "--spec", str(sp), "--out", str(mat)], "vs_matrix.log"):
            return
    d = np.load(mat, allow_pickle=True)
    X, y, names = d["X"], d["y"], [str(s) for s in d["model_names"]]
    log(f"行列: {X.shape[0]} 予測 × {len(names)} モデル  "
        f"TP={int((y==1).sum())} 完全FP={int((y==0).sum())}")

    # ---- 3. OOF 707枚での貪欲前進選択 ----
    singles = {n: auroc(X[:, i], y) for i, n in enumerate(names)}
    log("=== 単体AUROC（OOF 707枚）===")
    for n in sorted(singles, key=lambda k: -singles[k]):
        log(f"  {n:11s} {singles[n]:.4f}")
    cur = [max(singles, key=singles.get)]
    best = singles[cur[0]]
    while len(cur) < len(names):
        cand, cau = None, best
        for n in names:
            if n in cur:
                continue
            idx = [names.index(t) for t in cur + [n]]
            a = auroc(X[:, idx].mean(1), y)
            if a > cau:
                cand, cau = n, a
        if not cand:
            break
        cur.append(cand); best = cau
        log(f"  追加 {cand} → {len(cur)}種 AUROC={best:.4f}")
    base_idx = [names.index(t) for t in HAVE if t in names]
    log(f"選択: {cur} AUROC={best:.4f}  (現行3種 {auroc(X[:, base_idx].mean(1), y):.4f})")

    # ---- 4. OOF-PQ で確認 ----
    ens = WD / "probs_oof_sel.npz"
    parts = {k: [spec[t][k] for t in cur] for k in map(str, range(5))}
    code = f'''
import numpy as np
out = {{}}
for k, srcs in {parts!r}.items():
    ds = [np.load(s) for s in srcs]
    keys = set(ds[0].files)
    for dd in ds[1:]:
        keys &= set(dd.files)
    for key in keys:
        out[key] = np.mean([dd[key].astype(np.float32) for dd in ds], 0).astype(np.uint8)
np.savez_compressed({str(ens)!r}, **out)
print(len(out))
'''
    Path("/tmp/vs_ens.py").write_text(code)
    if not run(["python3", "/tmp/vs_ens.py"], "vs_ens.log"):
        return
    j = WD / "oof_raw_tta8_sel.json"
    if not run(["python3", "src/apply_verifier.py", "--preds", str(OOF),
                "--probs", str(ens), "--mode", "sqrt", "--out", str(j)], "vs_ver.log"):
        return
    o = WD / "postproc_oof_sel.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j), "--fold", "-1",
                "--out", str(o), "--score-thr", "0.25,0.30,0.35,0.40",
                "--min-area", "100,200,300"], "vs_tune.log"):
        return
    dd = json.loads(o.read_text())
    pq = dd["val_pq"]
    log(f"=== 結論 ===")
    log(f"  現行3種      OOF-PQ {BASE:.4f}")
    log(f"  選択{len(cur)}種  OOF-PQ {pq:.4f}  Δ={pq-BASE:+.4f}  "
        f"(thr={dd['score_thr']} area={dd['min_area']})")
    if pq < BASE + 0.005:
        log("  判定: +0.005 に届かず → 提出しない（現行維持）")
        return

    # ---- 5. test 側を作って提出 ----
    log("★ 基準を満たした → test 確率マップを作成（5fold × 選択アーキ）")
    tparts = []
    for t in cur:
        for k in range(5):
            d2 = fold0_dir(t) if k == 0 else WD / f"semseg_f{k}_{t}"
            pt = d2 / "probs_test.npz"
            if not pt.exists():
                if not run(["python3", "-u", "src/semseg_infer.py", "--weights",
                            str(d2 / "semseg_best.pth"), "--split", "test",
                            "--out", str(pt)], f"vs_tinf_f{k}_{t}.log"):
                    continue
            tparts.append(str(pt))
    tens = WD / "probs_test_sel.npz"
    code = f'''
import numpy as np
ds = [np.load(s) for s in {tparts!r}]
keys = set(ds[0].files)
for dd in ds[1:]:
    keys &= set(dd.files)
np.savez_compressed({str(tens)!r}, **{{k: np.mean([dd[k].astype(np.float32) for dd in ds],0).astype(np.uint8) for k in sorted(keys)}})
print(len(keys))
'''
    Path("/tmp/vs_tens.py").write_text(code)
    if not run(["python3", "/tmp/vs_tens.py"], "vs_tens.log"):
        return
    jt = WD / "test_raw_tta8_sel.json"
    if not run(["python3", "src/apply_verifier.py", "--preds", str(TEST8),
                "--probs", str(tens), "--mode", "sqrt", "--out", str(jt)], "vs_vertest.log"):
        return
    sub = REPO / "submission_oofsel.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", str(o), "--out", str(sub)], "vs_csv.log"):
        return
    if "PASS" not in (WD / "vs_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = f"MaskDINO 8-view + OOF-selected verifier ({'+'.join(cur)}, 5-fold) OOF-PQ {pq:.4f}"
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg[:480]], "vs_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 検証器メンバーのOOF再選択 終了 ====")
