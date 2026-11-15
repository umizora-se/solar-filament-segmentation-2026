"""検証器の教師を「多数派」に変えて OOF で判定する（方針案 第2優先）。

狙い: PQ が報いるのは多数派の形（検出率は合意度で決まる: 全員一致92.8% / 1人だけ33%）。
      現在の検証器は全アノテーターの和集合を教師にしており、採点基準とずれている。
      教師を過半数一致に変えれば、AUROC→PQ の伝達率（現在1/5）自体が改善しうる。

比較は同一条件で行う: sernx（OOF単体AUROCで最良のアーキ）を union / majority の
2条件で 5fold 学習し、OOF 707枚で AUROC と PQ を測る。
"""
import json
import subprocess
import time
from pathlib import Path

import numpy as np

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "majority_summary.txt"
OOF = WD / "oof_raw_tta8.json"
BASE_DET = 0.4162          # 検出器のみ
BASE_VER = 0.4228          # + 現行3種(union教師) sqrt
ARCH, ENC, TAG = "unet", "se_resnext50_32x4d", "sernx"


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


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 検証器の教師を多数派に {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: 検出器のみ {BASE_DET} / +現行検証器(union) {BASE_VER}")
    log(f"比較アーキ: {TAG} ({ARCH}/{ENC}) を union と majority で各5fold")

    # ---- majority 教師で 5fold 学習 ----
    probs = {}
    for k in range(5):
        out = WD / f"semseg_maj_f{k}_{TAG}"
        ck = out / "semseg_best.pth"
        if not ck.exists():
            ok = False
            for bs in [4, 2, 1]:
                log(f"学習(majority): fold{k} (batch={bs})")
                if run(["python3", "-u", "src/train_semseg.py", "--fold", str(k),
                        "--arch", ARCH, "--encoder", ENC, "--target", "majority",
                        "--out", str(out), "--epochs", "20", "--batch", str(bs)],
                       f"mj_train_f{k}.log"):
                    ok = True
                    break
                if "OutOfMemory" not in (WD / f"mj_train_f{k}.log").read_text(errors="ignore"):
                    break
            if not ok:
                continue
        pv = out / "probs_val.npz"
        if not pv.exists():
            if not run(["python3", "-u", "src/semseg_infer.py", "--weights", str(ck),
                        "--fold", str(k), "--split", "val", "--out", str(pv)],
                       f"mj_inf_f{k}.log"):
                continue
        probs[str(k)] = str(pv)
    log(f"majority 教師の確率マップ: {sorted(probs)}")
    if len(probs) < 5:
        log("★ 5fold 揃わず。終了")
        return

    # ---- AUROC 比較（semprob 行列で高速に）----
    spec = WD / "majority_spec.json"
    union_spec = json.loads((WD / "veroofsel_spec.json").read_text())
    spec.write_text(json.dumps({"maj_sernx": probs,
                                "union_sernx": union_spec[TAG]}, indent=2))
    mat = WD / "semprob_matrix_maj.npz"
    if not run(["python3", "-u", "src/semprob_matrix.py", "--preds", str(OOF),
                "--spec", str(spec), "--out", str(mat)], "mj_matrix.log"):
        return
    d = np.load(mat, allow_pickle=True)
    X, y, names = d["X"], d["y"], [str(s) for s in d["model_names"]]

    def auroc(p):
        pos, neg = p[y == 1], p[y == 0]
        a = np.concatenate([pos, neg]); r = np.empty(len(a)); o = np.argsort(a); sa = a[o]
        i = 0
        while i < len(sa):
            j = i
            while j + 1 < len(sa) and sa[j + 1] == sa[i]:
                j += 1
            r[o[i:j + 1]] = (i + j) / 2.0 + 1
            i = j + 1
        return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    log("=== 単体AUROC（OOF 707枚、同一アーキ・同一fold構成）===")
    aus = {}
    for i, n in enumerate(names):
        aus[n] = auroc(X[:, i])
        log(f"  {n:14s} {aus[n]:.4f}")
    both = auroc(X.mean(1))
    log(f"  union+majority 平均 {both:.4f}")

    # ---- PQ 比較 ----
    for tag, srcs in [("maj", [probs[str(k)] for k in range(5)]),
                      ("mix", None)]:
        if tag == "mix":
            srcs = [probs[str(k)] for k in range(5)] + \
                   [union_spec[TAG][str(k)] for k in range(5)]
        ens = WD / f"probs_oof_{tag}.npz"
        code = f'''
import numpy as np
from collections import defaultdict
acc = defaultdict(list)
for s in {srcs!r}:
    d = np.load(s)
    for k in d.files:
        acc[k].append(d[k].astype(np.float32))
np.savez_compressed({str(ens)!r}, **{{k: np.mean(v,0).astype(np.uint8) for k,v in acc.items()}})
print(len(acc))
'''
        Path("/tmp/mj_ens.py").write_text(code)
        if not run(["python3", "/tmp/mj_ens.py"], f"mj_ens_{tag}.log"):
            continue
        j = WD / f"oof_raw_tta8_{tag}.json"
        if not run(["python3", "src/apply_verifier.py", "--preds", str(OOF),
                    "--probs", str(ens), "--mode", "sqrt", "--out", str(j)],
                   f"mj_ver_{tag}.log"):
            continue
        o = WD / f"postproc_oof_{tag}.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j),
                    "--fold", "-1", "--out", str(o),
                    "--score-thr", "0.25,0.30,0.35", "--min-area", "100,200,300"],
                   f"mj_tune_{tag}.log"):
            continue
        dd = json.loads(o.read_text())
        label = "majority単独5fold" if tag == "maj" else "union+majority 10モデル"
        log(f"  {label}: OOF-PQ {dd['val_pq']:.4f}  "
            f"Δ(vs現行検証器)={dd['val_pq']-BASE_VER:+.4f} "
            f"(thr={dd['score_thr']} area={dd['min_area']})")

    log("=== 判定 ===")
    log(f"  現行検証器(union 3種) OOF-PQ {BASE_VER}")
    log("  上記と比べ +0.005 以上なら採用。提出は別途判断する")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 検証器の教師を多数派に 終了 ====")
