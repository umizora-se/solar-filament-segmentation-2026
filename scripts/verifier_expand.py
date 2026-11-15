"""検証器アンサンブルの拡張（施策1）。

現行 3種（全て resnet50 エンコーダ）に、エンコーダを変えた候補を追加し、
AUROC を基準に貪欲前進選択でメンバーを決める。

学んだ条件: 多様性があり、かつ実力が拮抗しているときに票決は効く
（RF-DETR は実力差が大きすぎて失敗した）。
そこでエンコーダだけ変えて実力を保ったまま誤り方を変える。
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
SUMMARY = WD / "verexp_summary.txt"
VAL4 = WD / "r50_f0" / "val_raw_tta.json"
VAL8 = WD / "r50_f0" / "val_raw_tta8.json"
TEST8 = WD / "r50_f0" / "test_raw_tta8.json"
BEST = 0.4322
# (タグ, arch, encoder)  現行3種はエンコーダが全部 resnet50 なので、そこを変える
NEW = [("cnx_s", "unet", "tu-convnext_small"),
       ("eff_b3", "unet", "efficientnet-b3"),
       ("sernx", "unet", "se_resnext50_32x4d"),
       ("dlv3_cnx", "deeplabv3p", "tu-convnext_tiny"),
       ("segf_b2", "segformer", "mit_b2")]
# 既存メンバー（学習済み）
HAVE = {"baseline": WD / "semseg_f0", "deeplabv3p": WD / "semseg_deeplabv3p",
        "segformer": WD / "semseg_segformer", "unetpp": WD / "semseg_unetpp",
        "fpn": WD / "semseg_fpn", "manet": WD / "semseg_manet"}


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=6 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:3])}… ({lf})")
    return r.returncode == 0


def tune(js, tag):
    out = str(Path(js).with_name(f"postproc_{tag}.json"))
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(js), "--fold", "0",
                "--out", out, "--score-thr", "0.20,0.25,0.30,0.35,0.40",
                "--min-area", "100,200,300"], f"ve_tune_{tag}.log"):
        return None, None
    return json.loads(Path(out).read_text())["val_pq"], out


def mean_npz(srcs, dst):
    code = f'''
import numpy as np
ds = [np.load(s) for s in {[str(s) for s in srcs]!r}]
keys = set(ds[0].files)
for d in ds[1:]:
    keys &= set(d.files)
np.savez_compressed({str(dst)!r}, **{{k: np.mean([d[k].astype(np.float32) for d in ds], 0).astype(np.uint8) for k in sorted(keys)}})
'''
    Path("/tmp/ve_mean.py").write_text(code)
    return run(["python3", "/tmp/ve_mean.py"], "ve_mean.log")


def auroc_of(probs, tag):
    aj = WD / f"ve_auroc_{tag}.json"
    if not run(["python3", "-u", "src/eval_verifier.py", "--preds", str(VAL4),
                "--probs", str(probs), "--out", str(aj)], f"ve_auroc_{tag}.log"):
        return None
    return json.loads(aj.read_text())["auroc"]


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 検証器拡張 {time.strftime('%m/%d %H:%M')} ====\n")
    log("現行: baseline+deeplabv3p+segformer(全て resnet50) AUROC=0.8867 / PQ 0.4322")

    pool = {}      # tag -> val確率npz
    for t, d in HAVE.items():
        p = d / "probs_val.npz"
        if p.exists():
            pool[t] = p
    log(f"既存メンバー: {sorted(pool)}")

    # --- 新エンコーダを学習 ---
    for tag, arch, enc in NEW:
        out = WD / f"semseg_{tag}"
        ok = False
        for bs in [4, 2, 1]:
            log(f"学習: {tag} ({arch}/{enc}, batch={bs})")
            if run(["python3", "-u", "src/train_semseg.py", "--fold", "0", "--arch", arch,
                    "--encoder", enc, "--out", str(out), "--epochs", "20",
                    "--batch", str(bs)], f"ve_train_{tag}.log"):
                ok = True
                break
            if "OutOfMemory" not in (WD / f"ve_train_{tag}.log").read_text(errors="ignore"):
                break
        if not ok:
            continue
        probs = out / "probs_val.npz"
        if not run(["python3", "-u", "src/semseg_infer.py", "--weights",
                    str(out / "semseg_best.pth"), "--fold", "0", "--split", "val",
                    "--out", str(probs)], f"ve_inf_{tag}.log"):
            continue
        au = auroc_of(probs, tag)
        if au:
            log(f"  {tag}: AUROC={au:.4f} (単体基準 baseline 0.8722)")
            pool[tag] = probs

    if len(pool) < 3:
        log("候補不足。終了")
        return

    # --- 貪欲前進選択（AUROC 最大化）---
    log("=== 貪欲前進選択 ===")
    singles = {}
    for t, p in pool.items():
        a = auroc_of(p, f"s_{t}")
        if a:
            singles[t] = a
    for t in sorted(singles, key=lambda k: -singles[k]):
        log(f"  単体 {t:11s} AUROC={singles[t]:.4f}")

    cur = [max(singles, key=singles.get)]
    best_au = singles[cur[0]]
    while len(cur) < min(6, len(pool)):
        cand_best, cand_au = None, best_au
        for t in pool:
            if t in cur:
                continue
            tmp = WD / "ve_tmp.npz"
            if not mean_npz([pool[x] for x in cur + [t]], tmp):
                continue
            a = auroc_of(tmp, "tmp")
            if a and a > cand_au:
                cand_best, cand_au = t, a
        if not cand_best:
            break
        cur.append(cand_best)
        best_au = cand_au
        log(f"  追加 {cand_best} → {len(cur)}種 AUROC={best_au:.4f}")
    log(f"選択結果: {cur}  AUROC={best_au:.4f} (現行 0.8867)")

    if best_au <= 0.8867 + 0.0005:
        log("現行アンサンブルを超えず → 提出見送り")
        return

    # --- PQ 確認（4view → 8view）---
    ens_val = WD / "probs_val_ens2.npz"
    if not mean_npz([pool[t] for t in cur], ens_val):
        return
    j8 = VAL8.with_name(VAL8.stem + "_ens2.json")
    if not run(["python3", "src/apply_verifier.py", "--preds", str(VAL8),
                "--probs", str(ens_val), "--mode", "sqrt", "--out", str(j8)], "ve_ver8.log"):
        return
    pq8, pp8 = tune(j8, "ens2")
    if not pq8:
        return
    log(f"8view + 新アンサンブル({len(cur)}種): val-PQ {pq8:.4f} (現行最良 {BEST})")
    if pq8 <= BEST + 0.0005:
        log("現行最良を超えず → 提出見送り（枠温存）")
        return

    # --- test 確率 → 適用 → 提出 ---
    parts = []
    for t in cur:
        d = HAVE.get(t, WD / f"semseg_{t}")
        pt = d / "probs_test.npz"
        if not pt.exists():
            if not run(["python3", "-u", "src/semseg_infer.py", "--weights",
                        str(d / "semseg_best.pth"), "--split", "test", "--out", str(pt)],
                       f"ve_tinf_{t}.log"):
                return
        parts.append(pt)
    ens_test = WD / "probs_test_ens2.npz"
    if not mean_npz(parts, ens_test):
        return
    jt = TEST8.with_name(TEST8.stem + "_ens2.json")
    if not run(["python3", "src/apply_verifier.py", "--preds", str(TEST8),
                "--probs", str(ens_test), "--mode", "sqrt", "--out", str(jt)], "ve_vertest.log"):
        return
    sub = REPO / "submission_verexp.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", pp8, "--out", str(sub)], "ve_csv.log"):
        return
    if "PASS" not in (WD / "ve_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = f"MaskDINO 8-view + {len(cur)}-model semantic verifier ({'+'.join(cur)}) val-PQ {pq8:.4f}"
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg[:480]], "ve_submit.log", timeout=900):
        log(f"✅ 提出完了: val-PQ {pq8:.4f}  メンバー={cur}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 検証器拡張 終了 ====")
