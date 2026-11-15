"""セマンティック検証器のアーキ比較 → 最良（または平均アンサンブル）で提出まで。

選択基準は val-IoU ではなく AUROC（TP vs 完全FP の分離力）。検証器の仕事に直結するため。
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "arch_sweep_summary.txt"
VAL4 = WD / "r50_f0" / "val_raw_tta.json"
VAL8 = WD / "r50_f0" / "val_raw_tta8.json"
TEST8 = WD / "r50_f0" / "test_raw_tta8.json"
BASE4, BASE8 = 0.4198, 0.4268
BEST_SO_FAR = 0.4296          # 8view + baseline検証器(sqrt)
ARCHS = ["unetpp", "manet", "deeplabv3p", "segformer", "fpn"]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=4 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(cmd[:3])}… ({lf})")
    return r.returncode == 0


def tune(js, tag):
    out = str(Path(js).with_name(f"postproc_{tag}.json"))
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(js), "--fold", "0",
                "--out", out, "--score-thr", "0.25,0.30,0.35,0.40",
                "--min-area", "100,200,300"], f"as_tune_{tag}.log"):
        return None, None
    return json.loads(Path(out).read_text())["val_pq"], out


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== アーキ比較 開始 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: baseline(自前U-Net) AUROC=0.8722 val-IoU=0.504 / 8view+sqrt PQ={BEST_SO_FAR}")

    results = {"baseline": {"auroc": 0.8722, "iou": 0.504,
                            "probs": str(WD / "semseg_f0" / "probs_val.npz")}}

    # ---- 各アーキを学習 → val確率マップ → AUROC ----
    for a in ARCHS:
        out = WD / f"semseg_{a}"
        ok = False
        for bs in [4, 2, 1]:                      # OOM ならバッチを半減して再挑戦
            log(f"学習: {a} (batch={bs})")
            if run(["python3", "-u", "src/train_semseg.py", "--fold", "0",
                    "--arch", a, "--out", str(out), "--epochs", "20",
                    "--batch", str(bs)], f"as_train_{a}.log"):
                ok = True
                break
            if "OutOfMemory" not in (WD / f"as_train_{a}.log").read_text():
                break                             # OOM 以外の失敗は諦める
            log(f"  OOM → batch を下げて再試行")
        if not ok:
            continue
        lines = [l for l in (WD / f"as_train_{a}.log").read_text().splitlines()
                 if "best val-IoU" in l]
        iou = float(lines[-1].split("=")[1].split()[0]) if lines else 0.0
        probs = out / "probs_val.npz"
        if not run(["python3", "-u", "src/semseg_infer.py",
                    "--weights", str(out / "semseg_best.pth"), "--fold", "0",
                    "--split", "val", "--out", str(probs)], f"as_inf_{a}.log"):
            continue
        aj = WD / f"auroc_{a}.json"
        if not run(["python3", "-u", "src/eval_verifier.py", "--preds", str(VAL4),
                    "--probs", str(probs), "--out", str(aj)], f"as_auroc_{a}.log"):
            continue
        au = json.loads(aj.read_text())["auroc"]
        results[a] = {"auroc": au, "iou": iou, "probs": str(probs)}
        log(f"  {a}: AUROC={au:.4f}  val-IoU={iou:.4f}  (baseline 0.8722 / 0.504)")

    # ---- 確率マップの平均アンサンブル ----
    if len(results) >= 3:
        top = sorted(results, key=lambda k: -results[k]["auroc"])[:3]
        log(f"アンサンブル: {top} の確率平均")
        ens = WD / "probs_val_ens.npz"
        code = f'''
import numpy as np
srcs = {[results[k]["probs"] for k in top]!r}
ds = [np.load(s) for s in srcs]
keys = set(ds[0].files)
for d in ds[1:]:
    keys &= set(d.files)
out = {{k: (np.mean([d[k].astype(np.float32) for d in ds], 0)).astype(np.uint8)
       for k in sorted(keys)}}
np.savez_compressed({str(ens)!r}, **out)
print("ens", len(out))
'''
        (REPO / "/tmp/mk_ens.py").write_text(code) if False else Path("/tmp/mk_ens.py").write_text(code)
        if run(["python3", "/tmp/mk_ens.py"], "as_ens.log"):
            aj = WD / "auroc_ens.json"
            if run(["python3", "-u", "src/eval_verifier.py", "--preds", str(VAL4),
                    "--probs", str(ens), "--out", str(aj)], "as_auroc_ens.log"):
                au = json.loads(aj.read_text())["auroc"]
                results["ens"] = {"auroc": au, "iou": 0.0, "probs": str(ens),
                                  "members": top}
                log(f"  ens: AUROC={au:.4f}")

    if len(results) < 2:
        log("有効な候補なし。終了")
        return

    log("=== AUROC ランキング ===")
    for k in sorted(results, key=lambda k: -results[k]["auroc"]):
        log(f"  {k:11s} AUROC={results[k]['auroc']:.4f}  val-IoU={results[k]['iou']:.4f}")

    best = max(results, key=lambda k: results[k]["auroc"])
    log(f"最良検証器: {best} (AUROC {results[best]['auroc']:.4f})")
    if best == "baseline":
        log("baseline を超えるアーキなし → 提出見送り（現状維持）")
        return

    # ---- 4view で PQ 確認（mode 探索）----
    cands = {}
    for mode, thr in [("sqrt", None), ("mul", None), ("gate", 0.10), ("gate", 0.20)]:
        tag = f"b{mode}{thr or ''}".replace(".", "")
        j = VAL4.with_name(VAL4.stem + f"_{tag}.json")
        cmd = ["python3", "src/apply_verifier.py", "--preds", str(VAL4),
               "--probs", results[best]["probs"], "--mode", mode, "--out", str(j)]
        if thr:
            cmd += ["--sem-thr", str(thr)]
        if run(cmd, f"as_ver_{tag}.log"):
            pq, _ = tune(j, tag)
            if pq:
                log(f"  4view {mode}{thr or ''}: val-PQ {pq:.4f} (baseline sqrt 0.4251)")
                cands[(mode, thr)] = pq
    if not cands:
        log("PQ 評価に失敗。終了")
        return
    bm, bt = max(cands, key=cands.get)
    log(f"最良モード: {bm}{bt or ''} PQ={cands[(bm,bt)]:.4f}")

    # ---- 8view で最終確認 ----
    j8 = VAL8.with_name(VAL8.stem + "_bestver.json")
    cmd = ["python3", "src/apply_verifier.py", "--preds", str(VAL8),
           "--probs", results[best]["probs"], "--mode", bm, "--out", str(j8)]
    if bt:
        cmd += ["--sem-thr", str(bt)]
    if not run(cmd, "as_ver8.log"):
        return
    pq8, pp8 = tune(j8, "best8")
    if not pq8:
        return
    log(f"8view+{best}/{bm}: val-PQ {pq8:.4f}  (現行最良 {BEST_SO_FAR})")
    if pq8 <= BEST_SO_FAR + 0.0005:
        log("現行最良を超えず → 提出見送り（枠温存）")
        return

    # ---- test 確率マップ → 適用 → CSV → 提出 ----
    if not TEST8.exists():
        log("★ test 生予測が無い。終了")
        return
    if best == "ens":
        parts = []
        for mem in results["ens"]["members"]:
            src = WD / ("semseg_f0" if mem == "baseline" else f"semseg_{mem}")
            pt = src / "probs_test.npz"
            if not pt.exists():
                if not run(["python3", "-u", "src/semseg_infer.py",
                            "--weights", str(src / "semseg_best.pth"),
                            "--split", "test", "--out", str(pt)], f"as_tinf_{mem}.log"):
                    return
            parts.append(str(pt))
        ptest = WD / "probs_test_ens.npz"
        code = f'''
import numpy as np
ds = [np.load(s) for s in {parts!r}]
keys = set(ds[0].files)
for d in ds[1:]:
    keys &= set(d.files)
np.savez_compressed({str(ptest)!r}, **{{k: np.mean([d[k].astype(np.float32) for d in ds],0).astype(np.uint8) for k in sorted(keys)}})
'''
        Path("/tmp/mk_ens_t.py").write_text(code)
        if not run(["python3", "/tmp/mk_ens_t.py"], "as_enst.log"):
            return
    else:
        src = WD / f"semseg_{best}"
        ptest = src / "probs_test.npz"
        if not ptest.exists():
            if not run(["python3", "-u", "src/semseg_infer.py",
                        "--weights", str(src / "semseg_best.pth"),
                        "--split", "test", "--out", str(ptest)], "as_tinf.log"):
                return
    jt = TEST8.with_name(TEST8.stem + "_bestver.json")
    cmd = ["python3", "src/apply_verifier.py", "--preds", str(TEST8),
           "--probs", str(ptest), "--mode", bm, "--out", str(jt)]
    if bt:
        cmd += ["--sem-thr", str(bt)]
    if not run(cmd, "as_vertest.log"):
        return
    sub = REPO / "submission_arch.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", pp8, "--out", str(sub)], "as_csv.log"):
        return
    if "PASS" not in (WD / "as_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = f"MaskDINO 8-view TTA + {best} semantic verifier ({bm}) val-PQ {pq8:.4f}"
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg], "as_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== アーキ比較 終了 ====")
