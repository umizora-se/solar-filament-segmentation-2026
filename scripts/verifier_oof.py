"""検証器を fold ごとに学習し、OOF 707枚で「検証器の効果は本物か」を確定させる。

背景: 検証器の +0.005 は fold0 の142枚で測った値。fold間PQの標準偏差は 0.0088 あり、
      この差が本物かどうか判断できていない。検証器は現時点で唯一機能している改善なので、
      その効果を確定させることは今後の全判断の土台になる。

リーク回避: 検証器は「その画像を学習していない fold のモデル」で確率を出す。
            検出器(MaskDINO)側も同じ規則で OOF を作ってある。

学習は 3種 × fold1-4 = 12回（fold0 分は学習済み）。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "veroof_summary.txt"
OOF = WD / "oof_raw_tta8.json"
BASE_OOF = 0.4162            # 検出器のみ・OOF707枚での基準
ARCHS = [("baseline", "baseline", "resnet50"),
         ("deeplabv3p", "deeplabv3p", "resnet50"),
         ("segformer", "segformer", "resnet50")]


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
            "segformer": WD / "semseg_segformer"}[tag]


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 検証器のOOF評価 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"問い: 検証器の効果は本物か（検出器のみ OOF707枚で {BASE_OOF}）")

    probs = {0: {}}
    for tag, _, _ in ARCHS:                      # fold0 は学習済み
        p = fold0_dir(tag) / "probs_val.npz"
        if p.exists():
            probs[0][tag] = p
    log(f"fold0: 既存の確率マップ {sorted(probs[0])}")

    # ---- fold1-4 で 3種を学習 → 各 fold の val に推論 ----
    for k in [1, 2, 3, 4]:
        probs[k] = {}
        for tag, arch, enc in ARCHS:
            out = WD / f"semseg_f{k}_{tag}"
            ck = out / "semseg_best.pth"
            if not ck.exists():
                ok = False
                for bs in [4, 2, 1]:
                    log(f"学習: fold{k}/{tag} (batch={bs})")
                    cmd = ["python3", "-u", "src/train_semseg.py", "--fold", str(k),
                           "--arch", arch, "--out", str(out), "--epochs", "20",
                           "--batch", str(bs)]
                    if arch != "baseline":
                        cmd += ["--encoder", enc]
                    if run(cmd, f"vo_train_f{k}_{tag}.log"):
                        ok = True
                        break
                    if "OutOfMemory" not in (WD / f"vo_train_f{k}_{tag}.log").read_text(
                            errors="ignore"):
                        break
                if not ok:
                    continue
            pv = out / "probs_val.npz"
            if not pv.exists():
                if not run(["python3", "-u", "src/semseg_infer.py", "--weights", str(ck),
                            "--fold", str(k), "--split", "val", "--out", str(pv)],
                           f"vo_inf_f{k}_{tag}.log"):
                    continue
            probs[k][tag] = pv
        log(f"fold{k}: 完了 {sorted(probs[k])}")

    # ---- OOF の確率マップを組み立てる（各画像を、学習していない fold のモデルで）----
    log("OOF 確率マップを統合（3種平均）")
    parts = {k: [str(v[t]) for t, _, _ in ARCHS if t in v] for k, v in probs.items()}
    ok_folds = [k for k, v in parts.items() if len(v) == len(ARCHS)]
    log(f"  3種そろった fold: {ok_folds}")
    if len(ok_folds) < 3:
        log("★ そろった fold が少なすぎる。終了")
        return
    ens = WD / "probs_oof_ens.npz"
    code = f'''
import numpy as np
parts = {{k: v for k, v in {parts!r}.items() if k in {ok_folds!r}}}
out = {{}}
for k, srcs in parts.items():
    ds = [np.load(s) for s in srcs]
    keys = set(ds[0].files)
    for d in ds[1:]:
        keys &= set(d.files)
    for key in keys:
        out[key] = np.mean([d[key].astype(np.float32) for d in ds], 0).astype(np.uint8)
np.savez_compressed({str(ens)!r}, **out)
print("OOF確率マップ:", len(out), "画像")
'''
    Path("/tmp/vo_ens.py").write_text(code)
    if not run(["python3", "/tmp/vo_ens.py"], "vo_ens.log"):
        return
    log("  " + (WD / "vo_ens.log").read_text().strip().splitlines()[-1])

    # ---- 検証器あり/なしで OOF-PQ を比較 ----
    results = {}
    for mode, thr in [("sqrt", None), ("mul", None), ("gate", 0.10), ("gate", 0.20)]:
        tag = f"{mode}{thr or ''}".replace(".", "")
        j = WD / f"oof_raw_tta8_{tag}.json"
        cmd = ["python3", "src/apply_verifier.py", "--preds", str(OOF),
               "--probs", str(ens), "--mode", mode, "--out", str(j)]
        if thr:
            cmd += ["--sem-thr", str(thr)]
        if not run(cmd, f"vo_ver_{tag}.log"):
            continue
        o = WD / f"postproc_oof_{tag}.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j),
                    "--fold", "-1", "--out", str(o),
                    "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                    "--min-area", "100,200,300"], f"vo_tune_{tag}.log"):
            continue
        d = json.loads(o.read_text())
        results[tag] = d["val_pq"]
        log(f"  検証器 {mode}{thr or ''}: OOF-PQ {d['val_pq']:.4f} "
            f"(thr={d['score_thr']} area={d['min_area']})  Δ={d['val_pq']-BASE_OOF:+.4f}")

    if not results:
        log("★ 評価に失敗。終了")
        return
    best = max(results, key=results.get)
    delta = results[best] - BASE_OOF
    log(f"=== 結論 ===")
    log(f"  検出器のみ           OOF-PQ {BASE_OOF:.4f}")
    log(f"  + 検証器（{best}）  OOF-PQ {results[best]:.4f}  Δ={delta:+.4f}")
    if delta >= 0.005:
        log("  ★ 検証器の効果は本物（707枚で +0.005 以上）")
    elif delta > 0:
        log("  △ 改善はあるが小さい。142枚での +0.005 は過大評価だった可能性")
    else:
        log("  ✗ 707枚では改善なし。検証器の採用を見直す必要がある")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 検証器のOOF評価 終了 ====")
