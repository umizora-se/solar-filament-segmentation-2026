"""原寸モデル（フル2048・クエリ150）で test 推論 → 検証器適用 → 提出。

fold0-val 0.4600（ベースライン 0.4268 から +0.033）。LB 0.40 更新を狙う。
検証器は旧モデル向けに学習したものを流用する（検証器は検出器に依存しないため
確率マップはそのまま使えるが、sqrt の較正が新モデルに合うかは LB で確認する）。
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
OUT = WD / "r50_full2048_f0"
SUMMARY = WD / "submit_full2048_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048.yaml"
PROBS_TEST = WD / "probs_test_ens.npz"      # 現行3種（LB0.40 と同じ構成）
PP = OUT / "postproc_prob50.json"


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=6 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 原寸モデルの提出 {time.strftime('%m/%d %H:%M')} ====\n")
    pp = json.loads(PP.read_text())
    log(f"fold0-val {pp['val_pq']:.4f}（基準 0.4268 / 現行LB 0.40）")
    log(f"後処理: score_thr={pp['score_thr']} min_area={pp['min_area']}, prob統合, mask_bin_thr 0.5")

    # test 推論（180枚 × 8方向TTA、約7.5秒/枚 → 約25分）
    traw = OUT / "test_raw_prob50.json"
    if not traw.exists():
        log("test 推論（8方向TTA・prob統合、180枚）")
        if not run(["python3", "-u", "src/predict_submit.py", "--config", CFG,
                    "--weights", str(OUT / "model_final.pth"),
                    "--tta", "--tta-views", "8", "--tta-merge", "prob",
                    "--mask-bin-thr", "0.5",
                    "--dump-raw", str(traw), "--out", "/tmp/sf_test.csv"],
                   "sf_test_infer.log"):
            return
    log("検証器（3種アンサンブル・sqrt）を適用")
    jt = OUT / "test_raw_prob50_ver.json"
    if not run(["python3", "src/apply_verifier.py", "--preds", str(traw),
                "--probs", str(PROBS_TEST), "--mode", "sqrt", "--out", str(jt)],
               "sf_ver.log"):
        return
    sub = REPO / "submission_full2048.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", str(PP), "--out", str(sub)], "sf_csv.log"):
        return
    if "PASS" not in (WD / "sf_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = (f"MaskDINO R50 NATIVE full-2048 (no downscale, 150 queries) + 8-view TTA "
           f"(prob merge) + 3-model verifier, fold0-val {pp['val_pq']:.4f}")
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg[:480]], "sf_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")
        time.sleep(150)
        r = subprocess.run(["kaggle", "competitions", "submissions",
                            "-c", "filament-segmentation-2026"],
                           capture_output=True, text=True, timeout=300)
        for l in r.stdout.splitlines()[:4]:
            log("  " + l[:150])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 原寸モデルの提出 終了 ====")
