"""全データ学習（施策3）の完了後: test 推論 → 検証器適用 → 提出。

val が存在しないため PQ で事前検証できない。よって:
  - レシピは fold0 で検証済みのものを固定（エポック数を揃えた 6218 iter）
  - 後処理パラメータと検証器も fold0 で決めた値をそのまま流用
  - 判断は LB で行う。悪化したら現行最良（submission_verexp.csv）に戻す
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "alldata_summary.txt"
PROBS_TEST = WD / "probs_test_ens2.npz"     # 施策1で選ばれた4種の平均
POSTPROC = WD / "r50_f0" / "postproc_ens2.json"


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


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 全データ学習パイプライン {time.strftime('%m/%d %H:%M')} ====\n")
    log("学習の完了を待機中（ETA 約4.3時間）")
    while subprocess.run(["pgrep", "-f", "train_filament"], capture_output=True).returncode == 0:
        time.sleep(180)

    ckpt = WD / "r50_alldata" / "model_final.pth"
    if not ckpt.exists():
        log("★ model_final.pth が無い（学習失敗）。終了")
        return
    log(f"学習完了: {ckpt}")

    if not POSTPROC.exists():
        log(f"★ 後処理パラメータが無い: {POSTPROC}")
        return
    pp = json.loads(POSTPROC.read_text())
    log(f"流用する後処理: score_thr={pp['score_thr']} min_area={pp['min_area']} "
        f"(fold0 で val-PQ {pp['val_pq']:.4f})")

    # --- test 8方向TTA 推論 ---
    traw = WD / "r50_alldata" / "test_raw_tta8.json"
    if not traw.exists():
        log("test 8方向TTA 推論（約1.5-2時間）")
        if not run(["python3", "-u", "src/predict_submit.py",
                    "--config", "configs/maskdino_filament_r50_alldata.yaml",
                    "--weights", str(ckpt), "--tta", "--tta-views", "8",
                    "--dump-raw", str(traw), "--out", "/tmp/alldata_test.csv"],
                   "ad_test_infer.log"):
            return

    # --- 検証器（施策1の4種アンサンブル）を適用 ---
    jt = traw.with_name(traw.stem + "_ver.json")
    if not run(["python3", "src/apply_verifier.py", "--preds", str(traw),
                "--probs", str(PROBS_TEST), "--mode", "sqrt", "--out", str(jt)],
               "ad_ver.log"):
        return

    sub = REPO / "submission_alldata.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", str(POSTPROC), "--out", str(sub)], "ad_csv.log"):
        return
    if "PASS" not in (WD / "ad_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = ("MaskDINO R50 trained on ALL 707 photos (6218 iter, epoch-matched) "
           "+ 8-view TTA + 4-model semantic verifier")
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg], "ad_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")
        log("  注意: val が無いため LB でのみ判断する。悪化なら submission_verexp.csv が最良")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 全データ学習パイプライン 終了 ====")
