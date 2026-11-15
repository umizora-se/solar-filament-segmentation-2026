"""外部 GONG フレームへの疑似ラベル学習（半教師あり）の効果を fold0 で判定する。

根拠（OOF 誤り分解, 11/09）: 失点の主因は見逃し（FN 3050 / FP 1711）で、注釈者1人の
画像の RQ が最も低い（0.666）。学習 707枚は 656日分しかなく、別の日の太陽面を増やす。
ラベル転写（数分以内の近傍に正解を写す, −0.012）と違い、新しい日の画像を使う。

手順（fold0）:
  1. 取得完了を待つ（src/gong_unlabeled.py）
  2. 教師 = 1組目 fold0 モデル（fold0 の val 写真を学習していない）、8向き TTA(prob)
  3. 疑似ラベル = 提出と同じ後処理（thr 0.35 / area 200）を通した予測
  4. 学習 = 実ラベル×2 + 疑似ラベル、構成・iter は現行と同一
  5. fold0-val で評価

判定: 1組目 fold0 0.4600 に対し +0.005 以上なら 5fold 展開を相談する。
      （シードだけで fold0 は ±0.006 動くことに注意: 2組目 0.4538）
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "pl_summary.txt"
UNL = REPO / "datasets" / "gong_unlabeled"
TEACHER_CFG = "configs/maskdino_filament_r50_full2048.yaml"
CFG = "configs/maskdino_filament_r50_full2048_pl.yaml"
OUT = WD / "r50_full2048_pl_f0"
BASE = 0.4600
FOLD = 0
BUSY = "^python3 -u src/train_filament.py|^bash scripts/train_with_watchdog.sh"


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def wait_for(pattern):
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        time.sleep(120)


def run(cmd, lf, timeout=24 * 3600):
    if "src/predict_submit.py" in cmd:
        wait_for(BUSY)        # 推論は学習と同時に走らせない
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 疑似ラベル fold0 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    log("外部フレームの取得完了と、先行ジョブ（TTA 票数の判定）の完了を待機")
    wait_for("^python3 -u src/gong_unlabeled.py")
    for l in (WD / "gong_unlabeled.log").read_text(errors="ignore").splitlines()[-2:]:
        log(f"  [取得] {l}")
    wait_for("^python3 -u scripts/vote_stage.py")
    n_jpg = len(list((UNL / "jpeg").glob("*.jpeg")))
    log(f"外部フレーム {n_jpg} 枚")

    teacher = WD / f"pl_teacher_f{FOLD}.json"
    if not teacher.exists():
        log(f"教師推論（1組目 fold{FOLD}、8向き TTA、{n_jpg}枚 × 約10秒）")
        if not run(["python3", "-u", "src/predict_submit.py", "--config", TEACHER_CFG,
                    "--weights", str(WD / f"r50_full2048_f{FOLD}" / "model_final.pth"),
                    "--image-dir", str(UNL / "jpeg"), "--tta", "--tta-views", "8",
                    "--tta-merge", "prob", "--mask-bin-thr", "0.5", "--dump-raw", str(teacher),
                    "--out", f"/tmp/pl_teacher_f{FOLD}.csv"], f"pl_teacher_f{FOLD}.log"):
            return
    pj = UNL / f"pseudo_f{FOLD}.json"
    if not run(["python3", "-u", "src/make_pseudo.py", "--preds", str(teacher), "--fold", str(FOLD),
                "--out", str(pj), "--score-thr", "0.35", "--min-area", "200"], "pl_make_f0.log"):
        return
    for l in (WD / "pl_make_f0.log").read_text().splitlines()[-2:]:
        log(f"  {l}")

    # ---- 期待レコード数（リーク照合）: 空の疑似ラベル画像は FILTER_EMPTY で落ちる ----
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0,'src');"
                        "from filament_dataset import get_train_dicts_with_pseudo as g;"
                        f"d=g({FOLD}); print('N=', sum(1 for x in d if x['annotations']))"],
                       cwd=REPO, capture_output=True, text=True)
    m = re.search(r"N= (\d+)", r.stdout)
    if not m:
        log(f"★ 期待レコード数を計算できない: {r.stderr[-400:]}")
        return
    want = int(m.group(1))
    log(f"fold{FOLD} の期待学習レコード数（空でないもの）: {want}")

    if not (OUT / "model_final.pth").exists():
        log("動作確認（20 iter）")
        if not run(["python3", "-u", "src/train_filament.py", "--config", CFG, "--fold", str(FOLD),
                    "--smoke", "--output-dir", str(WD / "pl_smoke")], "pl_smoke.log", timeout=3600):
            return
        st = (WD / "pl_smoke.log").read_text(errors="ignore")
        mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", st)
        if not mm or int(mm[-1]) != want:
            log(f"★★ 動作確認のレコード数 {mm[-1] if mm else '?'} ≠ 期待 {want}。中止")
            return
        log(f"  動作確認 OK（レコード数 {mm[-1]}）")
        log("本学習 fold0（約7.5時間）")
        if not run(["bash", "scripts/train_with_watchdog.sh", CFG, str(OUT),
                    str(WD / "pl_train_f0.log"), "5000", str(FOLD)], "pl_watchdog_f0.log",
                   timeout=30 * 3600):
            return
    txt = (OUT / "log.txt").read_text(errors="ignore")
    mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
    got = int(mm[-1]) if mm else -1
    if got != want:
        log(f"★★ 学習レコード数 {got} ≠ 期待 {want}。中止")
        return
    for ck in OUT.glob("model_0*.pth"):
        ck.unlink()
    log(f"学習完了  レコード数 {got} ✓")

    raw = OUT / "val_raw_prob50.json"
    if not raw.exists():
        if not run(["python3", "-u", "src/predict_submit.py", "--config", CFG,
                    "--weights", str(OUT / "model_final.pth"), "--split", "val", "--fold", "0",
                    "--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                    "--dump-raw", str(raw), "--out", "/tmp/pl_val.csv"], "pl_infer_f0.log"):
            return
    o = OUT / "postproc_prob50.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], "pl_tune_f0.log"):
        return
    d = json.loads(o.read_text())
    log("=== 結論 ===")
    log(f"  1組目 fold0（基準）  {BASE:.4f}   参考: 2組目 0.4538")
    log(f"  + 疑似ラベル         {d['val_pq']:.4f}  Δ={d['val_pq'] - BASE:+.4f} "
        f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} thr={d['score_thr']} area={d['min_area']})")
    if d["val_pq"] >= BASE + 0.005:
        log("  ★ 改善。5fold 展開を相談する")
    else:
        log("  改善は基準（+0.005）未満")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 疑似ラベル fold0 判定 終了 ====")
