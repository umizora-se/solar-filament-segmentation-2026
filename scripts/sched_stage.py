"""学習スケジュールの見直しを fold0 で判定する。

根拠: 現行（5000 iter・減衰点4000）の学習曲線（fold0 の50枚・TTA無し）が
        iter 2000 0.4364 / 3000 0.4392 / 4000 0.4380 / 5000 0.4516
      と、減衰前は横ばいで、学習率を下げた直後にだけ跳ねた。
      原寸化で1サンプルの情報量が4倍になったのに、学習量は縮小版で決めた値のまま。

比較（fold0 の142枚・8方向TTA(prob統合)・後処理チューニング後）:
    現行  5000 iter / 減衰 4000   fold0-val 0.4600
    今回  8000 iter / 減衰 6400   fold0-val ?
変更点は MAX_ITER と STEPS の2行だけ（configs/maskdino_filament_r50_full2048_8k.yaml）。
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "sched_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048_8k.yaml"
OUT = WD / "r50_full2048_8k_f0"
BASE = 0.4600
WANT_RECORDS = 928                  # fold0 の学習レコード数（近傍なし）
CURVE_5K = {2000: 0.4364, 3000: 0.4392, 4000: 0.4380, 5000: 0.4516}


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=20 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def tune(raw, out, grid_thr, grid_area, lf):
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(out), "--score-thr", grid_thr, "--min-area", grid_area], lf):
        return None
    return json.loads(Path(out).read_text())


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 学習スケジュール fold0 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"比較: 5000 iter/減衰4000 → fold0-val {BASE}  vs  8000 iter/減衰6400")

    if not (OUT / "model_final.pth").exists():
        log("本学習 fold0（約11時間）")
        if not run(["bash", "scripts/train_with_watchdog.sh", CFG, str(OUT),
                    str(WD / "sched_train_f0.log"), "8000", "0"], "sched_watchdog_f0.log",
                   timeout=30 * 3600):
            return
    txt = (OUT / "log.txt").read_text(errors="ignore")
    mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
    got = int(mm[-1]) if mm else -1
    if got != WANT_RECORDS:
        log(f"★★ 学習レコード数 {got} ≠ 期待 {WANT_RECORDS}。中止")
        return
    log(f"学習完了  レコード数 {got} ✓")

    # ---- 学習曲線（50枚・TTA無し、現行の曲線と同じ手順）----
    curve = {}
    for it, ck in [(5000, OUT / "model_0004999.pth"), (6250, OUT / "model_0006249.pth"),
                   (7000, OUT / "model_0006999.pth"), (8000, OUT / "model_final.pth")]:
        if not ck.exists():
            continue
        raw = OUT / f"val50_{it}.json"
        if not raw.exists() and not run(
                ["python3", "-u", "src/predict_submit.py", "--config", CFG, "--weights", str(ck),
                 "--split", "val", "--fold", "0", "--limit", "50", "--dump-raw", str(raw),
                 "--out", f"/tmp/sc{it}.csv"], f"sched_c{it}.log"):
            continue
        d = tune(raw, OUT / f"pp50_{it}.json", "0.25,0.30,0.35,0.40", "100,200",
                 f"sched_c{it}_tune.log")
        if d:
            curve[it] = d["val_pq"]
            log(f"  曲線 iter {it}: 50枚PQ {d['val_pq']:.4f}"
                + (f"（現行の同iter {CURVE_5K[it]:.4f}）" if it in CURVE_5K else ""))

    # ---- 本評価 ----
    raw = OUT / "val_raw_prob50.json"
    if not raw.exists():
        if not run(["python3", "-u", "src/predict_submit.py", "--config", CFG,
                    "--weights", str(OUT / "model_final.pth"), "--split", "val", "--fold", "0",
                    "--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                    "--dump-raw", str(raw), "--out", "/tmp/sched_val.csv"], "sched_infer.log"):
            return
    d = tune(raw, OUT / "postproc_prob50.json", "0.20,0.25,0.30,0.35,0.40,0.45",
             "100,200,300", "sched_tune.log")
    if not d:
        return
    log("=== 結論 ===")
    log(f"  5000 iter/減衰4000   fold0-val {BASE:.4f}")
    log(f"  8000 iter/減衰6400   fold0-val {d['val_pq']:.4f}  Δ={d['val_pq']-BASE:+.4f} "
        f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} thr={d['score_thr']} area={d['min_area']})")
    if d["val_pq"] >= BASE + 0.005:
        log("  ★ 改善。残り4fold へ展開し OOF 707枚で確定する価値あり")
    else:
        log("  改善は基準（+0.005）未満")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 学習スケジュール fold0 判定 終了 ====")
