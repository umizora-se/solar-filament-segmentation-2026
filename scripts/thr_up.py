"""mask_bin_thr を上げる方向を探索する（ご指示の施策1）。

根拠: 原寸学習で問題が反転した。
  旧モデル  惜しい外れの面積比 0.520（小さすぎ）→ 膨らませる施策を5回試し全失敗
  新モデル  惜しい外れの面積比 1.166（大きすぎ）、TP も 1.036
下げる方向は測って悪化を確認済み（0.4600 -> 0.4426）。上げる方向は未測定。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
OUT = WD / "r50_full2048_f0"
SUMMARY = WD / "thrup_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048.yaml"
BASE = 0.4600


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=4 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== mask_bin_thr を上げる {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: prob統合 / mask_bin_thr 0.50 → fold0-val {BASE}")
    log("面積比: TP 1.036 / 惜しい外れ 1.166（はみ出している）→ 締める方向を探る")

    res = {0.50: BASE}
    for thr in [0.60, 0.70, 0.80]:
        tag = f"prob{int(thr*100)}"
        raw = OUT / f"val_raw_{tag}.json"
        if not raw.exists():
            if not run(["python3", "-u", "src/predict_submit.py", "--config", CFG,
                        "--weights", str(OUT / "model_final.pth"),
                        "--split", "val", "--fold", "0", "--tta", "--tta-views", "8",
                        "--tta-merge", "prob", "--mask-bin-thr", str(thr),
                        "--dump-raw", str(raw), "--out", f"/tmp/tu_{tag}.csv"],
                       f"tu_{tag}_infer.log"):
                continue
        o = OUT / f"postproc_{tag}.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw),
                    "--fold", "0", "--out", str(o),
                    "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                    "--min-area", "100,200,300"], f"tu_{tag}_tune.log"):
            continue
        d = json.loads(o.read_text())
        res[thr] = d["val_pq"]
        log(f"  mask_bin_thr={thr:.2f}: PQ {d['val_pq']:.4f} "
            f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
            f"thr={d['score_thr']} area={d['min_area']})  Δ={d['val_pq']-BASE:+.4f}")
        # 面積比も見る（締まったか）
        if run(["python3", "-u", "src/error_decomp.py", "--preds", str(raw),
                "--postproc", str(o)], f"tu_{tag}_decomp.log"):
            for l in (WD / f"tu_{tag}_decomp.log").read_text().splitlines():
                if "中央値" in l:
                    log(f"    {l.strip()}")

    log("=== 結論 ===")
    for t in sorted(res):
        log(f"  mask_bin_thr {t:.2f}: {res[t]:.4f}")
    best = max(res, key=res.get)
    log(f"  最良 {best:.2f} → {res[best]:.4f}  Δ={res[best]-BASE:+.4f}")
    if res[best] >= BASE + 0.005:
        log("  ★ 採用に値する。test 側にも適用して提出候補にする")
    else:
        log("  改善は基準未満。0.50 のまま進める")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== mask_bin_thr を上げる 終了 ====")
