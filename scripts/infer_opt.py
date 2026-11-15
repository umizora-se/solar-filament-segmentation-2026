"""原寸モデル上で推論側を再最適化する（施策②③）。

旧モデル(縮小あり)向けに調整した推論設定を、新モデルで引き直す。
  ② TTA統合: vote（画素多数決）対 prob（確率平均）
     実測: vote は端を削る（長さ比 TTA無し0.754 -> 8方向0.664）
     prob は確率を平均してから一度だけ2値化するので端が残る
  ③ mask_bin_thr: vote では確率が飽和して無意味だったが、prob では中間値が
     生まれるため初めて意味を持つ
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
OUT = WD / "r50_full2048_f0"
SUMMARY = WD / "inferopt_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048.yaml"
BASE = 0.4571          # 原寸モデル・vote・mask_bin_thr 0.5


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


def evaluate(tag, extra):
    raw = OUT / f"val_raw_{tag}.json"
    if not raw.exists():
        cmd = ["python3", "-u", "src/predict_submit.py", "--config", CFG,
               "--weights", str(OUT / "model_final.pth"),
               "--split", "val", "--fold", "0", "--tta", "--tta-views", "8",
               "--dump-raw", str(raw), "--out", f"/tmp/io_{tag}.csv"] + extra
        if not run(cmd, f"io_{tag}_infer.log"):
            return None
    o = OUT / f"postproc_{tag}.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], f"io_{tag}_tune.log"):
        return None
    return json.loads(o.read_text())


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 推論側の再最適化 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: 原寸モデル・vote統合・mask_bin_thr 0.5 → fold0-val {BASE}")

    results = {}
    # prob 統合 × 2値化閾値（vote では飽和して効かなかったが prob では効く）
    for thr in [0.50, 0.40, 0.30, 0.20]:
        tag = f"prob{int(thr*100)}"
        d = evaluate(tag, ["--tta-merge", "prob", "--mask-bin-thr", str(thr)])
        if d:
            results[tag] = d["val_pq"]
            log(f"  prob統合 / mask_bin_thr={thr:.2f}: PQ {d['val_pq']:.4f} "
                f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
                f"thr={d['score_thr']} area={d['min_area']})  Δ={d['val_pq']-BASE:+.4f}")
    # 参考: vote でも閾値を動かしてみる
    for thr in [0.40]:
        tag = f"vote{int(thr*100)}"
        d = evaluate(tag, ["--tta-merge", "vote", "--mask-bin-thr", str(thr)])
        if d:
            results[tag] = d["val_pq"]
            log(f"  vote統合 / mask_bin_thr={thr:.2f}: PQ {d['val_pq']:.4f}  "
                f"Δ={d['val_pq']-BASE:+.4f}")

    if not results:
        log("★ 全条件で失敗")
        return
    best = max(results, key=results.get)
    log("=== 結論 ===")
    log(f"  基準(vote/0.5)  {BASE:.4f}")
    log(f"  最良 {best}     {results[best]:.4f}  Δ={results[best]-BASE:+.4f}")
    if results[best] >= BASE + 0.005:
        log("  ★ 採用。端の侵食対策が効いた")
    else:
        log("  改善は基準未満。vote/0.5 のまま進める")

    # 誤差分解を新モデルで測り直す（長さの途切れが治ったか）
    bestraw = OUT / f"val_raw_{best}.json"
    bestpp = OUT / f"postproc_{best}.json"
    log("誤差分解を新モデルで再測定")
    if run(["python3", "-u", "src/error_decomp.py", "--preds", str(bestraw),
            "--postproc", str(bestpp)], "io_decomp.log"):
        for l in (WD / "io_decomp.log").read_text().splitlines():
            if any(k in l for k in ["PQ=", "完全FP", "惜しい外れ", "中央値", "TP=", "面積比"]):
                log("  " + l.strip())


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 推論側の再最適化 終了 ====")
