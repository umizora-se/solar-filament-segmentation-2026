"""原寸クロップ（縮小しない）での学習と、fold0-val での直接比較。

発見: 従来の学習は ResizeScale(target=1024) により 2048 画像全体を約1024へ縮小していた。
      実測で個体の面積中央が 960px -> 188-386px（約2倍の縮小）。
      推論はフル2048の原寸で行うため、物体スケールが学習と推論で2倍ずれていた。
      設計原則として掲げていた「一度も縮小しない」が実際には成立していなかった。

これが説明する観測:
  ・惜しい外れの長さ比 0.695（細部が縮小で失われる）
  ・マスクヘッド高解像度化が失敗(-0.13)（入力側で既に情報が失われている）
  ・Swin-L でも SQ が 0.688 で微動もしない（表現力ではなく入力解像度の制約）

比較（fold0 の同じ142枚・8方向TTA・後処理チューニング後）:
  現行(縮小あり) val-PQ 0.4268
  原寸クロップ   val-PQ ?
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "full2048_summary.txt"
BASE = 0.4268


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=14 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 原寸クロップ検証 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"比較: 現行(2048->約1024に縮小) fold0-val {BASE}  vs  フル2048原寸")
    log("フル2048原寸・クエリ150・実効バッチ8・5000 iter（43エポック相当＝従来と同一）")

    out = WD / "r50_full2048_f0"
    if not (out / "model_final.pth").exists():
        # 10/11 19:42 に原因不明の無言終了が発生したため、watchdog 経由で起動する
        # （250 iter 毎のチェックポイントから自動再開。1回の損失は最大20分）
        cmd = ["bash", "scripts/train_with_watchdog.sh",
               "configs/maskdino_filament_r50_full2048.yaml", str(out),
               str(WD / "fl_train.log"), "5000"]
        if not run(cmd, "fl_watchdog.log", timeout=20 * 3600):
            return
        if not (out / "model_final.pth").exists():
            log("★ watchdog が20回試行しても完走しなかった")
            return
    log("学習完了")

    # --- 学習数の当否を見る: 中間チェックポイントを安価に比較（同じ50枚・TTA無し） ---
    log("学習曲線の確認（iter 2000/3000/4000/4999 を同じ50枚・TTA無しで比較）")
    import re
    curve = {}
    for it in [2000, 3000, 4000]:
        ck = out / f"model_{it-1:07d}.pth"
        if not ck.exists():
            continue
        rj = out / f"val50_{it}.json"
        if not run(["python3", "-u", "src/predict_submit.py",
                    "--config", "configs/maskdino_filament_r50_full2048.yaml",
                    "--weights", str(ck), "--split", "val", "--fold", "0",
                    "--limit", "50", "--dump-raw", str(rj),
                    "--out", f"/tmp/c{it}.csv"], f"fl_c{it}_infer.log"):
            continue
        oj = out / f"postproc50_{it}.json"
        if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(rj), "--fold", "0",
                "--out", str(oj), "--score-thr", "0.25,0.30,0.35,0.40",
                "--min-area", "100,200"], f"fl_c{it}_tune.log"):
            curve[it] = json.loads(oj.read_text())["val_pq"]
            log(f"  iter {it}: 50枚PQ {curve[it]:.4f}")
    fin = out / "model_final.pth"
    rj = out / "val50_5000.json"
    if run(["python3", "-u", "src/predict_submit.py",
            "--config", "configs/maskdino_filament_r50_full2048.yaml",
            "--weights", str(fin), "--split", "val", "--fold", "0",
            "--limit", "50", "--dump-raw", str(rj), "--out", "/tmp/c5000.csv"],
           "fl_c5000_infer.log"):
        oj = out / "postproc50_5000.json"
        if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(rj), "--fold", "0",
                "--out", str(oj), "--score-thr", "0.25,0.30,0.35,0.40",
                "--min-area", "100,200"], "fl_c5000_tune.log"):
            curve[5000] = json.loads(oj.read_text())["val_pq"]
            log(f"  iter 5000: 50枚PQ {curve[5000]:.4f}")
    if len(curve) >= 2:
        best_it = max(curve, key=curve.get)
        log(f"  曲線の最良点: iter {best_it}（5000が最良なら学習不足の可能性、"
            f"それ以前なら過学習の可能性）")

    raw = out / "val_raw_tta8.json"
    if not raw.exists():
        log("fold0-val 推論（8方向TTA、142枚）")
        if not run(["python3", "-u", "src/predict_submit.py",
                    "--config", "configs/maskdino_filament_r50_full2048.yaml",
                    "--weights", str(out / "model_final.pth"),
                    "--split", "val", "--fold", "0", "--tta", "--tta-views", "8",
                    "--dump-raw", str(raw), "--out", "/tmp/native_val.csv"],
                   "fl_infer.log"):
            return
    o = out / "postproc.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], "fl_tune.log"):
        return
    d = json.loads(o.read_text())
    pq = d["val_pq"]
    log("=== 結論 ===")
    log(f"  現行(縮小あり) fold0-val PQ {BASE:.4f}")
    log(f"  フル2048原寸   fold0-val PQ {pq:.4f}  Δ={pq-BASE:+.4f} "
        f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} thr={d['score_thr']} area={d['min_area']})")
    if pq >= BASE + 0.005:
        log("  ★ 改善。スケール不一致が本当のボトルネックだった")
        log("  → 次: OOF 5fold へ展開し、検証器も再学習して提出")
    elif pq > BASE:
        log("  △ 改善はあるが基準未満")
    else:
        log("  ✗ 改善なし。縮小は実害ではなかった（縮小が拡張として働いていた可能性）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 原寸クロップ検証 終了 ====")
