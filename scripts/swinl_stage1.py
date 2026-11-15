"""段階1: Swin-L を fold0 で学習し、fold0-val で R50 と直接比較する。

比較対象（すべて fold0 の同じ142枚、8方向TTA、後処理チューニング後）:
    R50@1024   val-PQ 0.4268   ← 現行の本番モデル
    Swin-L@768 val-PQ ?

Swin-L 4-scale は素では 1024 で OOM だが、Swin 自身の勾配チェックポイント
(MODEL.SWIN.USE_CHECKPOINT) を有効にすると 8.8 GiB に収まる（768版の 10.0 GiB より軽い）。
これまで有効だったチェックポイントは MaskDINO エンコーダ側だけで、Swin バックボーン
（stage3 だけで18ブロック）には効いていなかった。
おかげでクロップ条件を R50 と揃えられ、比較から交絡が消えた。代償は速度（7.0 s/it）。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "swinl_summary.txt"
BASE_R50 = 0.4268


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
        f.write(f"\n==== Swin-L 段階1 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"比較: R50@1024 fold0-val {BASE_R50}  vs  Swin-L@768")
    log("学習 5000 iter（実効バッチ8・1024クロップ、R50と完全に同条件）、約9.8時間")

    out = WD / "r50_swinl_f0"
    if not (out / "model_final.pth").exists():
        cmd = ["python3", "-u", "src/train_filament.py",
               "--config", "configs/maskdino_filament_swinl_1024crop.yaml",
               "--fold", "0", "--output-dir", str(out), "--force"]
        if (out / "last_checkpoint").exists():
            cmd.append("--resume")          # iter999 の OOM 中断から再開
        if not run(cmd, "sw_train.log"):
            return
    log("学習完了")

    # fold0-val を 8方向TTA で本評価
    raw = out / "val_raw_tta8.json"
    if not raw.exists():
        log("fold0-val 推論（8方向TTA、142枚）")
        if not run(["python3", "-u", "src/predict_submit.py",
                    "--config", "configs/maskdino_filament_swinl_1024crop.yaml",
                    "--weights", str(out / "model_final.pth"),
                    "--split", "val", "--fold", "0", "--tta", "--tta-views", "8",
                    "--dump-raw", str(raw), "--out", "/tmp/swinl_val.csv"],
                   "sw_infer.log"):
            return
    o = out / "postproc.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], "sw_tune.log"):
        return
    d = json.loads(o.read_text())
    pq = d["val_pq"]
    log("=== 段階1 結論 ===")
    log(f"  R50@1024    fold0-val PQ {BASE_R50:.4f}")
    log(f"  Swin-L@1024 fold0-val PQ {pq:.4f}  Δ={pq-BASE_R50:+.4f} "
        f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} thr={d['score_thr']} area={d['min_area']})")
    if pq >= BASE_R50 + 0.005:
        log("  ★ 明確に改善 → 段階2（残り4foldへ展開）に進む価値あり")
    elif pq > BASE_R50:
        log("  △ 改善はあるが基準未満。段階2に進むコストに見合わない")
    else:
        log("  ✗ 改善なし。クロップ条件は R50 と同一なのでバックボーンの効果は無しと断定できる")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== Swin-L 段階1 終了 ====")
