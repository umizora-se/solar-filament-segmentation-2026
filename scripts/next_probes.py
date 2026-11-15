"""施策3の完了後に走る検証（GPU競合を避けて逐次実行）。

1. SAM-HQ の適合性（GT箱プローブ）。判定: GT箱IoU > 0.75 で次段階、未満なら打ち切り
   （SAM2.1 は 0.6104、MaskDINO の TP は 0.686。実箱は GT箱より条件が悪いので余裕を要求）
2. タイリング推論 段階1（TTA無し、1024タイル）。判定: フル2048単体 0.3901 と比較
   学習は 1024 クロップ、推論はフル 2048 という視野の食い違いが未検証だったため。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "probes_summary.txt"
SAMHQ = REPO / "weights" / "sam_hq_vit_h.pth"
GATE_SAM = 0.75          # SAM-HQ の続行基準
BASE_PLAIN = 0.3901      # フル2048・TTA無し・チューニング後の val-PQ


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


def busy():
    """施策3のパイプライン（学習・推論・提出）が動いている間は待つ。"""
    for pat in ["train_filament", "alldata_pipeline", "predict_submit"]:
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 追加検証 {time.strftime('%m/%d %H:%M')} ====\n")
    log("施策3（全データ学習→test推論→提出）の完了を待機")
    while busy():
        time.sleep(180)
    log("GPU 解放を確認")

    # ---------- 1. SAM-HQ 適合性 ----------
    for _ in range(40):                              # 重みDL完了を最大20分待つ
        if subprocess.run(["pgrep", "-f", "curl.*sam_hq"],
                          capture_output=True).returncode != 0:
            break
        time.sleep(30)
    if SAMHQ.exists() and SAMHQ.stat().st_size > 2_000_000_000:
        log("【1】SAM-HQ 適合性テスト（GT箱プロンプト, 500件）")
        best = -1.0
        for mode in ["raw", "clahe"]:
            o = WD / f"samhq_gt_{mode}.json"
            if run(["python3", "-u", "src/sam_probe.py", "--mode", "gt", "--fold", "0",
                    "--limit", "500", "--ckpt", str(SAMHQ), "--img-mode", mode,
                    "--out", str(o)], f"pr_samhq_{mode}.log"):
                d = json.loads(o.read_text())
                log(f"  SAM-HQ vit_h / {mode}: 平均IoU={d['mean_iou']:.4f} "
                    f"IoU>0.5率={d['hit_rate']:.3f}")
                best = max(best, d["mean_iou"])
        log(f"  比較: SAM2.1 0.6104 / SAM1 0.5166 / MaskDINO TP 0.686")
        if best >= GATE_SAM:
            log(f"  ★ 基準 {GATE_SAM} を超えた（{best:.4f}）→ マスク置換の検証に進む価値あり")
        else:
            log(f"  判定: {best:.4f} < {GATE_SAM} → 打ち切り（マスク置換は却下のまま）")
    else:
        log("【1】SAM-HQ の重みが未取得。スキップ")

    # ---------- 2. タイリング推論 段階1 ----------
    log("【2】タイリング推論 段階1（TTA無し、1024タイル・overlap 256）")
    tj = WD / "r50_f0" / "val_raw_tile1024.json"
    if not run(["python3", "-u", "src/predict_submit.py",
                "--weights", "work_dirs/r50_f0/model_final.pth",
                "--split", "val", "--fold", "0", "--tile", "1024", "--tile-overlap", "256",
                "--dump-raw", str(tj), "--out", "/tmp/tile_val.csv"], "pr_tile_infer.log"):
        return
    out = WD / "r50_f0" / "postproc_tile.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(tj), "--fold", "0",
                "--out", str(out), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], "pr_tile_tune.log"):
        return
    d = json.loads(out.read_text())
    pq = d["val_pq"]
    log(f"  タイリング単体: val-PQ {pq:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
        f"thr={d['score_thr']} area={d['min_area']})")
    log(f"  比較: フル2048単体 {BASE_PLAIN}")
    if pq > BASE_PLAIN + 0.005:
        log("  ★ 明確に改善 → 段階2（TTA×タイリングの実装）へ進む価値あり")
    elif pq > BASE_PLAIN:
        log("  微増（測定誤差の範囲）。段階2の実装コストに見合うか要検討")
    else:
        log("  改善なし → 視野の食い違いは実害なし。タイリングは却下")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 追加検証 終了 ====")
