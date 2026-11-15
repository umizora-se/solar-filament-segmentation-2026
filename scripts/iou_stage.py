"""IoU 考慮型の分類ターゲット（FILAMENT.IOU_CLS）を fold0 で判定し、通れば 5fold に展開して OOF で判定する。

根拠（11/10 の考察）:
  PQ を最大にする条件は「予測の一致確率 p × 一致時の IoU s > PQ/2 ≈ 0.24」。s ≈ 0.69 なら p > 0.34 で、
  実測の最適閾値 0.35 と一致する。ところが分類ターゲットは一致すれば IoU に関係なく 1 なので、スコアが
  IoU を反映しない。見逃した GT の 75% は閾値未満の予測としては出ており、並び順の問題とみられる。
  後処理の再ランキング（-0.013）や MetaSeg（-0.030）は特徴に情報が無く失敗したため、学習時に直す。
  文献: Stable-DINO（position-supervised loss, +1.4 AP）、Align-DETR（IA-BCE）。

変更点は分類ターゲットだけ（構成・iter・データは 1組目と同一。SEED=-1 なのでシード差は残る）。

判定（事前に決定）:
  fold0: 1組目 0.4600 以上なら fold1-4 へ進む（fold0 はシードだけで ±0.006 動く: 2組目 0.4538。足切りに使う）
  OOF 707枚: 1組目 0.4711 と比較し、DECISION_RULES ルール1（fold 一貫性・画像ブートストラップ）で採否を決める
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "iou_summary.txt"
EXPECT_N = {0: 928, 1: 939, 2: 920, 3: 915, 4: 914}
BASE_F0 = 0.4600
BASE_OOF = 0.4711
BUSY = "^python3 -u (src/train_filament.py|src/predict_submit.py|scripts/vote1_submit.py)|^bash scripts/train_with_watchdog.sh"
TTA = ["--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5"]
GRID = ["--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45", "--min-area", "100,200,300"]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def busy():
    return subprocess.run(["pgrep", "-f", BUSY], capture_output=True).returncode == 0


def run(cmd, lf, timeout=24 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def n_records(text):
    m = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", text)
    return int(m[-1]) if m else -1


def fmt(d):
    return (f"{d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
            f"thr={d['score_thr']} area={d['min_area']})")


def train_eval(k):
    """fold k を学習し、val を 8向き TTA で推論して後処理を探索する。失敗時 None。"""
    cfg = f"configs/_iou_f{k}.yaml"
    out = WD / f"r50_full2048_iou_f{k}"
    if not (out / "model_final.pth").exists():
        log(f"fold{k}: 学習開始（約7.5時間）")
        if not run(["bash", "scripts/train_with_watchdog.sh", cfg, str(out),
                    str(WD / f"iou_train_f{k}.log"), "5000", str(k)], f"iou_watchdog_f{k}.log",
                   timeout=30 * 3600):
            return None
    got = n_records((out / "log.txt").read_text(errors="ignore"))
    if got != EXPECT_N[k]:
        log(f"★★ fold{k}: 学習レコード数 {got} ≠ 期待 {EXPECT_N[k]} → リークの疑い。中止")
        return None
    for ck in out.glob("model_0*.pth"):          # ディスク節約（空き約 49GB）
        ck.unlink()
    log(f"fold{k}: 学習完了  レコード数 {got} ✓")
    raw = out / "val_raw_prob50.json"
    if not raw.exists():
        if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg,
                    "--weights", str(out / "model_final.pth"), "--split", "val", "--fold", str(k)] +
                   TTA + ["--dump-raw", str(raw), "--out", f"/tmp/iou_val{k}.csv"], f"iou_infer_f{k}.log"):
            return None
    o = out / "postproc_prob50.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", str(k),
                "--out", str(o)] + GRID, f"iou_tune_f{k}.log"):
        return None
    d = json.loads(o.read_text())
    log(f"  fold{k}: {fmt(d)}")
    return raw


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== IoU 考慮型の分類ターゲット 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    log("先行ジョブ（1組目の票数つき test 推論と提出）の終了を待機")
    while busy():
        time.sleep(120)

    if not (WD / "r50_full2048_iou_f0" / "model_final.pth").exists():
        log("動作確認（20 iter）")
        if not run(["python3", "-u", "src/train_filament.py", "--config", "configs/_iou_f0.yaml",
                    "--fold", "0", "--smoke", "--output-dir", str(WD / "iou_smoke")], "iou_smoke.log",
                   timeout=3600):
            return
        st = (WD / "iou_smoke.log").read_text(errors="ignore")
        if "cls_target = mask IoU" not in st:
            log("★★ 動作確認: IoU 考慮型ターゲットが有効になっていない。中止")
            return
        if n_records(st) != EXPECT_N[0]:
            log(f"★★ 動作確認: レコード数 {n_records(st)} ≠ 期待 {EXPECT_N[0]}。中止")
            return
        if re.search(r"loss_ce[^\n]*nan|FloatingPointError", st, re.I):
            log("★★ 動作確認: 損失が NaN。中止")
            return
        log(f"  動作確認 OK（レコード数 {EXPECT_N[0]}、IoU 考慮型ターゲット有効）")

    raws = [train_eval(0)]
    if raws[0] is None:
        return
    d0 = json.loads((WD / "r50_full2048_iou_f0" / "postproc_prob50.json").read_text())
    log(f"  fold0 判定: 1組目 {BASE_F0:.4f} → IoU 考慮型 {d0['val_pq']:.4f}  Δ={d0['val_pq'] - BASE_F0:+.4f}")
    if d0["val_pq"] < BASE_F0:
        log("  fold0 で 1組目を下回った → 打ち切り（5fold には展開しない）")
        return
    log("  fold0 で 1組目以上 → fold1-4 に展開（約32時間）")

    for k in range(1, 5):
        r = train_eval(k)
        if r is None:
            return
        raws.append(r)

    merged = {}
    for r in raws:
        merged.update(json.loads(r.read_text())["preds"])
    j = WD / "oof_iou.json"
    j.write_text(json.dumps({"preds": merged}))
    o = WD / "postproc_oof_iou.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j), "--fold", "-1",
                "--out", str(o)] + GRID, "iou_tune_oof.log"):
        return
    d = json.loads(o.read_text())
    log("=== 結論（OOF 707枚）===")
    log(f"  1組目（基準）      {BASE_OOF:.4f}")
    log(f"  IoU 考慮型         {fmt(d)}  Δ={d['val_pq'] - BASE_OOF:+.4f}  [{len(merged)}枚]")
    log("  採否は fold 別の差と画像ブートストラップで判断する（DECISION_RULES ルール1）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== IoU 考慮型の分類ターゲット 判定 終了 ====")
