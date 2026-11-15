"""第3段階: 原寸レシピで全5fold展開 → OOF 707枚の判定基盤を作り直す。

目的:
  1. fold0-val 0.4600 が本物か（fold0 過適合でないか）を確定する
     ・LB が 0.40 で止まった原因として最有力な仮説
     ・検証器なしの提出で LB 0.39（検証器は効いている）→ 較正ずれ説は否定済み
  2. 以降の判定基盤を ±0.004 にする（fold0単独は ±0.0088）

構成は fold0 と完全に同一: フル2048原寸・クエリ150・実効バッチ8・5000 iter。
学習は watchdog 経由（PCシャットダウンや無言終了から自動再開）。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "stage3_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048.yaml"
BASE_OLD_OOF = 0.4228     # 旧モデル（縮小学習）+ 検証器 の OOF707枚
# 各 fold の正しい学習レコード数。これと一致しなければリークしている。
EXPECT_N = {0: 928, 1: 939, 2: 920, 3: 915, 4: 914}


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=24 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def busy():
    for pat in ["thr_up", "predict_submit", "tune_postproc"]:
        if subprocess.run(["pgrep", "-f", pat], capture_output=True).returncode == 0:
            return True
    return False


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 第3段階: 全5fold + OOF {time.strftime('%m/%d %H:%M')} ====\n")
    log("先行ジョブ（mask_bin_thr 探索）の完了を待機")
    while busy():
        time.sleep(120)
    log("GPU 解放を確認")
    log(f"旧モデルの OOF707枚 = {BASE_OLD_OOF}。原寸モデルでこれを超えるかを見る")

    # ---- fold1-4 を学習（fold0 は済み）----
    dirs = {0: WD / "r50_full2048_f0"}
    for k in [1, 2, 3, 4]:
        out = WD / f"r50_full2048_f{k}"
        dirs[k] = out
        if (out / "model_final.pth").exists():
            log(f"fold{k}: 学習済み")
            continue
        log(f"fold{k}: 学習開始（約7時間）")
        # fold ごとに DATASETS を差し替えた config が必要
        cfgk = REPO / "configs" / f"_s3_f{k}.yaml"
        s = Path(REPO / CFG).read_text()
        s = s.replace('TRAIN: ("filament_train_f0",)', f'TRAIN: ("filament_train_f{k}",)')
        s = s.replace('TEST: ("filament_val_f0",)', f'TEST: ("filament_val_f{k}",)')
        cfgk.write_text(s)
        if not run(["bash", "scripts/train_with_watchdog.sh", str(cfgk.relative_to(REPO)),
                    str(out), str(WD / f"s3_train_f{k}.log"), "5000", str(k)],
                   f"s3_watchdog_f{k}.log", timeout=24 * 3600):
            log(f"★ fold{k} の学習に失敗。以降をスキップ")
            continue
        if not (out / "model_final.pth").exists():
            log(f"★ fold{k}: model_final.pth が無い")
            continue
        # ★リーク検査: 学習に使われたレコード数が filament_train_f{k} と一致するか
        import re as _re
        txt = (out / "log.txt").read_text(errors="ignore")
        m = _re.search(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
        got = int(m.group(1)) if m else -1
        want = EXPECT_N[k]
        if got != want:
            log(f"★★ fold{k}: 学習レコード数 {got} ≠ 期待 {want} → リークの疑い。中止")
            return
        log(f"  fold{k}: 学習レコード数 {got}（期待 {want}）✓")

    # ---- 各fold の val を推論（8方向TTA・prob統合）----
    raws = {}
    for k in range(5):
        ck = dirs[k] / "model_final.pth"
        if not ck.exists():
            continue
        raw = dirs[k] / "val_raw_prob50.json"
        if k == 0:
            raw = dirs[0] / "val_raw_prob50.json"   # 既存の名前に合わせる
        if not raw.exists():
            log(f"fold{k}: val 推論（142枚、約18分）")
            cfgk = str(REPO / "configs" / f"_s3_f{k}.yaml") if k else CFG
            if not Path(cfgk).exists():
                cfgk = CFG
            if not run(["python3", "-u", "src/predict_submit.py", "--config", cfgk,
                        "--weights", str(ck), "--split", "val", "--fold", str(k),
                        "--tta", "--tta-views", "8", "--tta-merge", "prob",
                        "--mask-bin-thr", "0.5", "--dump-raw", str(raw),
                        "--out", f"/tmp/s3_f{k}.csv"], f"s3_infer_f{k}.log"):
                continue
        raws[k] = raw
        o = dirs[k] / "postproc_prob50.json"
        if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw),
                "--fold", str(k), "--out", str(o),
                "--score-thr", "0.20,0.25,0.30,0.35,0.40",
                "--min-area", "100,200,300"], f"s3_tune_f{k}.log"):
            d = json.loads(o.read_text())
            log(f"  fold{k}: val-PQ {d['val_pq']:.4f} "
                f"(thr={d['score_thr']} area={d['min_area']})")

    if len(raws) < 3:
        log("★ 揃った fold が少なすぎる。終了")
        return

    # ---- OOF 統合 → 基準値 ----
    merged = {}
    for k, p in sorted(raws.items()):
        d = json.loads(Path(p).read_text())
        merged.update(d["preds"] if "preds" in d else d)
    oof = WD / "oof_full2048.json"
    oof.write_text(json.dumps({"preds": merged}))
    log(f"[write] {oof}: {len(merged)} 画像（{len(raws)} fold 分）")

    o = WD / "postproc_oof_full2048.json"
    if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(oof), "--fold", "-1",
            "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40",
            "--min-area", "100,200,300"], "s3_tune_oof.log"):
        d = json.loads(o.read_text())
        log("=== 検出器のみ・OOF ===")
        log(f"  PQ {d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
            f"thr={d['score_thr']} area={d['min_area']})")

    # ---- 検証器を重ねて最終基準 ----
    pv = WD / "probs_oof_ens.npz"
    if pv.exists():
        j = WD / "oof_full2048_ver.json"
        if run(["python3", "src/apply_verifier.py", "--preds", str(oof),
                "--probs", str(pv), "--mode", "sqrt", "--out", str(j)], "s3_ver.log"):
            o2 = WD / "postproc_oof_full2048_ver.json"
            if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j),
                    "--fold", "-1", "--out", str(o2),
                    "--score-thr", "0.20,0.25,0.30,0.35,0.40",
                    "--min-area", "100,200,300"], "s3_tune_oof_ver.log"):
                d2 = json.loads(o2.read_text())
                log("=== 検出器 + 検証器・OOF（最終基準）===")
                log(f"  PQ {d2['val_pq']:.4f}  旧モデル {BASE_OLD_OOF} から "
                    f"Δ={d2['val_pq']-BASE_OLD_OOF:+.4f}")
                log(f"  参考: fold0単独では 0.4600 だった "
                    f"→ 差が大きければ fold0 過適合だった")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 第3段階 終了 ====")
