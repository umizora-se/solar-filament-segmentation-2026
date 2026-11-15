"""OOF（out-of-fold）検証基盤を作る。

狙い: 検証セットを 142枚（fold0のみ）から 707枚（全画像）に広げる。
各画像を「その画像を学習していないモデル」で予測すれば、全画像が検証に使える。
識別できる差が約 1/√5 ≒ 0.45 倍に縮み、+0.002 程度の改善も判定できるようになる。

これまで142枚で判断していたため、val +0.0009 を改善と誤認して LB -0.01 を招いた
（docs/DECISION_RULES.md）。その再発を防ぐ基盤。

手順:
  1. fold1-4 の val 推論（8方向TTA）… fold0 は済み（val_raw_tta8.json）
  2. 各 fold で独立に閾値をチューニング → 5つの推定値の分布を見る
     （全データ学習が失敗した原因＝閾値の転移性が悪いのか、をここで確認できる）
  3. 5つの生予測を1つに統合 → OOF 707枚の検証基盤
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "oof_summary.txt"


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=8 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== OOF基盤の構築 {time.strftime('%m/%d %H:%M')} ====\n")
    log("目的: 検証セットを 142枚 → 707枚 に拡張する")

    raws = {0: WD / "r50_f0" / "val_raw_tta8.json"}

    # ---- 1. fold1-4 の val 推論（8方向TTA）----
    for k in [1, 2, 3, 4]:
        out = WD / f"r50_f{k}" / "val_raw_tta8.json"
        if out.exists():
            log(f"fold{k}: 既存の推論結果を使う")
            raws[k] = out
            continue
        ck = WD / f"r50_f{k}" / "model_final.pth"
        if not ck.exists():
            log(f"★ fold{k}: model_final.pth が無い。スキップ")
            continue
        log(f"fold{k}: val 推論（8方向TTA、142枚、約40分）")
        if run(["python3", "-u", "src/predict_submit.py", "--weights", str(ck),
                "--split", "val", "--fold", str(k), "--tta", "--tta-views", "8",
                "--dump-raw", str(out), "--out", f"/tmp/oof_f{k}.csv"],
               f"oof_infer_f{k}.log"):
            raws[k] = out

    if len(raws) < 2:
        log("★ 推論結果が足りない。終了")
        return

    # ---- 2. 各 fold で独立に閾値チューニング（転移性の確認）----
    log("=== 各foldの最適閾値（分布を見る）===")
    params = {}
    for k, p in sorted(raws.items()):
        o = WD / f"r50_f{k}" / "postproc_oof.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(p),
                    "--fold", str(k), "--out", str(o),
                    "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                    "--min-area", "100,200,300"], f"oof_tune_f{k}.log"):
            continue
        d = json.loads(o.read_text())
        params[k] = d
        log(f"  fold{k}: thr={d['score_thr']} area={d['min_area']} "
            f"val-PQ={d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f})")

    if params:
        thrs = [d["score_thr"] for d in params.values()]
        areas = [d["min_area"] for d in params.values()]
        pqs = [d["val_pq"] for d in params.values()]
        import statistics as st
        log(f"  閾値の分布: {thrs}  → 中央値 {st.median(thrs)}")
        log(f"  面積の分布: {areas}  → 中央値 {st.median(areas)}")
        log(f"  fold間PQ: 平均 {st.mean(pqs):.4f}  標準偏差 {st.pstdev(pqs):.4f}")
        log(f"  ★ 標準偏差が大きいほど、142枚での判定が当てにならない")

    # ---- 3. OOF 統合（707枚の検証基盤）----
    merged = {}
    for k, p in sorted(raws.items()):
        d = json.loads(Path(p).read_text())
        pr = d["preds"] if "preds" in d else d
        for stem, items in pr.items():
            merged[stem] = items
    oof = WD / "oof_raw_tta8.json"
    oof.write_text(json.dumps({"preds": merged}))
    log(f"[write] {oof}: {len(merged)} 画像（目標707）")

    # OOF 全体で閾値を測る（fold=-1 は全fold対象の意味）
    o = WD / "postproc_oof_all.json"
    if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(oof), "--fold", "-1",
            "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
            "--min-area", "100,200,300"], "oof_tune_all.log"):
        d = json.loads(o.read_text())
        log(f"=== OOF 707枚での基準値 ===")
        log(f"  thr={d['score_thr']} area={d['min_area']} PQ={d['val_pq']:.4f} "
            f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f})")
        log(f"  ※ fold0の142枚での 0.4268 と比較。これが今後の判定基準になる")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== OOF基盤の構築 終了 ====")
