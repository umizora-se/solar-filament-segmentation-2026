"""1組目を SWA-A（重み平均）に差し替えた本命（5fold × 2モデル確率平均融合）を作る。

swa_stage.py: SWA-A 単体 OOF 0.4721（1組目 0.4711 から +0.0010）。2組目と融合して OOF と test CSV を作る。
比較対象: 現行本命（1組目+2組目）OOF 0.4721（α=0）/ 0.4724（α=0.5）。提出はしない。

以下は下敷きにした vote2_stage.py の説明:
TTA 票数スコア（α）を本命（5fold × 2シード 確率平均融合）に適用する。

根拠（vote_stage.py, 1組目 1モデル×8向き, OOF 707枚）:
  α=0 0.4711 → α=0.25 0.4716 / 0.75 0.4717 / 1.0 0.4715。画像ブートストラップで
  α=0.75 は +0.0006 [95%: +0.0002, +0.0010]、上回る確率 100%。小さいが確か。

本命は 2モデル×8向き = 16ビューなので、票数の分布が 1モデルと違う。α は
2モデルの OOF で確かめ直す（事前に決めた値: α=0.5、山がなだらかなので中ほど）。

    score' = score × (votes / n_views) ^ α

手順:
  1. pl_stage.py（疑似ラベル判定）の終了を待つ（GPU を取り合わない）
  2. fold k: val 2モデル確率平均（票数つき）→ OOF 707枚
     α=0 が既存 0.4721 を再現することを確認、α ∈ {0, 0.25, 0.5, 0.75, 1} を評価
  3. α=0.5 が α=0 を上回れば採用。test 2モデル（票数つき）→ fold ごとに α で
     スコア変換 → 5fold マスク融合 mv2 → OOF(α=0.5) の後処理 → CSV
     確認用に α=0 の CSV も作り、既存提出と一致するか照合する
  提出はしない（CSV を作るまで）。
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "swa2_summary.txt"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
ALPHA = 0.5
BASE_OOF = 0.4721
GRID = ["--score-thr", "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45", "--min-area", "100,200,300"]
TTA = ["--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5"]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def wait_for(pattern):
    while subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode == 0:
        time.sleep(120)


def run(cmd, lf, timeout=12 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def rescore(preds, alpha):
    return {s: [{"rle": it["rle"], "score": it["score"] * (it["votes"] / it["nv"]) ** alpha}
                for it in items] for s, items in preds.items()}


def infer(k, split, raw, lf):
    if raw.exists():
        return True
    cmd = ["python3", "-u", "src/predict_submit.py", "--config",
           str(REPO / "configs" / f"_seed2_f{k}.yaml"), "--weights",
           str(WD / f"r50_full2048_f{k}" / "model_swaA.pth"),
           str(WD / f"r50_full2048_seed2_f{k}" / "model_final.pth")] + TTA + \
          ["--dump-raw", str(raw), "--out", f"/tmp/swa2_{split}{k}.csv"]
    if split == "val":
        cmd += ["--split", "val", "--fold", str(k)]
    return run(cmd, lf)


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== SWA 版の本命（SWA-A + 2組目） {time.strftime('%m/%d %H:%M')} ====\n")
    log("疑似ラベル判定（pl_stage.py）の終了を待機")
    wait_for("^python3 -u scripts/pl_stage.py")
    wait_for("^python3 -u src/train_filament.py|^bash scripts/train_with_watchdog.sh")

    # ---- 1. OOF（2モデル×8向き、票数つき）----
    merged = {}
    for k in range(5):
        raw = WD / f"r50_full2048_seed2_f{k}" / "val_raw_prob2m_swa_votes.json"
        if not raw.exists():
            log(f"fold{k}: val 2モデル推論（票数つき、約1.5時間）")
        if not infer(k, "val", raw, f"swa2_vinfer_f{k}.log"):
            return
        d = json.loads(raw.read_text())["preds"]
        merged.update(d)
        log(f"  fold{k}: 完了 {len(d)} 画像")

    log(f"OOF {len(merged)} 画像で α を評価")
    res = {}
    for a in ALPHAS:
        j = WD / f"oof_swa2_a{a}.json"
        j.write_text(json.dumps({"preds": rescore(merged, a)}))
        o = WD / f"postproc_oof_swa2_a{a}.json"
        if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j), "--fold", "-1",
                "--out", str(o)] + GRID, f"swa2_tune_a{a}.log"):
            res[a] = json.loads(o.read_text())
            d = res[a]
            log(f"  α={a:<4}  PQ {d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
                f"thr={d['score_thr']} area={d['min_area']})")
    if 0.0 not in res or ALPHA not in res:
        return
    log(f"  参考: 現行本命（SWA なし）α=0 {BASE_OOF}")
    gain = res[ALPHA]["val_pq"] - res[0.0]["val_pq"]
    log(f"  α={ALPHA}（事前に決めた値） Δ={gain:+.4f}")

    # ---- 2. test（2モデル×8向き、票数つき）→ α 変換 → 5fold 融合 → CSV ----
    raws = []
    for k in range(5):
        traw = WD / f"r50_full2048_seed2_f{k}" / "test_raw_prob2m_swa_votes.json"
        if not traw.exists():
            log(f"fold{k}: test 2モデル推論（票数つき、約2時間）")
        if not infer(k, "test", traw, f"swa2_tinfer_f{k}.log"):
            return
        raws.append(traw)
        log(f"  fold{k}: 完了")

    subs = {}
    for a in (0.0, ALPHA):
        per = []
        for k, traw in enumerate(raws):
            p = WD / f"r50_full2048_seed2_f{k}" / f"test_raw_prob2m_swa_votes_a{a}.json"
            p.write_text(json.dumps({"preds": rescore(json.loads(traw.read_text())["preds"], a)}))
            per.append(str(p))
        j = WD / f"test_ens5x2p_swa_votes_a{a}_mv2.json"
        if not run(["python3", "-u", "src/ensemble.py", "--preds"] + per +
                   ["--out", str(j), "--min-votes", "2"], f"swa2_fuse_a{a}.log"):
            return
        sub = REPO / f"submission_ens5x2seed_swa_votes_a{a}_mv2.csv"
        pp = WD / f"postproc_oof_swa2_a{a}.json"
        lf = f"swa2_csv_a{a}.log"
        if not run(["python3", "src/json_to_csv.py", "--preds", str(j), "--postproc", str(pp),
                    "--out", str(sub)], lf):
            return
        if "PASS" not in (WD / lf).read_text():
            log(f"★ CSV verify 失敗（α={a}）")
            return
        subs[a] = sub

    log("=== 結論 ===")
    log(f"  OOF 707枚  α=0 {res[0.0]['val_pq']:.4f} → α={ALPHA} {res[ALPHA]['val_pq']:.4f} "
        f"(Δ={gain:+.4f})")
    log(f"  本命の新 CSV: {subs[ALPHA].name}（未提出）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== SWA 版の本命（SWA-A + 2組目） 終了 ====")
