"""TTA の票数をスコアに入れる効果を OOF 707枚で判定する（1組目・1モデル×8向き）。

根拠（OOF 誤り分解, 11/09）: 完全に見逃した GT 1914件の 75% は、スコア 0.35 未満の
予測としては見つかっていた。足切りの順位付けを改善できれば RQ が上がる。
TTA 統合後のスコアは「賛成したビューのスコア平均」で、何ビューが賛成したかを
使っていない。票数はモデルのスコアとは独立した確信度の手がかりになる。

    score' = score × (votes / n_views) ^ α     α を OOF で選ぶ（α=0 が現行）

比較: α=0（現行と同じ推論の再現） vs 最良 α。+0.005 以上で採用。
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "vote_summary.txt"
BASE_CFG = "configs/maskdino_filament_r50_full2048.yaml"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
GRID = ["--score-thr", "0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45", "--min-area", "100,200,300"]


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


def rescore(preds, alpha):
    out = {}
    for stem, items in preds.items():
        out[stem] = [{"rle": it["rle"],
                      "score": it["score"] * (it["votes"] / it["nv"]) ** alpha} for it in items]
    return out


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== TTA 票数スコア 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    merged = {}
    for k in range(5):
        out = WD / f"r50_full2048_f{k}"
        cfg = f"configs/_s3_f{k}.yaml" if k else BASE_CFG
        raw = out / "val_raw_votes.json"
        if not raw.exists():
            log(f"fold{k}: val 推論（票数つき、約55分）")
            if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg,
                        "--weights", str(out / "model_final.pth"), "--split", "val",
                        "--fold", str(k), "--tta", "--tta-views", "8", "--tta-merge", "prob",
                        "--mask-bin-thr", "0.5", "--dump-raw", str(raw),
                        "--out", f"/tmp/vote_f{k}.csv"], f"vote_infer_f{k}.log"):
                return
        d = json.loads(raw.read_text())["preds"]
        merged.update(d)
        log(f"  fold{k}: 完了 {len(d)} 画像")

    log(f"OOF {len(merged)} 画像で α を探索")
    res = {}
    for a in ALPHAS:
        j = WD / f"oof_votes_a{a}.json"
        j.write_text(json.dumps({"preds": rescore(merged, a)}))
        o = WD / f"postproc_oof_votes_a{a}.json"
        if run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j), "--fold", "-1",
                "--out", str(o)] + GRID, f"vote_tune_a{a}.log"):
            d = json.loads(o.read_text())
            res[a] = d
            log(f"  α={a:<4}  PQ {d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
                f"thr={d['score_thr']} area={d['min_area']})")
    if 0.0 not in res:
        return
    base = res[0.0]["val_pq"]
    best = max(res, key=lambda a: res[a]["val_pq"])
    log("=== 結論 ===")
    log(f"  α=0（現行の再現）  {base:.4f}  ※既存の OOF 0.4711 と一致するはず")
    log(f"  最良 α={best}        {res[best]['val_pq']:.4f}  Δ={res[best]['val_pq'] - base:+.4f}")
    if res[best]["val_pq"] >= base + 0.005:
        log("  ★ 改善。2モデル確率平均融合にも適用して確認する")
    else:
        log("  改善は基準（+0.005）未満")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== TTA 票数スコア 終了 ====")
