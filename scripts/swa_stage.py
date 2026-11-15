"""チェックポイントの重み平均（SWA: Izmailov et al. 2018）の効果を OOF 707枚で判定する（1組目・5fold）。

狙い: 汎化性能。学習の終盤に保存した重みを平均すると、損失の平坦な領域の中心に近づき、
未知データでの性能が上がりやすい。追加の学習は不要で、推論だけで検証できる。
1組目は 250 iter 毎のチェックポイントが残っている（学習率は 4000 iter で 1/10）。
バックボーンは FrozenBN、ヘッドは GN なので、平均後に BN 統計を取り直す必要はない。

  A（本命・事前に決定）: 4249/4499/4749/4999  学習率を下げた後の低学習率区間
  B（参考）            : 3749〜4999 の 6個      学習率を下げる前も含む

比較: 1組目（最終重み・同じ推論条件）OOF 0.4711。採否は fold 別の差と画像ブートストラップ（ルール1）。
"""
import json
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "swa_summary.txt"
BASE_OOF = 0.4711
VARIANTS = {"A": [4249, 4499, 4749, 4999], "B": [3749, 3999, 4249, 4499, 4749, 4999]}
TTA = ["--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5"]
GRID = ["--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45", "--min-area", "100,200,300"]


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


def average(k, its, dst):
    if dst.exists():
        return
    src = WD / f"r50_full2048_f{k}"
    acc = None
    for it in its:
        m = torch.load(src / f"model_{it:07d}.pth", map_location="cpu", weights_only=False)["model"]
        if acc is None:
            acc = {n: v.double().clone() for n, v in m.items()}
        else:
            for n, v in m.items():
                acc[n] += v.double()
    out = {n: (v / len(its)).to(torch.float32) for n, v in acc.items()}
    torch.save({"model": out, "iteration": its[-1]}, dst)


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== SWA（重み平均）判定 {time.strftime('%m/%d %H:%M')} ====\n")
    res = {}
    for v, its in VARIANTS.items():
        merged = {}
        for k in range(5):
            out = WD / f"r50_full2048_f{k}"
            w = out / f"model_swa{v}.pth"
            average(k, its, w)
            cfg = f"configs/_s3_f{k}.yaml" if k else "configs/maskdino_filament_r50_full2048.yaml"
            raw = out / f"val_raw_swa{v}.json"
            if not raw.exists():
                log(f"SWA-{v} fold{k}: val 推論（約20分）")
                if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg, "--weights", str(w),
                            "--split", "val", "--fold", str(k)] + TTA +
                           ["--dump-raw", str(raw), "--out", f"/tmp/swa{v}_f{k}.csv"], f"swa{v}_infer_f{k}.log"):
                    return
            merged.update(json.loads(raw.read_text())["preds"])
        j = WD / f"oof_swa{v}.json"
        j.write_text(json.dumps({"preds": {s: [{"rle": it["rle"], "score": it["score"]} for it in items]
                                           for s, items in merged.items()}}))
        o = WD / f"postproc_oof_swa{v}.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j), "--fold", "-1",
                    "--out", str(o)] + GRID, f"swa{v}_tune_oof.log"):
            return
        d = json.loads(o.read_text())
        res[v] = d
        log(f"  SWA-{v} OOF {d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
            f"thr={d['score_thr']} area={d['min_area']})  Δ={d['val_pq'] - BASE_OOF:+.4f}  [{len(merged)}枚]")
    log("=== 結論 ===")
    log(f"  1組目（最終重み）  {BASE_OOF:.4f}")
    for v, d in res.items():
        log(f"  SWA-{v} {VARIANTS[v]}  {d['val_pq']:.4f}  Δ={d['val_pq'] - BASE_OOF:+.4f}")
    log("  採否は fold 別の差と画像ブートストラップで判断する（DECISION_RULES ルール1）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== SWA（重み平均）判定 終了 ====")
