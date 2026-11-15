"""5fold × 2シード（確率平均融合）の test 予測を作り、提出する。

OOF 707枚: 確率平均融合 0.4721 / 1組目単独 0.4711（差 +0.0010、ブートストラップで
上回る確率 79%、2組目単独に対しては 95%）。基準 +0.005 には届かないが、シードの
当たり外れを均す（private での安定）目的で最終候補にする。

手順は現行提出（submission_ens5native_mv2.csv）と同じで、違いは各 fold の予測が
「1モデル×8向き」から「2モデル×8向きの確率平均」になることだけ:
    fold k: predict_submit --weights 1組目_k 2組目_k --tta 8 --tta-merge prob
    5本を src/ensemble.py min_votes=2 で融合 → OOF で決めた後処理 → CSV
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "seed2_submit_summary.txt"
PP = WD / "postproc_oof_seed2_3.json"        # 確率平均融合の OOF 707枚で決めた後処理
MV = 2


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
        f.write(f"\n==== 5fold×2シード 確率平均融合の提出 {time.strftime('%m/%d %H:%M')} ====\n")
    pp = json.loads(PP.read_text())
    log(f"後処理（OOF 707枚で決定）: thr={pp['score_thr']} area={pp['min_area']} "
        f"OOF-PQ {pp['val_pq']:.4f}")

    raws = []
    for k in range(5):
        out = WD / f"r50_full2048_seed2_f{k}"
        traw = out / "test_raw_prob2m.json"
        if not traw.exists():
            log(f"fold{k}: test 推論 2モデル確率平均（180枚、約2時間）")
            if not run(["python3", "-u", "src/predict_submit.py", "--config",
                        str(REPO / "configs" / f"_seed2_f{k}.yaml"), "--weights",
                        str(WD / f"r50_full2048_f{k}" / "model_final.pth"),
                        str(out / "model_final.pth"), "--tta", "--tta-views", "8",
                        "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                        "--dump-raw", str(traw), "--out", f"/tmp/seed2p_t{k}.csv"],
                       f"seed2_ptinfer_f{k}.log"):
                return
        raws.append(str(traw))
        log(f"  fold{k}: 完了")

    j = WD / f"test_ens5x2p_mv{MV}.json"
    if not run(["python3", "-u", "src/ensemble.py", "--preds"] + raws +
               ["--out", str(j), "--min-votes", str(MV)], f"seed2_ptfuse_mv{MV}.log"):
        return
    sub = REPO / f"submission_ens5x2seed_prob_mv{MV}.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(j), "--postproc", str(PP),
                "--out", str(sub)], "seed2_csv.log"):
        return
    txt = (WD / "seed2_csv.log").read_text()
    if "PASS" not in txt:
        log("★ CSV verify 失敗 → 提出見送り")
        return
    for l in txt.splitlines():
        if "画像数" in l:
            log(f"  {l.strip()}")
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = (f"NATIVE full-2048 5-fold x 2 seeds: per-fold 2-model prob fusion (16 views) "
           f"+ 5-fold mask fusion (min_votes={MV}) + OOF postproc. OOF-707 PQ {pp['val_pq']:.4f}")
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg[:480]], "seed2_submit.log", timeout=900):
        log(f"✅ 提出完了: {sub.name}")
        time.sleep(180)
        r = subprocess.run(["kaggle", "competitions", "submissions",
                            "-c", "filament-segmentation-2026"],
                           capture_output=True, text=True, timeout=300)
        for l in r.stdout.splitlines()[:4]:
            log("  " + l[:140])


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 5fold×2シード 提出 終了 ====")
