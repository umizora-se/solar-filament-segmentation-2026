"""TTA 票数スコアを 1組目（5fold × 1モデル）に適用して test CSV を作り、提出する。

根拠（vote_stage.py, OOF 707枚）: α=0.75 で +0.0006 [95%: +0.0002, +0.0010]、4/5 fold で改善。
比較対象は 10/15 の提出 submission_ens5native_mv2.csv（OOF 0.4711 / LB 0.41）。

手順:
  1. fold k: test 推論（1モデル×8向き、票数つき）
  2. α=0 と α=0.75 でスコア変換 → 5fold マスク融合 mv2 → 各 α の OOF 後処理 → CSV
  3. α=0 の CSV が既存提出と一致することを確認（推論の再現性）
  4. 一致し、verify PASS なら α=0.75 の CSV を提出（ユーザー指示 11/10）
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "vote1_summary.txt"
ALPHA = 0.75
REF = REPO / "submission_ens5native_mv2.csv"
TTA = ["--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5"]


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
    return {s: [{"rle": it["rle"], "score": it["score"] * (it["votes"] / it["nv"]) ** alpha}
                for it in items] for s, items in preds.items()}


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 1組目への TTA 票数スコア適用と提出 {time.strftime('%m/%d %H:%M')} ====\n")

    raws = []
    for k in range(5):
        out = WD / f"r50_full2048_f{k}"
        cfg = f"configs/_s3_f{k}.yaml" if k else "configs/maskdino_filament_r50_full2048.yaml"
        traw = out / "test_raw_votes.json"
        if not traw.exists():
            log(f"fold{k}: test 推論（1モデル、票数つき）")
            if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg,
                        "--weights", str(out / "model_final.pth")] + TTA +
                       ["--dump-raw", str(traw), "--out", f"/tmp/vote1_test{k}.csv"],
                       f"vote1_tinfer_f{k}.log"):
                return
        raws.append(traw)
        log(f"  fold{k}: 完了")

    subs = {}
    for a in (0.0, ALPHA):
        per = []
        for k, traw in enumerate(raws):
            p = traw.with_name(f"test_raw_votes_a{a}.json")
            p.write_text(json.dumps({"preds": rescore(json.loads(traw.read_text())["preds"], a)}))
            per.append(str(p))
        j = WD / f"test_ens5_votes_a{a}_mv2.json"
        if not run(["python3", "-u", "src/ensemble.py", "--preds"] + per +
                   ["--out", str(j), "--min-votes", "2"], f"vote1_fuse_a{a}.log"):
            return
        sub = REPO / f"submission_ens5native_votes_a{a}_mv2.csv"
        pp = WD / f"postproc_oof_votes_a{a}.json"
        lf = f"vote1_csv_a{a}.log"
        if not run(["python3", "src/json_to_csv.py", "--preds", str(j), "--postproc", str(pp),
                    "--out", str(sub)], lf):
            return
        if "PASS" not in (WD / lf).read_text():
            log(f"★ CSV verify 失敗（α={a}）")
            return
        subs[a] = sub

    same = md5(subs[0.0]) == md5(REF)
    log(f"  照合: α=0 の CSV と既存提出 {REF.name} は{'一致 ✓' if same else '不一致'}")
    if not same:
        log("★ 推論が既存提出を再現しない → 提出を見送る（要確認）")
        return
    new = subs[ALPHA]
    for f in REPO.glob("submission*.csv"):
        if f != new and md5(f) == md5(new):
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    pq = json.loads((WD / f"postproc_oof_votes_a{ALPHA}.json").read_text())["val_pq"]
    msg = (f"NATIVE full-2048 5-fold x 1 seed + 8-view TTA (prob) + TTA vote score "
           f"(alpha={ALPHA}) + 5-fold mask fusion mv2 + OOF postproc. OOF-707 PQ {pq:.4f}")
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(new), "-m", msg[:480]], "vote1_submit.log", timeout=900):
        log(f"✅ 提出完了: {new.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 1組目への TTA 票数スコア適用と提出 終了 ====")
