"""予測統合のパラメータを OOF 707枚で探索し、+0.005 以上なら提出まで進む。"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "merge_summary.txt"
OOF = WD / "oof_raw_tta8_sqrt.json"          # 検証器(sqrt)適用済みのOOF予測
TEST8 = WD / "r50_f0" / "test_raw_tta8.json"
PROBS_TEST = WD / "probs_test_ens.npz"       # 現行3種（LB0.40 と同じ構成）
BASE = 0.4228
GRID = [(20, 0.06), (20, 0.09), (35, 0.06), (35, 0.09), (35, 0.12),
        (50, 0.09), (50, 0.12), (65, 0.12)]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=6 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 予測統合の探索 {time.strftime('%m/%d %H:%M')} ====\n")
    log(f"基準: 統合なし OOF-PQ {BASE}（上限は診断より +0.013）")
    best = (BASE, None, None)
    for gap, fill in GRID:
        tag = f"g{int(gap)}f{int(fill*100)}"
        j = WD / f"oof_merged_{tag}.json"
        if not run(["python3", "-u", "src/merge_preds.py", "--preds", str(OOF),
                    "--gap", str(gap), "--fill", str(fill), "--out", str(j)],
                   f"mg_merge_{tag}.log"):
            continue
        nmerge = (WD / f"mg_merge_{tag}.log").read_text()
        n = nmerge.split("(")[-1].split(" 本")[0] if "(" in nmerge else "?"
        o = WD / f"postproc_merged_{tag}.json"
        if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(j),
                    "--fold", "-1", "--out", str(o),
                    "--score-thr", "0.25,0.30,0.35", "--min-area", "100,200,300"],
                   f"mg_tune_{tag}.log"):
            continue
        d = json.loads(o.read_text())
        log(f"  gap<={gap} fill>={fill}: OOF-PQ {d['val_pq']:.4f} "
            f"Δ={d['val_pq']-BASE:+.4f}  統合{n}本 (thr={d['score_thr']} area={d['min_area']})")
        if d["val_pq"] > best[0]:
            best = (d["val_pq"], (gap, fill), str(o))

    if best[1] is None:
        log("=== 結論: 全条件で改善なし → 統合は却下 ===")
        return
    gap, fill = best[1]
    delta = best[0] - BASE
    log(f"=== 最良: gap<={gap} fill>={fill}  OOF-PQ {best[0]:.4f}  Δ={delta:+.4f} ===")
    if delta < 0.005:
        log("判定: +0.005 に届かず → 提出しない（現行維持）")
        return

    log("★ 基準を満たした → test 側に適用して提出")
    jt = WD / "test_raw_tta8_ver.json"
    if not run(["python3", "src/apply_verifier.py", "--preds", str(TEST8),
                "--probs", str(PROBS_TEST), "--mode", "sqrt", "--out", str(jt)],
               "mg_vertest.log"):
        return
    jm = WD / "test_merged.json"
    if not run(["python3", "-u", "src/merge_preds.py", "--preds", str(jt),
                "--gap", str(gap), "--fill", str(fill), "--out", str(jm)], "mg_mergetest.log"):
        return
    sub = REPO / "submission_merge.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jm),
                "--postproc", best[2], "--out", str(sub)], "mg_csv.log"):
        return
    if "PASS" not in (WD / "mg_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = (f"MaskDINO 8-view + verifier(sqrt) + prediction merging "
           f"(gap<={gap}, fill>={fill}) OOF-PQ {best[0]:.4f}")
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg[:480]], "mg_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 予測統合の探索 終了 ====")
