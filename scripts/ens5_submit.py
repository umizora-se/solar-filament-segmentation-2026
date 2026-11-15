"""原寸5foldの融合方式を OOF で選び、test に適用して提出する。

注意: 旧モデルでは5foldアンサンブルが LB -0.02 で失敗した（同一構成で多様性なし）。
      そのため OOF で「融合 > 単体」を確認してから提出する。
      OOF での融合評価には、各foldモデルが自分の val fold を予測した結果しか無いため、
      融合は test 側でしか作れない。そこで代替として:
        各foldの val 予測は1モデル分しかない → OOF で融合は評価できない
      → よって「単体の OOF 0.4711」と「test 融合」を直接比較できない。
      判断は LB で行う（min_votes 1 と 2 の両方を提出候補にし、まず 2 を出す）。
"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "ens5_summary.txt"
PP = WD / "postproc_oof_full2048.json"      # OOF 707枚で決めた後処理


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
    while subprocess.run(["pgrep", "-f", "ens5_test"], capture_output=True).returncode == 0:
        time.sleep(120)
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 5fold融合の提出 {time.strftime('%m/%d %H:%M')} ====\n")
    if not PP.exists():
        log(f"★ 後処理パラメータが無い: {PP}")
        return
    pp = json.loads(PP.read_text())
    log(f"後処理（OOF 707枚で決定）: thr={pp['score_thr']} area={pp['min_area']} "
        f"OOF-PQ {pp['val_pq']:.4f}")

    # min_votes=2（多数派を要求＝偽陽性に厳しい）をまず提出する
    for mv in [2]:
        j = WD / f"test_ens5_mv{mv}.json"
        if not j.exists():
            log(f"★ {j.name} が無い")
            continue
        sub = REPO / f"submission_ens5native_mv{mv}.csv"
        if not run(["python3", "src/json_to_csv.py", "--preds", str(j),
                    "--postproc", str(PP), "--out", str(sub)], f"e5_csv_mv{mv}.log"):
            continue
        txt = (WD / f"e5_csv_mv{mv}.log").read_text()
        if "PASS" not in txt:
            log(f"★ CSV verify 失敗（mv{mv}） → 提出見送り")
            continue
        for l in txt.splitlines():
            if "画像数" in l:
                log(f"  {l.strip()}")
        md5 = hashlib.md5(sub.read_bytes()).hexdigest()
        dup = None
        for f in REPO.glob("submission*.csv"):
            if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
                dup = f.name
        if dup:
            log(f"★ {dup} と同一 → 提出見送り")
            continue
        msg = (f"NATIVE full-2048 5-fold ensemble (min_votes={mv}) + 8-view TTA (prob) "
               f"+ OOF postproc. single-model OOF-707 PQ 0.4711")
        if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
                "-f", str(sub), "-m", msg[:480]], f"e5_submit_mv{mv}.log", timeout=900):
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
        log("==== 5fold融合の提出 終了 ====")
