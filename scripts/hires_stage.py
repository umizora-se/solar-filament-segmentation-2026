"""マスクの高解像度化（stride2 + stem の実特徴の skip 接続）を fold0 で判定する。

原寸2048での学習は、デコーダー各層が全クエリ分の stride2 マスクを保持するため
メモリに収まらない（実測 OOM、クエリ50でも不可）。そこで学習は原寸1024クロップで行い、
推論は従来通りフル2048で行う。クロップ化の影響を切り分けるため、高解像度化なしの
1024クロップ版も同条件で学習し、両者の差を高解像度化の効果とする。

    A: 原寸1024クロップ                    configs/maskdino_filament_r50_n1024.yaml
    B: 原寸1024クロップ + 高解像度マスク     configs/maskdino_filament_r50_n1024_hires.yaml
    （差分は FILAMENT.HIRES_MASK / HIRES_SKIP の2行だけ）

判定:
    高解像度化の効果 = B − A （fold0-val、元画像142枚・8方向TTA(prob)・後処理チューニング後）
    参考: 現行の最良（フル2048学習）fold0-val 0.4600
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "hires_summary.txt"
BASE_FULL = 0.4600
WANT_RECORDS = 928
RUNS = [("A_n1024", "configs/maskdino_filament_r50_n1024.yaml", WD / "r50_n1024_f0"),
        ("B_n1024_hires", "configs/maskdino_filament_r50_n1024_hires.yaml", WD / "r50_n1024_hires_f0")]


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=20 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def train_and_eval(tag, cfg, out):
    if not (out / "model_final.pth").exists():
        log(f"{tag}: 学習 fold0")
        if not run(["bash", "scripts/train_with_watchdog.sh", cfg, str(out),
                    str(WD / f"hires_train_{tag}.log"), "5000", "0"],
                   f"hires_watchdog_{tag}.log", timeout=24 * 3600):
            return None
    txt = (out / "log.txt").read_text(errors="ignore")
    mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
    got = int(mm[-1]) if mm else -1
    if got != WANT_RECORDS:
        log(f"★★ {tag}: 学習レコード数 {got} ≠ 期待 {WANT_RECORDS}。中止")
        return None
    log(f"{tag}: 学習完了  レコード数 {got} ✓")
    raw = out / "val_raw_prob50.json"
    if not raw.exists():
        if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg,
                    "--weights", str(out / "model_final.pth"), "--split", "val", "--fold", "0",
                    "--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                    "--dump-raw", str(raw), "--out", f"/tmp/hires_{tag}.csv"],
                   f"hires_infer_{tag}.log"):
            return None
    o = out / "postproc_prob50.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], f"hires_tune_{tag}.log"):
        return None
    d = json.loads(o.read_text())
    log(f"{tag}: fold0-val {d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
        f"thr={d['score_thr']} area={d['min_area']})")
    return d


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 高解像度マスク fold0 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    res = {}
    for tag, cfg, out in RUNS:
        d = train_and_eval(tag, cfg, out)
        if d is None:
            return
        res[tag] = d
    a, b = res["A_n1024"], res["B_n1024_hires"]
    log("=== 結論 ===")
    log(f"  A 原寸1024クロップ              {a['val_pq']:.4f}  SQ={a['val_sq']:.3f}")
    log(f"  B 原寸1024クロップ + 高解像度    {b['val_pq']:.4f}  SQ={b['val_sq']:.3f}")
    log(f"  高解像度化の効果（B−A）         {b['val_pq']-a['val_pq']:+.4f}  SQ {b['val_sq']-a['val_sq']:+.3f}")
    log(f"  参考: フル2048学習（現行）       {BASE_FULL:.4f}  → B との差 {b['val_pq']-BASE_FULL:+.4f}")
    if b["val_pq"] - a["val_pq"] >= 0.005:
        log("  ★ 高解像度化に効果あり")
    else:
        log("  高解像度化の効果は基準（+0.005）未満")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== 高解像度マスク fold0 判定 終了 ====")
