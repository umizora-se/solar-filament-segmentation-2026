"""ラベル転写の効果を fold0 で判定する（取得完了後に自動で走る）。

比較（fold0 の元画像142枚・8方向TTA(prob統合)・後処理チューニング後）:
    現行  フル2048原寸                 fold0-val 0.4600
    今回  同上 + 近傍フレームへのラベル転写  fold0-val ?
変更点はラベル転写だけ（構成・iter数・TTA・後処理の探索範囲は同一）。

リーク対策:
  ・近傍は元画像と同じ fold に所属（filament_dataset で保証、assert あり）
  ・学習ログのレコード数が get_train_dicts_with_neighbors(0) と一致するか照合
  ・watchdog には fold を明示的に渡す（以前 --fold 0 固定で31時間を失った）
"""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "nb_summary.txt"
CFG = "configs/maskdino_filament_r50_full2048_nb.yaml"
OUT = WD / "r50_full2048_nb_f0"
BASE = 0.4600


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


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== ラベル転写 fold0 判定 {time.strftime('%m/%d %H:%M')} ====\n")
    log("近傍フレームの取得完了を待機")
    while subprocess.run(["pgrep", "-f", "gong_neighbors.py"], capture_output=True).returncode == 0:
        time.sleep(120)
    tail = (WD / "gong_neighbors.log").read_text(errors="ignore").splitlines()[-3:]
    for l in tail:
        log(f"  [取得] {l}")
    # 一時的な失敗（FTP のデータ接続が張れない 150/227 など）を1回だけ再試行する。
    # 取得済みは飛ばすので、失敗分と、一覧が取れなかった日の分だけが対象になる。
    log("再試行（失敗分のみ）")
    run(["python3", "-u", "src/gong_neighbors.py", "--window-min", "3", "--sleep", "2"],
        "gong_neighbors_retry.log", timeout=6 * 3600)
    for l in (WD / "gong_neighbors_retry.log").read_text(errors="ignore").splitlines()[-2:]:
        log(f"  [再試行] {l}")

    # ---- 品質管理 ----
    if not run(["python3", "-u", "src/gong_neighbors_qc.py"], "nb_qc.log", timeout=4 * 3600):
        return
    for l in (WD / "nb_qc.log").read_text().splitlines()[-5:]:
        log(f"  [qc] {l}")

    # ---- 期待レコード数（リーク照合用）----
    r = subprocess.run([sys.executable, "-c",
                        "import sys; sys.path.insert(0,'src');"
                        "from filament_dataset import get_train_dicts_with_neighbors as g;"
                        "print('N=', len(g(0)))"], cwd=REPO, capture_output=True, text=True)
    m = re.search(r"N= (\d+)", r.stdout)
    if not m:
        log(f"★ 期待レコード数を計算できない: {r.stderr[-300:]}")
        return
    want = int(m.group(1))
    log(f"fold0 の期待学習レコード数: {want}（元 928 + 近傍）")

    # ---- 動作確認（20 iter）----
    log("動作確認（20 iter）")
    if not run(["python3", "-u", "src/train_filament.py", "--config", CFG, "--fold", "0",
                "--smoke", "--output-dir", str(WD / "nb_smoke")], "nb_smoke.log", timeout=3600):
        return
    txt = (WD / "nb_smoke.log").read_text(errors="ignore")
    mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
    got = int(mm[-1]) if mm else -1
    if got != want:
        log(f"★★ 学習レコード数 {got} ≠ 期待 {want}。近傍データが使われていない／リークの疑い。中止")
        return
    log(f"  学習レコード数 {got}（期待 {want}）✓")

    # ---- 本学習（fold0、watchdog 経由・fold を明示）----
    if not (OUT / "model_final.pth").exists():
        log("本学習 fold0（約7時間）")
        if not run(["bash", "scripts/train_with_watchdog.sh", CFG, str(OUT),
                    str(WD / "nb_train_f0.log"), "5000", "0"], "nb_watchdog_f0.log"):
            return
    txt = (OUT / "log.txt").read_text(errors="ignore")
    mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
    got = int(mm[-1]) if mm else -1
    if got != want:
        log(f"★★ 本学習のレコード数 {got} ≠ 期待 {want}。中止")
        return
    log(f"学習完了  レコード数 {got} ✓")

    # ---- fold0-val 評価（元画像のみ）----
    raw = OUT / "val_raw_prob50.json"
    if not raw.exists():
        if not run(["python3", "-u", "src/predict_submit.py", "--config", CFG,
                    "--weights", str(OUT / "model_final.pth"), "--split", "val", "--fold", "0",
                    "--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                    "--dump-raw", str(raw), "--out", "/tmp/nb_val.csv"], "nb_infer.log"):
            return
    o = OUT / "postproc_prob50.json"
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", "0",
                "--out", str(o), "--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45",
                "--min-area", "100,200,300"], "nb_tune.log"):
        return
    d = json.loads(o.read_text())
    log("=== 結論 ===")
    log(f"  現行            fold0-val {BASE:.4f}")
    log(f"  + ラベル転写    fold0-val {d['val_pq']:.4f}  Δ={d['val_pq']-BASE:+.4f} "
        f"(SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} thr={d['score_thr']} area={d['min_area']})")
    if d["val_pq"] >= BASE + 0.005:
        log("  ★ 改善。5fold へ展開する価値あり")
    else:
        log("  改善は基準（+0.005）未満")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== ラベル転写 fold0 判定 終了 ====")
