"""原寸5foldモデルで test 推論 → アンサンブル → 提出候補を作る。

注意: 旧モデルでは5foldアンサンブルが LB -0.02 と失敗した（同一構成で多様性がない）。
      原寸モデルでも同じ可能性があるため、以下を両方作って OOF で比較してから選ぶ:
        ・各foldモデル単体の test 予測
        ・min_votes 1/2 の融合
"""
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "ens5_summary.txt"


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
        f.write(f"\n==== 原寸5fold の test 推論 {time.strftime('%m/%d %H:%M')} ====\n")
    log("OOF 707枚 = 0.4694（検出器のみ 0.4711）/ 現行 LB 0.40")

    raws = []
    for k in range(5):
        out = WD / f"r50_full2048_f{k}"
        ck = out / "model_final.pth"
        if not ck.exists():
            log(f"★ fold{k}: model_final.pth が無い")
            continue
        cfg = f"configs/_s3_f{k}.yaml" if k else "configs/maskdino_filament_r50_full2048.yaml"
        if not (REPO / cfg).exists():
            cfg = "configs/maskdino_filament_r50_full2048.yaml"
        traw = out / "test_raw_prob50.json"
        if not traw.exists():
            log(f"fold{k}: test 推論（180枚、約25分）")
            if not run(["python3", "-u", "src/predict_submit.py", "--config", cfg,
                        "--weights", str(ck), "--tta", "--tta-views", "8",
                        "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                        "--dump-raw", str(traw), "--out", f"/tmp/e5_f{k}.csv"],
                       f"e5_infer_f{k}.log"):
                continue
        raws.append(str(traw))
        log(f"  fold{k}: 完了")

    if len(raws) < 3:
        log("★ 揃ったモデルが少なすぎる。終了")
        return
    log(f"test 予測 {len(raws)} 本を融合")
    for mv in [1, 2]:
        j = WD / f"test_ens5_mv{mv}.json"
        if run(["python3", "-u", "src/ensemble.py", "--preds"] + raws +
               ["--out", str(j), "--min-votes", str(mv)], f"e5_fuse_mv{mv}.log"):
            log(f"  min_votes={mv}: {j.name}")
    log("※ 採否は OOF での融合評価と合わせて判断する（単独提出はまだしない）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== test 推論 終了 ====")
