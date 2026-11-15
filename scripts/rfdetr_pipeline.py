"""RF-DETR 学習完了 → val推論 → MaskDINO と融合 → 検証器適用 → 改善時のみ提出。"""
import hashlib
import json
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "rfdetr_summary.txt"
MD_VAL8 = WD / "r50_f0" / "val_raw_tta8.json"
MD_TEST8 = WD / "r50_f0" / "test_raw_tta8.json"
PROBS_VAL = WD / "probs_val_ens.npz"
PROBS_TEST = WD / "probs_test_ens.npz"
BEST = 0.4322          # 現行最良（8view + 検証器アンサンブル sqrt）


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=6 * 3600):
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:3])}… ({lf})")
    return r.returncode == 0


def tune(js, tag):
    out = str(Path(js).with_name(f"postproc_{tag}.json"))
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(js), "--fold", "0",
                "--out", out, "--score-thr", "0.20,0.25,0.30,0.35,0.40",
                "--min-area", "100,200,300"], f"rf_tune_{tag}.log"):
        return None, None
    return json.loads(Path(out).read_text())["val_pq"], out


def find_ckpt():
    d = WD / "rfdetr_f0"
    for pat in ["checkpoint_best_ema.pth", "checkpoint_best_total.pth",
                "checkpoint_best_regular.pth", "checkpoint.pth"]:
        p = d / pat
        if p.exists():
            return str(p)
    c = sorted(d.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return str(c[-1]) if c else None


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== RF-DETR パイプライン {time.strftime('%m/%d %H:%M')} ====\n")
    log("学習の完了を待機中")
    while subprocess.run(["pgrep", "-f", "train_rfdetr"], capture_output=True).returncode == 0:
        time.sleep(120)
    ck = find_ckpt()
    if not ck:
        log("★ チェックポイントが見つからない。終了")
        return
    lines = [l for l in (WD / "rfdetr_train.log").read_text(errors="ignore").splitlines()
             if "Best EMA metric" in l]
    log(f"学習完了: {ck}")
    if lines:
        log(f"  {lines[-1].split('] ')[-1]}")

    # --- val 推論 ---
    rf_val = WD / "rfdetr_f0" / "val_raw.json"
    if not run(["python3", "-u", "src/rfdetr_infer.py", "--weights", ck,
                "--split", "val", "--fold", "0", "--out", str(rf_val)], "rf_val_infer.log"):
        return
    pq_rf, _ = tune(rf_val, "rfonly")
    log(f"RF-DETR 単体: val-PQ {pq_rf:.4f}  (MaskDINO 8view 0.4268)")

    # --- MaskDINO と融合（min_votes 探索）---
    cands = {}
    for mv in [1, 2]:
        j = WD / "rfdetr_f0" / f"ens_mv{mv}.json"
        if not run(["python3", "-u", "src/ensemble.py", "--preds", str(MD_VAL8),
                    str(rf_val), "--out", str(j), "--min-votes", str(mv)],
                   f"rf_ens_mv{mv}.log"):
            continue
        pq, _ = tune(j, f"mv{mv}")
        if pq:
            log(f"  融合 min_votes={mv}: val-PQ {pq:.4f}")
            cands[f"mv{mv}"] = (pq, str(j))
        # 検証器も重ねる
        jv = WD / "rfdetr_f0" / f"ens_mv{mv}_ver.json"
        if PROBS_VAL.exists() and run(
                ["python3", "src/apply_verifier.py", "--preds", str(j),
                 "--probs", str(PROBS_VAL), "--mode", "sqrt", "--out", str(jv)],
                f"rf_ver_mv{mv}.log"):
            pq2, _ = tune(jv, f"mv{mv}ver")
            if pq2:
                log(f"  融合 min_votes={mv} + 検証器: val-PQ {pq2:.4f}")
                cands[f"mv{mv}ver"] = (pq2, str(jv))

    if not cands:
        log("融合候補ゼロ。終了")
        return
    best = max(cands, key=lambda k: cands[k][0])
    pq = cands[best][0]
    log(f"最良: {best} val-PQ {pq:.4f}  (現行最良 {BEST})")
    if pq <= BEST + 0.0005:
        log("現行最良を超えず → 提出見送り（枠温存）")
        return

    # --- test 側を同じ手順で作る ---
    rf_test = WD / "rfdetr_f0" / "test_raw.json"
    if not run(["python3", "-u", "src/rfdetr_infer.py", "--weights", ck,
                "--split", "test", "--out", str(rf_test)], "rf_test_infer.log"):
        return
    mv = 1 if best.startswith("mv1") else 2
    jt = WD / "rfdetr_f0" / f"test_ens_mv{mv}.json"
    if not run(["python3", "-u", "src/ensemble.py", "--preds", str(MD_TEST8),
                str(rf_test), "--out", str(jt), "--min-votes", str(mv)], "rf_enstest.log"):
        return
    if best.endswith("ver"):
        jt2 = WD / "rfdetr_f0" / f"test_ens_mv{mv}_ver.json"
        if not run(["python3", "src/apply_verifier.py", "--preds", str(jt),
                    "--probs", str(PROBS_TEST), "--mode", "sqrt", "--out", str(jt2)],
                   "rf_vertest.log"):
            return
        jt = jt2
    _, pp = tune(cands[best][1], f"{best}_final")
    sub = REPO / "submission_rfdetr.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(jt),
                "--postproc", pp, "--out", str(sub)], "rf_csv.log"):
        return
    if "PASS" not in (WD / "rf_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一 → 提出見送り")
            return
    msg = f"MaskDINO 8-view + RF-DETR-Seg ensemble ({best}) val-PQ {pq:.4f}"
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg], "rf_submit.log", timeout=900):
        log(f"✅ 提出完了: {msg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== RF-DETR パイプライン 終了 ====")
