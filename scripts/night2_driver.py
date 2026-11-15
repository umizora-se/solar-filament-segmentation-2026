"""夜間パイプライン 2026-09-08: SAM 適合性 → U-Net 検証器 → 最良構成 → (改善時) 提出。
ユーザー事前承認: 「はい。進めてください。提出までお願いします」
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "overnight2_summary.txt"
BASE4, BASE8 = 0.4198, 0.4268   # 現行基準（4方向 / 8方向）
VAL4 = WD / "r50_f0" / "val_raw_tta.json"
VER_OPS = {}   # tag -> (mode, sem_thr)  文字列パースを避けるための正典
VAL8 = WD / "r50_f0" / "val_raw_tta8.json"


def log(msg):
    line = f"[{time.strftime('%m/%d %H:%M')}] {msg}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, logfile, timeout=4 * 3600):
    with open(WD / logfile, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT,
                           timeout=timeout)
    if r.returncode != 0:
        log(f"★失敗 rc={r.returncode}: {' '.join(cmd[:4])}...  ({logfile} 参照)")
        return False
    return True


def tune(preds_json, tag):
    """tune_postproc を回して val-PQ を返す（失敗時 None）。"""
    out = str(Path(preds_json).with_name(f"postproc_{tag}.json"))
    ok = run(["python3", "-u", "src/tune_postproc.py", "--preds", str(preds_json),
              "--fold", "0", "--out", out,
              "--score-thr", "0.25,0.30,0.35,0.40", "--min-area", "100,200,300"],
             f"n2_tune_{tag}.log")
    if not ok:
        return None, None
    d = json.loads(Path(out).read_text())
    return d["val_pq"], out


def sam_ckpts():
    """DL完了を待ち、比較対象の SAM 重みリストを返す。"""
    for _ in range(60):                          # 最大30分待つ
        if subprocess.run(["pgrep", "-f", "curl.*sam"],
                          capture_output=True).returncode != 0:
            break
        time.sleep(30)
    out = []
    h = REPO / "weights" / "sam_vit_h_4b8939.pth"
    b = REPO / "weights" / "sam_vit_b_01ec64.pth"
    s2 = REPO / "weights" / "sam2.1_hiera_large.pt"
    if h.exists() and h.stat().st_size > 2_500_000_000:
        out.append(str(h))
    elif b.exists():
        out.append(str(b))
    if s2.exists() and s2.stat().st_size > 800_000_000:
        out.append(str(s2))
    return out


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== 夜間パイプライン2 開始 {time.strftime('%m/%d %H:%M')} ====\n")
    cands = {}   # name -> (val_pq, json_path)

    # ---- Phase A: SAM 適合性（GT箱, 500件, {vit_h, sam2.1} × {raw, clahe}）----
    ckpts = sam_ckpts()
    log(f"PhaseA: SAM 1a 適合性テスト（候補: {[Path(c).name for c in ckpts]}）")
    ckpt, best_mode, best_iou = None, None, -1
    for ck in ckpts:
        for mode in ["raw", "clahe"]:
            tag = f"{Path(ck).stem.split('_')[0]}_{mode}"
            o = WD / f"sam_gt_{tag}.json"
            if run(["python3", "-u", "src/sam_probe.py", "--mode", "gt", "--fold", "0",
                    "--limit", "500", "--ckpt", ck, "--img-mode", mode, "--out", str(o)],
                   f"n2_samgt_{tag}.log"):
                d = json.loads(o.read_text())
                log(f"  {Path(ck).name} / {mode}: 平均IoU={d['mean_iou']:.4f} "
                    f"IoU>0.5率={d['hit_rate']:.3f} (n={d['n']})  [MaskDINO TP≒0.686]")
                if d["mean_iou"] > best_iou:
                    ckpt, best_mode, best_iou = ck, mode, d["mean_iou"]
    sam_ok = ckpt is not None and best_iou >= 0.60
    log(f"PhaseA 判定: best={Path(ckpt).name if ckpt else '-'}/{best_mode} "
        f"IoU={best_iou:.4f} → " + ("1bへ進む" if sam_ok else "SAM閉鎖（ドメイン不適合）"))

    # ---- Phase B: SAM でマスク描き直し（val 4view）----
    if sam_ok:
        log("PhaseB: SAM 1b マスク置換（val 4view）")
        if run(["python3", "-u", "src/sam_probe.py", "--mode", "preds",
                "--preds", str(VAL4), "--ckpt", ckpt, "--img-mode", best_mode],
               "n2_sampreds.log"):
            for tag in ["samrep", "samgd"]:
                j = VAL4.with_name(VAL4.stem + f"_{tag}.json")
                pq, _ = tune(j, tag)
                if pq is not None:
                    log(f"  {tag}: val-PQ {pq:.4f} (基準 {BASE4})")
                    cands[tag] = (pq, str(j))

    # ---- Phase C: U-Net 学習 → 検証器 ----
    log("PhaseC: セマンティックU-Net学習（20ep）")
    if run(["python3", "-u", "src/train_semseg.py", "--fold", "0",
            "--out", "work_dirs/semseg_f0", "--epochs", "20"], "n2_semseg_train.log"):
        tail = subprocess.run(["grep", "-a", "best", str(WD / "n2_semseg_train.log")],
                              capture_output=True, text=True).stdout.strip().splitlines()
        log(f"  学習完了: {tail[-1] if tail else '?'}")
        probs = WD / "semseg_f0" / "probs_val.npz"
        if run(["python3", "-u", "src/semseg_infer.py",
                "--weights", "work_dirs/semseg_f0/semseg_best.pth",
                "--fold", "0", "--split", "val", "--out", str(probs)], "n2_semseg_inf.log"):
            for mode, thr in [("gate", 0.05), ("gate", 0.10), ("gate", 0.20),
                              ("mul", 0), ("sqrt", 0)]:
                tag = f"v{mode}{thr if mode == 'gate' else ''}".replace(".", "")
                VER_OPS[tag] = (mode, thr)
                j = VAL4.with_name(VAL4.stem + f"_{tag}.json")
                cmd = ["python3", "src/apply_verifier.py", "--preds", str(VAL4),
                       "--probs", str(probs), "--mode", mode, "--out", str(j)]
                if mode == "gate":
                    cmd += ["--sem-thr", str(thr)]
                if run(cmd, f"n2_ver_{tag}.log"):
                    pq, _ = tune(j, tag)
                    if pq is not None:
                        log(f"  検証器 {tag}: val-PQ {pq:.4f} (基準 {BASE4})")
                        cands[tag] = (pq, str(j))

    # ---- Phase D: 組み合わせ ----
    sam_best = max((k for k in cands if k.startswith("sam")),
                   key=lambda k: cands[k][0], default=None)
    ver_best = max((k for k in cands if k.startswith("v")),
                   key=lambda k: cands[k][0], default=None)
    if (sam_best and ver_best and cands[sam_best][0] > BASE4
            and cands[ver_best][0] > BASE4):
        log(f"PhaseD: 組み合わせ {sam_best} + {ver_best}")
        mode, thr = VER_OPS[ver_best]
        j = VAL4.with_name(VAL4.stem + "_combo.json")
        cmd = ["python3", "src/apply_verifier.py", "--preds", cands[sam_best][1],
               "--probs", str(WD / "semseg_f0" / "probs_val.npz"),
               "--mode", mode, "--out", str(j)]
        if mode == "gate":
            cmd += ["--sem-thr", str(thr)]
        if run(cmd, "n2_combo.log"):
            pq, _ = tune(j, "combo")
            if pq is not None:
                log(f"  combo: val-PQ {pq:.4f}")
                cands["combo"] = (pq, str(j))

    if not cands:
        log("候補ゼロ（全フェーズ失敗）。提出見送り")
        return
    name = max(cands, key=lambda k: cands[k][0])
    pq4 = cands[name][0]
    log(f"PhaseD 結論: 最良 {name} val-PQ(4view) {pq4:.4f} vs 基準 {BASE4}")

    # ---- Phase E: 改善時のみ 8view で最終確認 → test → 提出 ----
    if pq4 <= BASE4 + 0.001:
        log("改善なし → 提出は行わない（既存 LB 0.39 が最良のまま。枠温存）")
        return

    log(f"PhaseE: 8方向TTAへ {name} を適用して最終確認")
    ops = []                                        # (kind, param)
    if name in ("samrep", "samgd", "combo") and sam_best:
        ops.append(("sam", "samgd" if "gd" in (sam_best if name == "combo" else name)
                    else "samrep"))
    if name.startswith("v") or name == "combo":
        vb = ver_best if name == "combo" else name
        ops.append(("ver", vb))

    cur = VAL8
    for kind, p in ops:
        if kind == "sam":
            if not run(["python3", "-u", "src/sam_probe.py", "--mode", "preds",
                        "--preds", str(cur), "--ckpt", ckpt, "--img-mode", best_mode],
                       "n2_sam8.log"):
                return
            cur = Path(str(cur).replace(".json", f"_{p}.json"))
        else:
            mode, thr = VER_OPS[p]
            j = VAL8.with_name(VAL8.stem + f"_{p}.json")
            cmd = ["python3", "src/apply_verifier.py", "--preds", str(cur),
                   "--probs", str(WD / "semseg_f0" / "probs_val.npz"),
                   "--mode", mode, "--out", str(j)]
            if mode == "gate":
                cmd += ["--sem-thr", str(thr)]
            if not run(cmd, "n2_ver8.log"):
                return
            cur = j
    pq8, pp8 = tune(cur, "final8")
    if pq8 is None:
        return
    log(f"  最終形(8view+{name}): val-PQ {pq8:.4f} vs 基準 {BASE8}")
    if pq8 <= BASE8:
        log("8方向では基準未達 → 提出見送り")
        return

    # ---- Phase F: test 推論 → CSV → 提出 ----
    log("PhaseF: test 8方向TTA 推論（約1.5-2時間）")
    traw = WD / "r50_f0" / "test_raw_tta8.json"
    if not run(["python3", "-u", "src/predict_submit.py",
                "--weights", "work_dirs/r50_f0/model_final.pth",
                "--tta", "--tta-views", "8", "--dump-raw", str(traw),
                "--out", "/tmp/night_test.csv"], "n2_test_infer.log", timeout=4 * 3600):
        return
    cur = traw
    for kind, p in ops:
        if kind == "sam":
            if not run(["python3", "-u", "src/sam_probe.py", "--mode", "preds",
                        "--preds", str(cur), "--ckpt", ckpt, "--img-mode", best_mode],
                       "n2_samtest.log"):
                return
            cur = Path(str(cur).replace(".json", f"_{p}.json"))
        else:
            probs_t = WD / "semseg_f0" / "probs_test.npz"
            if not probs_t.exists():
                if not run(["python3", "-u", "src/semseg_infer.py",
                            "--weights", "work_dirs/semseg_f0/semseg_best.pth",
                            "--split", "test", "--out", str(probs_t)], "n2_seminf_t.log"):
                    return
            mode, thr = VER_OPS[p]
            j = traw.with_name(traw.stem + f"_{p}.json")
            cmd = ["python3", "src/apply_verifier.py", "--preds", str(cur),
                   "--probs", str(probs_t), "--mode", mode, "--out", str(j)]
            if mode == "gate":
                cmd += ["--sem-thr", str(thr)]
            if not run(cmd, "n2_vertest.log"):
                return
            cur = j
    sub = REPO / "submission_night.csv"
    if not run(["python3", "src/json_to_csv.py", "--preds", str(cur),
                "--postproc", pp8, "--out", str(sub)], "n2_csv.log"):
        return
    if "PASS" not in (WD / "n2_csv.log").read_text():
        log("★ CSV verify 失敗 → 提出見送り")
        return
    md5 = hashlib.md5(sub.read_bytes()).hexdigest()
    for f in REPO.glob("submission*.csv"):
        if f != sub and hashlib.md5(f.read_bytes()).hexdigest() == md5:
            log(f"★ {f.name} と同一内容 → 提出見送り")
            return
    msg = f"MaskDINO R50 + 8-view TTA + {name} (SAM/semseg-verifier) val-PQ {pq8:.4f}"
    if run(["kaggle", "competitions", "submit", "-c", "filament-segmentation-2026",
            "-f", str(sub), "-m", msg], "n2_submit.log", timeout=600):
        log(f"✅ 提出完了: {msg}")
    else:
        log("★ 提出コマンド失敗")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ ドライバ例外: {e!r}")
        raise
    finally:
        log("==== 夜間パイプライン2 終了 ====")
