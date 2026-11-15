"""シード違いの2組目（全5fold）を学習し、2組融合の効果を OOF 707枚で判定する。

根拠: 高解像度化の判定で SQ は頭打ち（+0.001）、伸びしろは RQ（検出）側と分かった。
      同一構成を別シードで学習した2モデルの融合は、検出のばらつきを均して RQ を上げる。

構成は現行（フル2048原寸・クエリ150・実効バッチ8・5000 iter）と完全に同一で、
違いは SEED を固定値にしたことだけ（現行1組目は detectron2 が乱数で決めたシード）。

判定（OOF 707枚・8方向TTA(prob)・後処理チューニング後）:
    1組目単独                 0.4711
    2組目単独                 ?   （シードによるばらつきの目安）
    1組目+2組目 融合（fold毎）  ?   ← +0.005 以上で採用
      ・マスク融合: 0/1 マスクを束ねて票決（src/ensemble.py）
      ・確率平均融合: 2モデル×8向きを16ビューとして確率を平均（predict_submit --weights 2本）
        fold0 でマスク融合が2モデル平均止まり（0.4570）だったため 11/06 に追加
採用なら test 予測も作る（提出はしない）。
"""
import json
import re
import subprocess
import time
from pathlib import Path

REPO = Path(".")
WD = REPO / "work_dirs"
SUMMARY = WD / "seed_summary.txt"
BASE_CFG = REPO / "configs/maskdino_filament_r50_full2048.yaml"
SEED = 20261105
BASE_OOF = 0.4711
EXPECT_N = {0: 928, 1: 939, 2: 920, 3: 915, 4: 914}
GRID = ["--score-thr", "0.20,0.25,0.30,0.35,0.40,0.45", "--min-area", "100,200,300"]


# 学習プロセスだけに一致させる（先頭一致）。部分一致だと、この文字列を含む
# 別のシェルのコマンド行にも一致して永久に待ってしまう（11/06 19:23〜 実際に発生）
BUSY = "^python3 -u src/train_filament.py|^bash scripts/train_with_watchdog.sh"


def busy_fold(k):
    return (f"^python3 -u src/train_filament.py --config configs/_seed2_f{k}.yaml"
            f"|^bash scripts/train_with_watchdog.sh configs/_seed2_f{k}.yaml")


def log(m):
    line = f"[{time.strftime('%m/%d %H:%M')}] {m}"
    print(line, flush=True)
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")


def run(cmd, lf, timeout=24 * 3600):
    if "src/predict_submit.py" in cmd:
        # 推論は学習と同時に走らせない（GPU 16GB に収まらず WSL ごと落ちる）
        while subprocess.run(["pgrep", "-f", BUSY],
                             capture_output=True).returncode == 0:
            time.sleep(120)
    with open(WD / lf, "w") as f:
        r = subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    if r.returncode:
        log(f"★失敗 rc={r.returncode}: {' '.join(str(c) for c in cmd[:4])}… ({lf})")
    return r.returncode == 0


def tune(raw, out, fold, lf):
    if not run(["python3", "-u", "src/tune_postproc.py", "--preds", str(raw), "--fold", str(fold),
                "--out", str(out)] + GRID, lf):
        return None
    return json.loads(Path(out).read_text())


def fmt(d):
    return (f"{d['val_pq']:.4f} (SQ={d['val_sq']:.3f} RQ={d['val_rq']:.3f} "
            f"thr={d['score_thr']} area={d['min_area']})")


def write_cfg(k):
    s = BASE_CFG.read_text()
    s = s.replace('TRAIN: ("filament_train_f0",)', f'TRAIN: ("filament_train_f{k}",)')
    s = s.replace('TEST: ("filament_val_f0",)', f'TEST: ("filament_val_f{k}",)')
    assert f"filament_train_f{k}" in s and f"filament_val_f{k}" in s
    # 学習中 GPU 15.5/16.3GB で 3倍遅くなり PC ごと落ちた（11/06 03:01）。実使用は 9.9GB なので
    # PyTorch の確保上限を 0.85(13.9GB) -> 0.78(12.4GB) に下げ、共有メモリへのはみ出しを防ぐ
    s = s.replace("CUDA_MEM_FRACTION: 0.85", "CUDA_MEM_FRACTION: 0.78")
    assert "CUDA_MEM_FRACTION: 0.78" in s
    s += f"\nSEED: {SEED}                # 2組目。1組目とは初期化・データ順が異なる\n"
    p = REPO / "configs" / f"_seed2_f{k}.yaml"
    p.write_text(s)
    return p.relative_to(REPO)


def merge(paths, out):
    merged = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        merged.update(d["preds"] if "preds" in d else d)
    Path(out).write_text(json.dumps({"preds": merged}))
    return len(merged)


def main():
    with open(SUMMARY, "a") as f:
        f.write(f"\n==== シード2組目 全5fold + 融合判定 {time.strftime('%m/%d %H:%M')} ====\n")
    raws1, raws2, fused = {}, {}, {1: {}, 2: {}, "p": {}}
    for k in range(5):
        out = WD / f"r50_full2048_seed2_f{k}"
        cfg = write_cfg(k)
        # スクリプト差し替え時に学習中だった fold は、その学習の終了を待つ（二重起動しない）
        if subprocess.run(["pgrep", "-f", busy_fold(k)], capture_output=True).returncode == 0:
            log(f"fold{k}: 実行中の学習の終了を待機")
            while subprocess.run(["pgrep", "-f", busy_fold(k)],
                                 capture_output=True).returncode == 0:
                time.sleep(120)
        if not (out / "model_final.pth").exists():
            log(f"fold{k}: 学習開始（約6時間）")
            if not run(["bash", "scripts/train_with_watchdog.sh", str(cfg), str(out),
                        str(WD / f"seed2_train_f{k}.log"), "5000", str(k)],
                       f"seed2_watchdog_f{k}.log", timeout=30 * 3600):
                return
        txt = (out / "log.txt").read_text(errors="ignore")
        mm = re.findall(r"Removed \d+ images with no usable annotations\. (\d+) images left", txt)
        got = int(mm[-1]) if mm else -1
        if got != EXPECT_N[k]:
            log(f"★★ fold{k}: 学習レコード数 {got} ≠ 期待 {EXPECT_N[k]} → リークの疑い。中止")
            return
        # ディスク節約（空き 53GB に対し1foldで途中保存 10GB）。model_final だけ残す
        for ck in out.glob("model_0*.pth"):
            ck.unlink()
        log(f"fold{k}: 学習完了  レコード数 {got} ✓")

        raw = out / "val_raw_prob50.json"
        if not raw.exists():
            if not run(["python3", "-u", "src/predict_submit.py", "--config", str(cfg),
                        "--weights", str(out / "model_final.pth"), "--split", "val",
                        "--fold", str(k), "--tta", "--tta-views", "8", "--tta-merge", "prob",
                        "--mask-bin-thr", "0.5", "--dump-raw", str(raw),
                        "--out", f"/tmp/seed2_f{k}.csv"], f"seed2_infer_f{k}.log"):
                return
        raws2[k] = raw
        raws1[k] = WD / f"r50_full2048_f{k}" / "val_raw_prob50.json"
        d2 = tune(raw, out / "postproc_prob50.json", k, f"seed2_tune_f{k}.log")
        line = f"  fold{k}: 2組目単独 {fmt(d2)}" if d2 else f"  fold{k}: 2組目単独 評価失敗"
        for mv in [1, 2]:
            j = out / f"val_fused_mv{mv}.json"
            if run(["python3", "-u", "src/ensemble.py", "--preds", str(raws1[k]), str(raw),
                    "--out", str(j), "--min-votes", str(mv)], f"seed2_fuse_f{k}_mv{mv}.log"):
                fused[mv][k] = j
                df = tune(j, out / f"postproc_fused_mv{mv}.json", k, f"seed2_tune_f{k}_mv{mv}.log")
                if df:
                    line += f" / マスク融合mv{mv} {df['val_pq']:.4f}"
        praw = out / "val_raw_prob2m.json"
        if not praw.exists():
            run(["python3", "-u", "src/predict_submit.py", "--config", str(cfg),
                 "--weights", str(WD / f"r50_full2048_f{k}" / "model_final.pth"),
                 str(out / "model_final.pth"), "--split", "val", "--fold", str(k),
                 "--tta", "--tta-views", "8", "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                 "--dump-raw", str(praw), "--out", f"/tmp/seed2p_f{k}.csv"], f"seed2_pinfer_f{k}.log")
        if praw.exists():
            fused["p"][k] = praw
            dp = tune(praw, out / "postproc_prob2m.json", k, f"seed2_ptune_f{k}.log")
            if dp:
                line += f" / 確率平均融合 {dp['val_pq']:.4f}"
        log(line)

    # ---- OOF 707枚 ----
    log("=== OOF 707枚 ===")
    res = {}
    for name, paths in [("2組目単独", raws2), ("融合 min_votes=1", fused[1]),
                        ("融合 min_votes=2", fused[2]), ("融合 確率平均", fused["p"])]:
        if len(paths) < 5:
            log(f"  {name}: fold が揃わない（{len(paths)}）")
            continue
        j = WD / f"oof_seed2_{len(res)}.json"
        n = merge([paths[k] for k in range(5)], j)
        d = tune(j, WD / f"postproc_oof_seed2_{len(res)}.json", -1, f"seed2_tune_oof_{len(res)}.log")
        if d:
            res[name] = d
            log(f"  {name:18s} {fmt(d)}  [{n}枚]")
    log(f"  1組目単独（基準）   {BASE_OOF:.4f}")
    best = max((v["val_pq"], k) for k, v in res.items() if k.startswith("融合")) if res else None
    if not best or best[0] < BASE_OOF + 0.005:
        log("  融合の効果は基準（+0.005）未満")
        return
    log(f"  ★ {best[1]} で {best[0] - BASE_OOF:+.4f}。採用")

    if best[1] == "融合 確率平均":
        # fold ごとに2モデルの確率平均 → 5本を現行提出と同じマスク融合（min_votes=2 など）で束ねる
        praws = []
        for k in range(5):
            out = WD / f"r50_full2048_seed2_f{k}"
            traw = out / "test_raw_prob2m.json"
            if not traw.exists():
                log(f"  test 推論 fold{k} 2モデル確率平均（180枚、約50分）")
                if not run(["python3", "-u", "src/predict_submit.py", "--config",
                            str(REPO / "configs" / f"_seed2_f{k}.yaml"), "--weights",
                            str(WD / f"r50_full2048_f{k}" / "model_final.pth"),
                            str(out / "model_final.pth"), "--tta", "--tta-views", "8",
                            "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                            "--dump-raw", str(traw), "--out", f"/tmp/seed2p_t{k}.csv"],
                           f"seed2_ptinfer_f{k}.log"):
                    return
            praws.append(str(traw))
        for mv in [1, 2, 3]:
            j = WD / f"test_ens5x2p_mv{mv}.json"
            if run(["python3", "-u", "src/ensemble.py", "--preds"] + praws +
                   ["--out", str(j), "--min-votes", str(mv)], f"seed2_ptfuse_mv{mv}.log"):
                log(f"  test 5fold×(2モデル確率平均) min_votes={mv}: {j.name}")
        log("※ 提出はまだしない（min_votes と後処理は OOF に基づき相談して決める）")
        return

    # ---- 採用時（マスク融合）: test を2組目でも推論し、10モデル融合を作る ----
    traws = []
    for k in range(5):
        for out, cfg in [(WD / f"r50_full2048_f{k}",
                          REPO / "configs" / (f"_s3_f{k}.yaml" if k else BASE_CFG.name)),
                         (WD / f"r50_full2048_seed2_f{k}", REPO / "configs" / f"_seed2_f{k}.yaml")]:
            traw = out / "test_raw_prob50.json"
            if not traw.exists():
                log(f"  test 推論 {out.name}（180枚、約25分）")
                if not run(["python3", "-u", "src/predict_submit.py", "--config", str(cfg),
                            "--weights", str(out / "model_final.pth"), "--tta", "--tta-views", "8",
                            "--tta-merge", "prob", "--mask-bin-thr", "0.5",
                            "--dump-raw", str(traw), "--out", f"/tmp/seed2_t_{out.name}.csv"],
                           f"seed2_tinfer_{out.name}.log"):
                    return
            traws.append(str(traw))
    for mv in [2, 3, 4, 5]:
        j = WD / f"test_ens10_mv{mv}.json"
        if run(["python3", "-u", "src/ensemble.py", "--preds"] + traws +
               ["--out", str(j), "--min-votes", str(mv)], f"seed2_tfuse_mv{mv}.log"):
            log(f"  test 10モデル融合 min_votes={mv}: {j.name}")
    log("※ 提出はまだしない（min_votes と後処理は OOF に基づき相談して決める）")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"★★ 例外: {e!r}")
        raise
    finally:
        log("==== シード2組目 終了 ====")
