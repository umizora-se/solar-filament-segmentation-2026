"""疑似ラベル学習用に、学習・test のどちらにも無い日の GONG フレームを取得する。

目的（半教師あり学習）:
  学習 707枚は 656日分しかない。別の日のフレームに各 fold のモデルで疑似ラベルを付け、
  学習データの多様性を増やす。ラベル転写（数分以内の近傍＝ほぼ同じ画像に正解を写す）
  が効かなかったのは新しい情報が無かったためで、今回は別の日の太陽面を使う。

規則（Kaggle discussion 748045 の主催者裁定）:
  外部データは学習にのみ使う。test 画像と、test の撮影時刻の前後1時間の観測は含めない。
  ここではさらに安全側に、学習画像・test 画像のある日は丸ごと除外する。

選び方:
  年の分布を学習+test に合わせて日を抽選し（乱数シード固定）、1日あたり最大2枚、
  観測所を変え、撮影時刻を4時間以上離して選ぶ。

取得元・作法・変換は src/gong_neighbors.py と同じ（FTP、接続1本、既定2秒間隔）。

出力:
  datasets/gong_unlabeled/jpeg/        変換後の JPEG（競技画像と同じ処理）
  datasets/gong_unlabeled/manifest.csv 全候補の記録（URL・採否・ハッシュ）

    python src/gong_unlabeled.py --n-days 650 --sleep 2
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, TEST_IMG_DIR  # noqa: E402
from gong_convert import fz_to_jpeg  # noqa: E402
from gong_neighbors import FTP_BASE, TEST_EXCLUDE_SEC, curl, ts  # noqa: E402

OUT = REPO / "datasets" / "gong_unlabeled"
SITES = ["Bh", "Ch", "Lh", "Mh", "Th", "Uh"]
MIN_GAP_SEC = 4 * 3600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=650)
    ap.add_argument("--per-day", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--max-consecutive-fail", type=int, default=5)
    args = ap.parse_args()
    for d in ["fits", "jpeg", "listings"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    train = {Path(im["file_name"]).stem for im in coco["images"]}
    test = {p.stem for p in Path(TEST_IMG_DIR).glob("*.jpeg")}
    test_t = sorted(ts(s) for s in test)
    used_days = {s[:8] for s in train | test}
    years = Counter(s[:4] for s in train | test)

    # ---- 日の抽選（年の分布を学習+test に合わせる）----
    rng = random.Random(args.seed)
    tot = sum(years.values())
    days = []
    for y, c in sorted(years.items()):
        n = round(args.n_days * c / tot)
        pool = [date(int(y), 1, 1) + timedelta(d) for d in range(366)]
        pool = [d.strftime("%Y%m%d") for d in pool if d.year == int(y)]
        pool = [d for d in pool if d not in used_days]
        days += rng.sample(pool, min(n, len(pool)))
    days.sort()
    print(f"[plan] 抽選 {len(days)} 日（学習/test のある {len(used_days)} 日は除外）")

    # ---- 日別一覧 → 1日最大 per-day 枚を選ぶ ----
    rows, fails = [], 0
    for n, day in enumerate(days, 1):
        lst = OUT / "listings" / f"{day}.txt"
        if not lst.exists() or lst.stat().st_size == 0:
            code = curl(f"{FTP_BASE}/{day[:6]}/{day}/", lst, timeout=90, list_only=True)
            time.sleep(args.sleep)
            if code not in (226, 250, 200):
                lst.unlink(missing_ok=True)
                fails += 1
                if fails >= args.max_consecutive_fail * 4:     # 観測の無い日は普通にある
                    print("[stop] 一覧の失敗が続いたため停止")
                    break
                continue
            fails = 0
        names = [l.strip()[:-len(".fits.fz")] for l in lst.read_text().splitlines()
                 if l.strip().endswith(".fits.fz") and l.strip()[14:16] in SITES]
        drng = random.Random(f"{args.seed}-{day}")
        drng.shuffle(names)
        picked = []
        for c in names:
            t = ts(c)
            if c in test or c in train:
                continue
            if any(abs((tt - t).total_seconds()) <= TEST_EXCLUDE_SEC for tt in test_t
                   if abs((tt - t).days) <= 1):
                continue                                       # 念のため（日単位で除外済み）
            if any(c[14:] == p[14:] or abs((ts(p) - t).total_seconds()) < MIN_GAP_SEC
                   for p in picked):
                continue
            picked.append(c)
            if len(picked) >= args.per_day:
                break
        for c in picked:
            rows.append({"stem": c, "url": f"{FTP_BASE}/{day[:6]}/{day}/{c}.fits.fz"})
        if n % 50 == 0:
            print(f"  一覧 [{n}/{len(days)}] 候補 {len(rows)}", flush=True)
    print(f"[plan] 取得対象 {len(rows)} 枚  観測所 {dict(Counter(r['stem'][14:] for r in rows))}")

    # ---- 取得と変換（取得済みは飛ばす）----
    man = OUT / "manifest.csv"
    status = {}
    if man.exists():
        status = {r["stem"]: r for r in csv.DictReader(open(man)) if r.get("status") == "ok"}
    fails, t0 = 0, time.time()
    for i, r in enumerate(rows, 1):
        jpg, fz = OUT / "jpeg" / f"{r['stem']}.jpeg", OUT / "fits" / f"{r['stem']}.fits.fz"
        if r["stem"] in status and jpg.exists():
            continue
        code = curl(r["url"], fz)
        time.sleep(args.sleep)
        if code != 226 or not fz.exists() or fz.stat().st_size < 100_000:
            fz.unlink(missing_ok=True)
            status[r["stem"]] = {"status": f"取得失敗 code={code}"}
            fails += 1
            if fails >= args.max_consecutive_fail:
                print("[stop] 失敗が続いたため停止")
                break
            continue
        fails = 0
        try:
            fz_to_jpeg(fz, jpg)
            status[r["stem"]] = {"status": "ok",
                                 "sha256": hashlib.sha256(fz.read_bytes()).hexdigest()}
        except Exception as e:
            status[r["stem"]] = {"status": f"変換失敗 {type(e).__name__}"}
        if i % 25 == 0 or i == len(rows):
            el = time.time() - t0
            ok = sum(v.get("status") == "ok" for v in status.values())
            print(f"  [{i}/{len(rows)}] 成功 {ok}  {el/60:.0f}分経過 "
                  f"残り約{el/i*(len(rows)-i)/60:.0f}分", flush=True)
            _write(man, rows, status)
    _write(man, rows, status)
    print(f"[done] 成功 {sum(v.get('status') == 'ok' for v in status.values())} / {len(rows)}")


def _write(man, rows, status):
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "url", "status", "sha256"])
        w.writeheader()
        for r in rows:
            s = status.get(r["stem"], {})
            w.writerow({**r, "status": s.get("status", ""), "sha256": s.get("sha256", "")})


if __name__ == "__main__":
    main()
