"""学習画像の近傍 GONG フレームを取得し、競技画像と同一の JPEG に変換する。

目的（ラベル転写）:
  学習画像と同じ観測所で ±3分以内に撮られたフレームに、競技の学習ラベルを
  そのまま写して追加の学習データにする。
  主催者の裁定（Kaggle discussion 748045）:
    "yes this is allowed; It does not violate any rules, or the fairness of the competition."
  条件: test 画像、および test の撮影時刻の前後1時間以内の観測を含めないこと。

取得元:
  NSO/GONG アーカイブ ftp://gong2.nso.edu/HA/haf/YYYYMM/YYYYMMDD/<stamp><site>h.fits.fz
  （VSO が案内する公式の配信先。HTTPS 側は外部接続を拒否していたため FTP を使う）

作法:
  接続は常に1本、各リクエストの間に既定2秒。失敗が5回続いたら即停止する。
  再実行すると取得済みのものは飛ばす（途中から再開できる）。

出力:
  datasets/gong_neighbors/fits/   取得した FITS（.fits.fz）
  datasets/gong_neighbors/jpeg/   変換後の JPEG（競技画像と同じ処理）
  datasets/gong_neighbors/listings/  日別のディレクトリ一覧
  datasets/gong_neighbors/manifest.csv  全候補の記録（採否・除外理由・URL・ハッシュ）

    python src/gong_neighbors.py --window-min 3 --sleep 2
    python src/gong_neighbors.py --dry-run        # 取得せず候補だけ数える
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from filament_dataset import SINGLECLASS_JSON, TEST_IMG_DIR  # noqa: E402
from gong_convert import fz_to_jpeg  # noqa: E402

OUT = REPO / "datasets" / "gong_neighbors"
FTP_BASE = "ftp://gong2.nso.edu/HA/haf"
TEST_EXCLUDE_SEC = 3600            # 主催者の条件: test の前後1時間


def ts(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def curl(url: str, out: Path | None, timeout: int = 180, list_only: bool = False) -> int:
    cmd = ["curl", "-s", "--connect-timeout", "30", "--max-time", str(timeout),
           "-w", "%{http_code}"]
    if list_only:
        cmd.append("--list-only")
    cmd += ["-o", str(out) if out else "/dev/null", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window-min", type=float, default=3.0)
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--max-consecutive-fail", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="先頭N件だけ（試験用）")
    ap.add_argument("--limit-days", type=int, default=0, help="先頭N日分だけ一覧を取る（試験用）")
    args = ap.parse_args()

    for d in ["fits", "jpeg", "listings"]:
        (OUT / d).mkdir(parents=True, exist_ok=True)

    test_t = sorted(ts(p.stem) for p in Path(TEST_IMG_DIR).glob("*.jpeg"))

    def near_test(t: datetime) -> bool:
        i = bisect.bisect_left(test_t, t)
        return any(abs((test_t[j] - t).total_seconds()) <= TEST_EXCLUDE_SEC
                   for j in (i - 1, i) if 0 <= j < len(test_t))

    coco = json.loads(Path(SINGLECLASS_JSON).read_text())
    train = sorted({Path(im["file_name"]).stem for im in coco["images"]})
    train_set = set(train)
    test_set = {p.stem for p in Path(TEST_IMG_DIR).glob("*.jpeg")}
    print(f"[info] 学習写真 {len(train)}枚 / test {len(test_t)}枚  窓 ±{args.window_min}分")

    # ---- 日別一覧を取得して候補を決める ----
    by_day: dict[str, list[str]] = {}
    for s in train:
        by_day.setdefault(s[:8], []).append(s)
    rows, fails = [], 0
    days = sorted(by_day.items())
    if args.limit_days:
        days = days[: args.limit_days]
    for n, (day, stems) in enumerate(days, 1):
        lst = OUT / "listings" / f"{day}.txt"
        if not lst.exists() or lst.stat().st_size == 0:
            if args.dry_run and not lst.exists():
                pass
            url = f"{FTP_BASE}/{day[:6]}/{day}/"
            code = curl(url, lst, timeout=90, list_only=True)
            time.sleep(args.sleep)
            if code not in (226, 250, 200):
                fails += 1
                print(f"  ★一覧取得失敗 {day} code={code}（連続{fails}）", flush=True)
                lst.unlink(missing_ok=True)
                if fails >= args.max_consecutive_fail:
                    print("[stop] 失敗が続いたため停止（サーバー側の制限の可能性）")
                    break
                continue
            fails = 0
        names = [l.strip() for l in lst.read_text().splitlines() if l.strip().endswith(".fits.fz")]
        for s in stems:
            site = s[14:]                          # 例 "Bh"
            t0 = ts(s)
            for name in names:
                cand = name[:-len(".fits.fz")]
                if cand[14:] != site or cand == s:
                    continue
                dt = (ts(cand) - t0).total_seconds()
                if abs(dt) > args.window_min * 60 + 30:
                    continue
                reason = ""
                if cand in test_set:
                    reason = "test画像そのもの"
                elif near_test(ts(cand)):
                    reason = "testの前後1時間以内"
                elif cand in train_set:
                    reason = "別の学習画像そのもの"
                rows.append({"neighbor": cand, "source": s, "delta_sec": int(dt),
                             "url": f"{FTP_BASE}/{day[:6]}/{day}/{name}",
                             "excluded": reason})
        if n % 50 == 0:
            print(f"  一覧 [{n}/{len(days)}] 候補 {len(rows)}", flush=True)

    keep = [r for r in rows if not r["excluded"]]
    print(f"[plan] 候補 {len(rows)}  除外 {len(rows) - len(keep)}  取得対象 {len(keep)}")
    if args.dry_run:
        _write_manifest(rows, {})
        return

    # ---- 取得と変換 ----
    done = {}
    man = OUT / "manifest.csv"
    if man.exists():
        for r in csv.DictReader(open(man)):
            if r.get("status") == "ok":
                done[r["neighbor"]] = r
    targets = keep[: args.limit] if args.limit else keep
    fails = 0
    t_start = time.time()
    status: dict[str, dict] = dict(done)
    for i, r in enumerate(targets, 1):
        nb = r["neighbor"]
        jpg = OUT / "jpeg" / f"{nb}.jpeg"
        fz = OUT / "fits" / f"{nb}.fits.fz"
        if nb in done and jpg.exists():
            continue
        code = curl(r["url"], fz)
        time.sleep(args.sleep)
        if code != 226 or not fz.exists() or fz.stat().st_size < 100_000:
            fails += 1
            status[nb] = {"status": f"取得失敗 code={code}"}
            print(f"  ★取得失敗 {nb} code={code}（連続{fails}）", flush=True)
            fz.unlink(missing_ok=True)
            if fails >= args.max_consecutive_fail:
                print("[stop] 失敗が続いたため停止")
                break
            continue
        fails = 0
        try:
            fz_to_jpeg(fz, jpg)
            sha = hashlib.sha256(fz.read_bytes()).hexdigest()
            status[nb] = {"status": "ok", "sha256": sha}
        except Exception as e:                  # 変換失敗は記録して続行
            status[nb] = {"status": f"変換失敗 {type(e).__name__}"}
        if i % 25 == 0 or i == len(targets):
            el = time.time() - t_start
            ok = sum(1 for v in status.values() if v.get("status") == "ok")
            print(f"  [{i}/{len(targets)}] 成功 {ok}  {el/60:.0f}分経過 "
                  f"残り約{el/i*(len(targets)-i)/60:.0f}分", flush=True)
            _write_manifest(rows, status)
    _write_manifest(rows, status)
    ok = sum(1 for v in status.values() if v.get("status") == "ok")
    print(f"[done] 成功 {ok} / 対象 {len(targets)}  → {man}")


def _write_manifest(rows, status):
    man = OUT / "manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["neighbor", "source", "delta_sec", "url",
                                          "excluded", "status", "sha256"])
        w.writeheader()
        for r in rows:
            s = status.get(r["neighbor"], {})
            w.writerow({**r, "status": s.get("status", "" if not r["excluded"] else "除外"),
                        "sha256": s.get("sha256", "")})


if __name__ == "__main__":
    main()
