"""
MAGFiLO COCO -> 単一クラス化 + GroupKFold(file_name) 分割。

出力:
  data/magfilo_singleclass.json   … 全アノテーション、category を 1 クラス(filament, id=1)に統合
                                     （image の COCO `id` 文字列・annotation の uuid はそのまま保持）
  data/folds.csv                  … stem, station, year, n_annotators, n_instances, fold(0..K-1)
  data/stats.json                 … 参考統計

分割の単位は「物理写真ファイル名（stem）」。同じ写真の別アノテーターレコードは
必ず同じ fold に入る（= image_id 単位のリークを防ぐ）。層化キーは観測所×太陽周期ビン。

使い方:
  python src/prepare_data.py --k 5 --seed 42
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import numpy as np

# --- パス解決（設計ドキュメントと同じ優先順）-------------------------------
_CANDIDATES = [
    os.environ.get("FILAMENT_ROOT"),
    "./datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "../datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
]
ROOT = next((Path(c) for c in _CANDIDATES if c and Path(c).exists()), None)
if ROOT is None:
    raise FileNotFoundError(f"dataset root not found; tried: {_CANDIDATES}")

TRAIN_DIR = ROOT / "train" / "train_images"
TEST_DIR = ROOT / "test" / "test_images"
ANN_PATH = ROOT / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"

REPO = Path(__file__).resolve().parents[1]
DATA_OUT = REPO / "data"

STATION_NAME = {
    "Bh": "BigBear", "Mh": "MaunaLoa", "Lh": "Learmonth",
    "Uh": "Udaipur", "Th": "ElTeide", "Ch": "CerroTololo",
}


def stem_of(file_name: str) -> str:
    return Path(file_name).stem  # e.g. "20140609195854Bh"


def station_of(stem: str) -> str:
    return stem[-2:]  # "Bh" ...


def year_of(stem: str) -> int:
    return int(stem[:4])


def cycle_bin(year: int) -> str:
    if year <= 2017:
        return "c24"
    if year <= 2020:
        return "min"
    return "c25"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5, help="fold 数")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    DATA_OUT.mkdir(parents=True, exist_ok=True)
    coco = json.loads(ANN_PATH.read_text())

    images = coco["images"]
    anns = coco["annotations"]
    print(f"[load] images(records)={len(images)}  annotations={len(anns)}")

    # --- ディスク上の画像存在チェック ---
    disk_train = {p.stem for p in TRAIN_DIR.glob("*.jpeg")}
    disk_test = {p.stem for p in TEST_DIR.glob("*.jpeg")}
    json_stems = {stem_of(im["file_name"]) for im in images}
    missing = sorted(json_stems - disk_train)
    if missing:
        print(f"[warn] {len(missing)} json stems not found in train_images (例: {missing[:3]})")
    overlap_test = json_stems & disk_test
    print(f"[check] unique photos in json={len(json_stems)}  "
          f"train jpeg on disk={len(disk_train)}  test jpeg on disk={len(disk_test)}  "
          f"json∩test={len(overlap_test)}  (0 が望ましい)")

    # --- 単一クラス化 ---
    out = dict(coco)
    out["categories"] = [{"id": 1, "name": "filament", "supercategory": "filament"}]
    for a in anns:
        a["category_id"] = 1
    (DATA_OUT / "magfilo_singleclass.json").write_text(json.dumps(out))
    print(f"[write] {DATA_OUT/'magfilo_singleclass.json'}")

    # --- stem ごとの集計 ---
    stem_records = collections.defaultdict(list)   # stem -> [image_id, ...]
    for im in images:
        stem_records[stem_of(im["file_name"])].append(im["id"])
    ann_per_image = collections.Counter(a["image_id"] for a in anns)

    stems = sorted(stem_records)
    groups = np.array(stems)
    strat = np.array([f"{station_of(s)}_{cycle_bin(year_of(s))}" for s in stems])

    # --- StratifiedGroupKFold（group とレコードが1:1なので実質 StratifiedKFold）---
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        skf = StratifiedGroupKFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        splits = list(skf.split(stems, strat, groups))
    except Exception as e:  # 古い sklearn 用フォールバック
        print(f"[warn] StratifiedGroupKFold 使用不可 ({e}) -> KFold にフォールバック")
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=args.k, shuffle=True, random_state=args.seed)
        splits = list(kf.split(stems))

    fold_of = {}
    for fi, (_, val_idx) in enumerate(splits):
        for i in val_idx:
            fold_of[stems[i]] = fi

    # --- folds.csv ---
    rows = ["stem,station,station_name,year,n_annotators,n_instances,fold"]
    fold_hist = collections.Counter()
    for s in stems:
        n_ann = len(stem_records[s])
        n_inst = sum(ann_per_image[iid] for iid in stem_records[s])
        f = fold_of[s]
        fold_hist[f] += 1
        rows.append(f"{s},{station_of(s)},{STATION_NAME.get(station_of(s), '?')},"
                    f"{year_of(s)},{n_ann},{n_inst},{f}")
    (DATA_OUT / "folds.csv").write_text("\n".join(rows) + "\n")
    print(f"[write] {DATA_OUT/'folds.csv'}  fold sizes={dict(sorted(fold_hist.items()))}")

    # --- fold ごとの観測所分布を確認 ---
    print("[check] fold × station (photos):")
    per = collections.defaultdict(collections.Counter)
    for s in stems:
        per[fold_of[s]][station_of(s)] += 1
    stations = sorted(STATION_NAME)
    print("      " + " ".join(f"{st:>10}" for st in stations))
    for f in sorted(per):
        print(f"  f{f}: " + " ".join(f"{per[f][st]:>10d}" for st in stations))

    # --- stats.json ---
    stats = {
        "root": str(ROOT),
        "n_image_records": len(images),
        "n_unique_photos": len(stems),
        "n_annotations": len(anns),
        "annotators_per_photo": dict(collections.Counter(len(v) for v in stem_records.values())),
        "instances_per_record": {
            "min": int(min(ann_per_image.values())),
            "p50": float(np.percentile(list(ann_per_image.values()), 50)),
            "p90": float(np.percentile(list(ann_per_image.values()), 90)),
            "max": int(max(ann_per_image.values())),
        },
        "k": args.k, "seed": args.seed,
        "json_test_overlap": len(overlap_test),
        "disk_train_jpeg": len(disk_train), "disk_test_jpeg": len(disk_test),
    }
    (DATA_OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[write] {DATA_OUT/'stats.json'}")
    print("[done]")


if __name__ == "__main__":
    main()
