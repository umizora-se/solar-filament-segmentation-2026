"""
Detectron2 へのデータ登録。

- pycocotools の COCO() を通さず、MAGFiLO json から直接 detectron2 標準 dict を作る
  （image の `id` が文字列、annotation id が uuid でも問題なく扱うため）。
- train: アノテーターレコード単位（1154 サンプル / 全 fold）。fold で train/val に振り分け。
- val:   物理写真単位（1 stem = 1 dict、画像だけ使う）。GT は PQ 評価側が json から引く。

登録名:
  filament_train_f{fold}
  filament_val_f{fold}
  filament_test            （test_images、annotation なし）
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import pycocotools.mask as mask_util

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

_CANDIDATES = [
    os.environ.get("FILAMENT_ROOT"),
    str(REPO / "datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026"),
    "datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
    "/kaggle/input/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026",
]
ROOT = next((Path(c) for c in _CANDIDATES if c and Path(c).exists()), None)
if ROOT is None:
    raise FileNotFoundError(f"dataset root not found; tried {_CANDIDATES}")

TRAIN_IMG_DIR = ROOT / "train" / "train_images"
TEST_IMG_DIR = ROOT / "test" / "test_images"
SINGLECLASS_JSON = DATA / "magfilo_singleclass.json"
FOLDS_CSV = DATA / "folds.csv"

FILAMENT_CATEGORIES = [{"id": 1, "name": "filament"}]


def _stem(file_name: str) -> str:
    return Path(file_name).stem


def load_folds() -> dict[str, int]:
    fold_of = {}
    with open(FOLDS_CSV) as f:
        for row in csv.DictReader(f):
            fold_of[row["stem"]] = int(row["fold"])
    return fold_of


def _coco_index():
    """magfilo_singleclass.json を読み、(images, anns_by_image_id) を返す。"""
    coco = json.loads(SINGLECLASS_JSON.read_text())
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    return coco["images"], anns_by_img


def _to_d2_annos(anns):
    out = []
    for a in anns:
        seg = a.get("segmentation") or []
        if not seg or not seg[0] or len(seg[0]) < 6:
            continue  # 退化ポリゴンは捨てる
        out.append({
            "bbox": [float(v) for v in a["bbox"]],
            "bbox_mode": BoxMode.XYWH_ABS,
            "segmentation": [[float(v) for v in seg[0]]],
            "category_id": 0,          # 単一クラス -> 0-index
            "iscrowd": int(a.get("iscrowd", 0)),
        })
    return out


def get_train_dicts(fold: int, split: str):
    """split in {'train','val'}. アノテーターレコード単位。"""
    fold_of = load_folds()
    images, anns_by_img = _coco_index()
    want_val = (split == "val")
    dicts = []
    for im in images:
        st = _stem(im["file_name"])
        if fold_of.get(st) is None:
            continue
        is_val = (fold_of[st] == fold)
        if is_val != want_val:
            continue
        dicts.append({
            "file_name": str(TRAIN_IMG_DIR / im["file_name"]),
            "image_id": im["id"],                 # 文字列（アノテーターレコードID）
            "stem": st,
            "height": int(im.get("height", 2048)),
            "width": int(im.get("width", 2048)),
            "annotations": _to_d2_annos(anns_by_img.get(im["id"], [])),
        })
    return dicts


def get_val_photo_dicts(fold: int):
    """物理写真単位（PQ 評価・推論用）。1 stem = 1 dict、annotations は空。"""
    fold_of = load_folds()
    images, _ = _coco_index()
    seen = set()
    dicts = []
    for im in images:
        st = _stem(im["file_name"])
        if fold_of.get(st) != fold or st in seen:
            continue
        seen.add(st)
        dicts.append({
            "file_name": str(TRAIN_IMG_DIR / im["file_name"]),
            "image_id": st,               # stem をそのまま画像IDに
            "stem": st,
            "height": int(im.get("height", 2048)),
            "width": int(im.get("width", 2048)),
            "annotations": [],
        })
    return dicts


def get_test_dicts():
    dicts = []
    for p in sorted(TEST_IMG_DIR.glob("*.jpeg")):
        dicts.append({
            "file_name": str(p),
            "image_id": p.stem,
            "stem": p.stem,
            "height": 2048, "width": 2048,
            "annotations": [],
        })
    return dicts


NEIGHBOR_DIR = REPO / "datasets" / "gong_neighbors"


def get_train_dicts_with_neighbors(fold: int):
    """学習レコード + 近傍 GONG フレームへのラベル転写レコード。

    近傍フレーム（同じ観測所で ±3分以内、src/gong_neighbors.py で取得・変換）に、
    元画像の全アノテーターのラベルをそのまま写す。ずれは実測で1画素未満
    （位相相関 0.16px / 0.14px）なので補正しない。

    ★リーク防止: 近傍フレームは必ず元画像と同じ fold に所属させる。
      元画像が val fold の写真なら、その近傍は学習に一切入れない。
      近傍は元画像とほぼ同じ内容なので、他 fold の学習に入ると実質的に
      val 画像を学習したことになる。
    """
    base = get_train_dicts(fold, "train")
    fold_of = load_folds()
    val_stems = {s for s, f in fold_of.items() if f == fold}
    by_stem = defaultdict(list)
    for d in base:
        by_stem[d["stem"]].append(d)

    man = NEIGHBOR_DIR / "manifest.csv"
    if not man.exists():
        raise FileNotFoundError(f"近傍フレームの記録が無い: {man}")
    qc_pass = None                             # qc.csv があれば合格分だけ使う
    qc = NEIGHBOR_DIR / "qc.csv"
    if qc.exists():
        qc_pass = {r["neighbor"] for r in csv.DictReader(open(qc)) if r.get("pass") == "1"}
    extra, skipped_val, skipped_qc = [], 0, 0
    with open(man) as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or r.get("excluded"):
                continue
            src = r["source"]
            jpg = NEIGHBOR_DIR / "jpeg" / f"{r['neighbor']}.jpeg"
            if not jpg.exists():
                continue
            if src in val_stems:
                skipped_val += 1               # val 写真の近傍は学習に入れない
                continue
            if qc_pass is not None and r["neighbor"] not in qc_pass:
                skipped_qc += 1                # 品質管理で不合格（雲・ずれ）
                continue
            for d in by_stem.get(src, []):
                extra.append({
                    "file_name": str(jpg),
                    "image_id": f"{d['image_id']}@{r['neighbor']}",
                    "stem": r["neighbor"],
                    "source_stem": src,
                    "height": d["height"],
                    "width": d["width"],
                    "annotations": d["annotations"],
                })
    # 最終確認: 近傍の元画像がこの fold の val に含まれていないこと
    bad = [e for e in extra if e["source_stem"] in val_stems]
    assert not bad, f"リーク: val 写真の近傍が学習に {len(bad)} 件入っている"
    print(f"[neighbors] fold{fold}: 元 {len(base)} + 近傍 {len(extra)} = {len(base) + len(extra)} "
          f"レコード（val写真の近傍 {skipped_val} 件・品質不合格 {skipped_qc} 件は除外）", flush=True)
    return base + extra


PSEUDO_DIR = REPO / "datasets" / "gong_unlabeled"
PSEUDO_REAL_REPEAT = 2          # 実ラベルを2回入れて、疑似ラベル（約1300枚）より多く保つ


def get_train_dicts_with_pseudo(fold: int):
    """学習レコード（×PSEUDO_REAL_REPEAT）+ 外部 GONG フレームの疑似ラベル。

    疑似ラベルは src/make_pseudo.py が作る pseudo_f{fold}.json。教師は同じ fold の
    モデル（= この fold の val 写真を学習していない）なので、val の情報は入らない。
    外部フレームは学習・test のある日を丸ごと除外して取得している（src/gong_unlabeled.py）。
    """
    base = get_train_dicts(fold, "train")
    real = []
    for r in range(PSEUDO_REAL_REPEAT):
        real += [{**d, "image_id": f"{d['image_id']}#r{r}"} for d in base]
    pj = PSEUDO_DIR / f"pseudo_f{fold}.json"
    if not pj.exists():
        raise FileNotFoundError(f"疑似ラベルが無い: {pj}")
    meta = json.loads(pj.read_text())
    assert meta["teacher_fold"] == fold, f"教師の fold が違う: {meta['teacher_fold']} != {fold}"
    fold_of = load_folds()
    extra = []
    for stem, rles in meta["labels"].items():
        assert stem not in fold_of, f"疑似ラベルに競技の学習画像が混入: {stem}"
        annos = []
        for counts in rles:
            rle = {"size": [2048, 2048], "counts": counts}
            annos.append({"bbox": [float(v) for v in mask_util.toBbox(rle)],
                          "bbox_mode": BoxMode.XYWH_ABS, "segmentation": rle,
                          "category_id": 0, "iscrowd": 0})
        extra.append({"file_name": str(PSEUDO_DIR / "jpeg" / f"{stem}.jpeg"),
                      "image_id": f"pl@{stem}", "stem": stem, "pseudo": True,
                      "height": 2048, "width": 2048, "annotations": annos})
    print(f"[pseudo] fold{fold}: 実 {len(base)}×{PSEUDO_REAL_REPEAT} + 疑似 {len(extra)} = "
          f"{len(real) + len(extra)} レコード（疑似の個体数 "
          f"{sum(len(e['annotations']) for e in extra)}）", flush=True)
    return real + extra


def register_filament(fold: int):
    """fold に対応する train/val/test データセットを登録。"""
    names = {
        f"filament_train_f{fold}": (lambda: get_train_dicts(fold, "train"), "train"),
        f"filament_train_nb_f{fold}": (lambda: get_train_dicts_with_neighbors(fold), "train"),
        f"filament_train_pl_f{fold}": (lambda: get_train_dicts_with_pseudo(fold), "train"),
        f"filament_val_f{fold}":   (lambda: get_val_photo_dicts(fold), "val"),
        "filament_test":           (get_test_dicts, "test"),
    }
    for name, (fn, kind) in names.items():
        if name in DatasetCatalog.list():
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
        DatasetCatalog.register(name, fn)
        md = MetadataCatalog.get(name)
        md.set(
            thing_classes=["filament"],
            evaluator_type="filament_pq",
            singleclass_json=str(SINGLECLASS_JSON),
            train_img_dir=str(TRAIN_IMG_DIR),
            split_kind=kind,
            fold=fold,
        )
    return list(names)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()
    register_filament(args.fold)
    tr = get_train_dicts(args.fold, "train")
    va = get_train_dicts(args.fold, "val")
    vp = get_val_photo_dicts(args.fold)
    te = get_test_dicts()
    ninst = sum(len(d["annotations"]) for d in tr)
    print(f"fold {args.fold}: train_records={len(tr)} (instances={ninst})  "
          f"val_records={len(va)}  val_photos={len(vp)}  test_photos={len(te)}")
    print("sample train dict keys:", sorted(tr[0].keys()))
    print("sample annotation:", tr[0]["annotations"][0] if tr[0]["annotations"] else None)
