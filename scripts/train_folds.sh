#!/usr/bin/env bash
# fold 1-4 を逐次学習する（fold 0 は学習済み）。
# CUDA カーネル導入後は 1fold 約2.7時間、合計約11時間。
cd .
for F in "$@"; do
  echo "##################### fold $F 開始 $(date '+%m/%d %H:%M') #####################"
  python3 -u src/train_filament.py \
    --config configs/maskdino_filament_r50_1024crop.yaml \
    --fold "$F" --output-dir "work_dirs/r50_f${F}"
  echo "##################### fold $F 終了 $(date '+%m/%d %H:%M') #####################"
done
echo "ALL DONE $(date '+%m/%d %H:%M')"
