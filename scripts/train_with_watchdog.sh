#!/usr/bin/env bash
# 学習が無言終了しても自動再開する（10/11 19:42 に原因不明の停止が発生したため）。
# チェックポイントは250 iter毎なので、1回の損失は最大20分。
cd .
CFG="$1"; OUT="$2"; LOG="$3"; MAXIT="${4:-5000}"; FOLD="${5:-0}"
# FOLD は必ず渡すこと。train_filament.py は --fold から DATASETS.TRAIN/TEST を
# 上書きするため、config 側の指定は無視される。
# 以前ここで --fold 0 をハードコードしており、fold1-4 の学習が全て
# filament_train_f0 になってリークした（10/12-13 の約31時間が無駄になった）。
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 旧名（torch のバージョンにより こちらしか効かない）
for attempt in $(seq 1 20); do
  # 進捗はログではなくチェックポイントから読む。
  # （古い実行のログが残っていると「完了済み」と誤認して即終了するため。
  #   10/14 に実際に発生した）
  it=0
  if [ -f "$OUT/last_checkpoint" ]; then
    ck=$(cat "$OUT/last_checkpoint")
    it=$(echo "$ck" | grep -oE "[0-9]{7}" | head -1 | sed 's/^0*//')
    it=${it:-0}
  fi
  if [ "$it" -ge "$((MAXIT-5))" ] || [ -f "$OUT/model_final.pth" ]; then
    echo "[watchdog] 完了を検知 (iter=$it)"; break
  fi
  rm -f "$OUT/train.lock"
  RES=""
  [ -f "$OUT/last_checkpoint" ] && RES="--resume"
  echo "[watchdog] 試行 $attempt: iter=$it から開始 $RES ($(date '+%H:%M'))"
  python3 -u src/train_filament.py --config "$CFG" --fold "$FOLD" \
      --output-dir "$OUT" --force $RES >> "$LOG" 2>&1
  rc=$?
  echo "[watchdog] 終了 rc=$rc ($(date '+%H:%M'))"
  [ -f "$OUT/model_final.pth" ] && { echo "[watchdog] model_final.pth あり → 完了"; break; }
  sleep 30
done
