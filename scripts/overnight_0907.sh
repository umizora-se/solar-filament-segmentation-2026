#!/usr/bin/env bash
# 夜間自動パイプライン 2026-09-07
#   Phase1: 高解像度学習の完走待ち → val評価 → test推論 → Kaggle提出（ユーザー事前承認済み）
#   Phase2: mask_bin_thr 探索（0.40 / 0.35 / 0.45）on Dice版fold0
cd .
SUMMARY=work_dirs/overnight_summary.txt
log(){ echo "[$(date '+%m/%d %H:%M')] $*" | tee -a "$SUMMARY"; }
echo "==== 夜間パイプライン開始 $(date) ====" > "$SUMMARY"

# ---- Phase 0: 学習完走を待つ ----
log "Phase0: 高解像度学習の完走待ち"
while ps -eo comm,args | awk '$1=="python3" && /train_filament/' | grep -q .; do sleep 60; done
log "学習プロセス終了を確認"
if [ ! -f work_dirs/r50_hires/model_final.pth ]; then
  log "★ model_final.pth が無い（学習失敗）。Phase1をスキップ"
else
  # ---- Phase 1: hires 評価と提出 ----
  export FILAMENT_HIRES_MASK=1
  log "Phase1a: hires val推論（4方向TTA）"
  python3 -u src/predict_submit.py --config configs/maskdino_filament_r50_hires.yaml \
    --weights work_dirs/r50_hires/model_final.pth \
    --split val --fold 0 --tta --tta-views 4 \
    --dump-raw work_dirs/r50_hires/val_raw_tta.json --out /tmp/hires_val.csv \
    > work_dirs/hires_val_infer.log 2>&1
  if [ -f work_dirs/r50_hires/val_raw_tta.json ]; then
    log "Phase1b: 閾値チューニング"
    python3 -u src/tune_postproc.py --preds work_dirs/r50_hires/val_raw_tta.json --fold 0 \
      --out work_dirs/r50_hires/postproc.json \
      --score-thr 0.20,0.25,0.30,0.35,0.40 --min-area 100,200,300 \
      > work_dirs/hires_tune.log 2>&1
    VALPQ=$(python3 -c "import json;print(round(json.load(open('work_dirs/r50_hires/postproc.json'))['val_pq'],4))" 2>/dev/null)
    log "hires val-PQ = ${VALPQ}（基準: Dice版 0.4198）"
    log "Phase1c: hires test推論"
    python3 -u src/predict_submit.py --config configs/maskdino_filament_r50_hires.yaml \
      --weights work_dirs/r50_hires/model_final.pth \
      --postproc work_dirs/r50_hires/postproc.json \
      --tta --tta-views 4 --out submission_hires.csv \
      > work_dirs/hires_test.log 2>&1
    if grep -aq "\[verify\] PASS" work_dirs/hires_test.log; then
      DUP=0
      for f in submission.csv submission_tuned.csv submission_tta.csv submission_tta8.csv submission_ens5.csv submission_tversky.csv; do
        [ -f "$f" ] && [ "$(md5sum < submission_hires.csv)" = "$(md5sum < "$f")" ] && DUP=1
      done
      if [ "$DUP" = "0" ]; then
        log "Phase1d: Kaggle提出（ユーザー事前承認済み）"
        kaggle competitions submit -c filament-segmentation-2026 \
          -f submission_hires.csv \
          -m "MaskDINO R50 hires mask head (stride2, 768crop) + 4-view TTA (val-PQ ${VALPQ})" \
          >> "$SUMMARY" 2>&1 && log "提出完了" || log "★提出コマンド失敗"
        sleep 120
        kaggle competitions submissions -c filament-segmentation-2026 2>/dev/null | head -4 >> "$SUMMARY"
      else
        log "★過去提出と同一内容のため提出せず"
      fi
    else
      log "★ test CSV の verify 失敗。提出せず"
    fi
  else
    log "★ hires val推論失敗（work_dirs/hires_val_infer.log 参照）。提出せず"
  fi
  unset FILAMENT_HIRES_MASK
fi

# ---- Phase 2: mask_bin_thr 探索（Dice版fold0） ----
export FILAMENT_HIRES_MASK=0
for T in 0.40 0.35 0.45; do
  log "Phase2: mask_bin_thr=${T} の val推論"
  python3 -u src/predict_submit.py --config configs/maskdino_filament_r50_1024crop.yaml \
    --weights work_dirs/r50_f0/model_final.pth \
    --split val --fold 0 --tta --tta-views 4 --mask-bin-thr "$T" \
    --dump-raw "work_dirs/r50_f0/val_raw_tta_mb${T}.json" --out /tmp/mb_val.csv \
    > "work_dirs/mb${T}_infer.log" 2>&1
  if [ -f "work_dirs/r50_f0/val_raw_tta_mb${T}.json" ]; then
    python3 -u src/tune_postproc.py --preds "work_dirs/r50_f0/val_raw_tta_mb${T}.json" --fold 0 \
      --out "work_dirs/r50_f0/postproc_mb${T}.json" \
      --score-thr 0.25,0.30,0.35,0.40 --min-area 100,200,300 \
      > "work_dirs/mb${T}_tune.log" 2>&1
    PQ=$(python3 -c "import json;d=json.load(open('work_dirs/r50_f0/postproc_mb${T}.json'));print(f\"{d['val_pq']:.4f} (SQ={d['val_sq']:.3f} thr={d['score_thr']} area={d['min_area']})\")" 2>/dev/null)
    log "mask_bin_thr=${T}: val-PQ ${PQ}"
  else
    log "★ mask_bin_thr=${T} の推論失敗"
  fi
done

log "==== 全処理完了 ===="
echo "" >> "$SUMMARY"
echo "基準: Dice fold0 4方向TTA(mask_bin 0.50) val-PQ 0.4198 / 8方向 0.4268 / LB 0.39" >> "$SUMMARY"
