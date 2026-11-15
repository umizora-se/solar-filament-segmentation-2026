#!/usr/bin/env bash
# 学習ログを読みやすい形でターミナルに流す
#   ./scripts/watch_train.sh                       # 5-fold のログを追う
#   ./scripts/watch_train.sh work_dirs/xxx.log     # 任意のログを追う
LOG="${1:-work_dirs/train_folds.log}"
echo "=== $LOG を監視中 (Ctrl-C で終了。学習は止まりません) ==="
echo

tail -n +1 -f "$LOG" | stdbuf -oL grep -aE "#####|vram-guard|iter:|FilamentPQEvaluator|Inference done|Total training time|Traceback|CUDA out of memory|Saving checkpoint|ALL DONE" \
| stdbuf -oL awk '
  /#####|ALL DONE/          { print "\033[1;33m" $0 "\033[0m"; next }
  /vram-guard/              { print "\033[36m" $0 "\033[0m"; next }
  /Traceback|out of memory/ { print "\033[31m★ " $0 "\033[0m"; next }
  /FilamentPQEvaluator/     { print "\033[1;32m" $0 "\033[0m"; next }
  /Total training time/     { print "\033[1;32m✓ " $0 "\033[0m"; next }
  /Saving checkpoint/       { n=split($0,a,"/"); printf "  [保存] %s\n", a[n]; next }
  /Inference done/ {
      match($0, /done [0-9]+\/[0-9]+/); d=substr($0,RSTART+5,RLENGTH-5)
      match($0, /ETA=[0-9:]+/);         e=substr($0,RSTART,RLENGTH)
      printf "  [評価] %s  %s\n", d, e; next }
  /iter:/ {
      for(i=1;i<=NF;i++){
        if($i=="iter:")       it=$(i+1)
        if($i=="total_loss:") tl=$(i+1)
        if($i=="loss_ce:")    ce=$(i+1)
        if($i=="loss_mask:")  mk=$(i+1)
        if($i=="loss_dice:")  dc=$(i+1)
        if($i=="time:")       tm=$(i+1)
        if($i=="lr:")         lr=$(i+1)
        if($i=="max_mem:")    mm=$(i+1)
        if($i=="eta:")        eta=$(i+1)
      }
      printf "iter %6s | loss %8s (ce %7s mask %6s dice %6s) | %5ss/it | lr %s | VRAM %s | 残り %s\n",
             it, tl, ce, mk, dc, tm, lr, mm, eta
  }'
