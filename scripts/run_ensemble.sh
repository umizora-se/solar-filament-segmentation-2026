#!/usr/bin/env bash
# 5-fold アンサンブルの検証と本番推論を一気通貫で行う。
# 前提: fold 0-4 の work_dirs/r50_f{0-4}/model_final.pth が揃っていること。
#
#   ./scripts/run_ensemble.sh oof       # OOF単一モデルの正直なval-PQ（707枚）
#   ./scripts/run_ensemble.sh check     # fold0 valで5モデル融合の楽観的キルスイッチ
#   ./scripts/run_ensemble.sh test      # test 180枚を5モデルで融合 → submission
set -e
cd .
CFG=configs/maskdino_filament_r50_1024crop.yaml
MODE="${1:-check}"

# 前提チェック
for f in 0 1 2 3 4; do
  [ -f "work_dirs/r50_f${f}/model_final.pth" ] || { echo "★ fold $f のモデルが無い"; exit 1; }
done

infer() {  # infer <fold> <stems_file|test> <out> [tta_views]
  local f=$1 target=$2 out=$3 views=${4:-4}
  local extra=""
  [ "$target" != "test" ] && extra="--split val --fold 0 --stems-file $target"
  [ "$target" == "test" ] && extra="--split test"
  python3 -u src/predict_submit.py \
    --config "$CFG" --weights "work_dirs/r50_f${f}/model_final.pth" \
    $extra --tta --tta-views "$views" \
    --dump-raw "$out" --out "/tmp/ens_sub_f${f}.csv"
}

case "$MODE" in
  oof)
    echo "===== OOF: 各foldのvalを、そのfoldのモデルだけで予測（正直なbaseline）====="
    mkdir -p work_dirs/ens
    : > /tmp/oof_merge.jsonl
    for f in 0 1 2 3 4; do
      echo "--- fold $f の val を model $f で推論 ---"
      python3 -u src/predict_submit.py --config "$CFG" \
        --weights "work_dirs/r50_f${f}/model_final.pth" \
        --split val --fold "$f" --tta --tta-views 4 \
        --dump-raw "work_dirs/r50_f${f}/val_raw_tta.json" \
        --out "/tmp/oof_f${f}.csv"
    done
    python3 - <<'PY'
import json; from pathlib import Path
merged={}
for f in range(5):
    d=json.loads(Path(f"work_dirs/r50_f{f}/val_raw_tta.json").read_text())["preds"]
    merged.update(d)   # 各stemは1つのfoldのvalにしか無いので衝突しない
Path("work_dirs/ens/oof_val.json").write_text(json.dumps({"preds":merged}))
print(f"OOF統合: {len(merged)} 画像")
PY
    python3 -u src/tune_postproc.py --preds work_dirs/ens/oof_val.json --fold -1 \
      --out work_dirs/ens/postproc_oof.json \
      --score-thr 0.25,0.30,0.35,0.40 --min-area 100,200,300 || true
    ;;

  check)
    echo "===== キルスイッチ: fold0 val(142枚)を5モデルで予測して融合（楽観的）====="
    mkdir -p work_dirs/ens
    python3 -c "import json,sys; sys.path.insert(0,'src'); from filament_dataset import get_val_photo_dicts; \
      json.dump([d['stem'] for d in get_val_photo_dicts(0)], open('/tmp/f0_val_stems.json','w'))"
    for f in 0 1 2 3 4; do
      echo "--- model $f を fold0 val に推論 ---"
      infer "$f" /tmp/f0_val_stems.json "work_dirs/ens/f0val_by_m${f}.json" 4
    done
    for MV in 2 3; do
      python3 -u src/ensemble.py \
        --preds work_dirs/ens/f0val_by_m{0,1,2,3,4}.json \
        --out "work_dirs/ens/f0val_ens_mv${MV}.json" --min-votes "$MV"
      echo "### min_votes=$MV のPQ ###"
      python3 -u src/tune_postproc.py --preds "work_dirs/ens/f0val_ens_mv${MV}.json" --fold 0 \
        --out "/tmp/pp_ens_mv${MV}.json" \
        --score-thr 0.15,0.20,0.25,0.30,0.35,0.40 --min-area 100,200,300 | tail -4
    done
    echo "比較: fold0単独(4方向TTA)は val-PQ 0.4198"
    ;;

  test)
    echo "===== 本番: test 180枚を5モデルで融合 ====="
    mkdir -p work_dirs/ens
    for f in 0 1 2 3 4; do
      echo "--- model $f を test に推論 ---"
      infer "$f" test "work_dirs/r50_f${f}/test_raw_tta.json" 4
    done
    python3 -u src/ensemble.py \
      --preds work_dirs/r50_f{0,1,2,3,4}/test_raw_tta.json \
      --out work_dirs/ens/test_ens.json --min-votes "${MV:-2}"
    echo "→ work_dirs/ens/test_ens.json を tune_postproc の設定でCSV化して提出"
    ;;
esac
