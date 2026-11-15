# Reproducing the final submission

Final file: `release/submission_final.csv` (OOF PQ 0.4777). Notebook: `notebooks/pipeline_from_weights.ipynb`. Part A runs sections 2, 3 and 5 below (data, training, SWA, OOF, post-processing search; off by default, `RUN_TRAINING = True` to run). Part B runs section 4 from the released weights in `release/weights/`, or from the Part A weights.

## 1. Environment

| Item | Version or spec |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB (training needs about 10 GB at native 2048 px) |
| OS | Ubuntu 22.04 on WSL2 |
| Python | 3.10.12 |
| PyTorch | 2.9.1 (CUDA 12.8) |
| detectron2 | 0.6, `facebookresearch/detectron2` at commit `a2f4a87` |
| Mask DINO | `IDEA-Research/MaskDINO` at commit `3831d85` |

Install detectron2 with `pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git@a2f4a8771ab77e8411c26b27f24f9489a28a2453'`.

Our changes to Mask DINO are in `patches/maskdino_filament.patch`. One new file, `patches/point_mask.py`, goes into `maskdino/modeling/`. Apply both before building the CUDA ops:

```bash
cd external/MaskDINO
git apply ../../patches/maskdino_filament.patch
cp ../../patches/point_mask.py maskdino/modeling/
cd maskdino/modeling/pixel_decoder/ops && sh make.sh
```

## 2. Data and folds

Put the competition data under `datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026/`; on Kaggle, `/kaggle/input/...` is also detected. Then run:

```bash
python3 src/prepare_data.py   # writes data/magfilo_singleclass.json and data/folds.csv
```

`prepare_data.py` converts all annotations to a single class and splits the photos into 5 folds with StratifiedGroupKFold. Each photo is one group, and the strata are GONG station × solar-cycle bin.

## 3. Training: 5 folds × 2 seeds (about 8 h per run)

Seed 1 uses `configs/maskdino_filament_r50_full2048.yaml` for fold 0 and `configs/_s3_f{k}.yaml` for folds 1–4. Seed 2 uses `configs/_seed2_f{k}.yaml`. The configs differ only in the fold. `SEED=-1` gives a different random seed for each run.

```bash
for k in 0 1 2 3 4; do
  cfg=configs/_s3_f$k.yaml; [ $k = 0 ] && cfg=configs/maskdino_filament_r50_full2048.yaml
  bash scripts/train_with_watchdog.sh $cfg work_dirs/r50_full2048_f$k work_dirs/train_f$k.log 5000 $k
  bash scripts/train_with_watchdog.sh configs/_seed2_f$k.yaml work_dirs/r50_full2048_seed2_f$k \
       work_dirs/train_seed2_f$k.log 5000 $k
done
```

`train_with_watchdog.sh` resumes from the last checkpoint if a run stops. Checkpoints are saved every 250 iterations. Keep `model_0004249/4499/4749/4999.pth` of seed 1, which are needed for SWA.

Each run checks that its training record count matches the fold definition and aborts if it does not. The expected counts are 928, 939, 920, 915 and 914 for folds 0–4.

## 4. Test inference to CSV (about 10 h)

```bash
bash scripts/reproduce_final.sh
```

The script uses the released weights (seed-1 SWA and seed 2 per fold). If you trained the models yourself, first build the seed-1 SWA weights:

```bash
for k in 0 1 2 3 4; do
  python3 src/swa_average.py --run-dir work_dirs/r50_full2048_f$k --iters 4249 4499 4749 4999 \
      --out work_dirs/r50_full2048_f$k/model_swaA.pth
done
```

For each fold the script runs these steps:

1. 16-view TTA of the two models, merged by probability averaging, with the vote count recorded (`src/predict_submit.py`).
2. Vote score with α = 0.75 (`src/vote_rescore.py`).

It then fuses the 5 folds (`src/ensemble.py --min-votes 2`) and writes the CSV with threshold 0.32, area 200 and retain fraction 0.8 (`src/json_to_csv.py`; an instance is dropped if less than 80 % of its own pixels remain after painting). Finally it compares the CSV byte for byte with the submitted file.

## 5. Out-of-fold evaluation and choice of post-processing

1. **OOF predictions.** Run `src/predict_submit.py --split val --fold k` with the same two weights and flags as step 1 of section 4.
2. **Merge and search.** Merge the folds and run `scripts/pp_fine.py`. It searches α × threshold × area (275 combinations) and scores the search with nested cross-validation.
   Then `scripts/pp_retain.py` adds the retain fraction to the search (threshold × area × fraction, 120 combinations, nested CV); it chose threshold 0.32, area 200, fraction 0.8 in every outer fold (+0.0029 OOF PQ).
3. **Compare settings.** `scripts/paired_oof.py A.json B.json` gives the per-fold differences and a paired image bootstrap.
4. **Gap analysis.** `scripts/lb_gap.py` reproduces the OOF-versus-leaderboard analysis in §7 of the report.

## Module map

| Module | Role |
|---|---|
| `src/filament_dataset.py`, `src/filament_mapper.py` | Data registration and training augmentation |
| `src/train_filament.py`, `src/filament_config.py` | Training, memory settings and config flags |
| `src/predict_submit.py` | Inference, D4 TTA with probability merge and vote counting, and CSV validation |
| `src/ensemble.py` | Cross-model instance fusion (IoU clustering, minimum votes, pixel majority) |
| `src/postproc.py`, `src/json_to_csv.py` | Pixel-exclusive painting, thresholds, submission writing |
| `src/pq_metric.py`, `src/tune_postproc.py` | Official pooled PQ and the post-processing search |
| `src/swa_average.py`, `src/vote_rescore.py` | SWA and the vote score |
| `scripts/*_stage.py` | The experiment drivers, kept as the record behind every number in the report |
