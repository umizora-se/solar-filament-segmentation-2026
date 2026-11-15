# Solar Filament Segmentation Challenge 2026 — Mask DINO at native resolution

Final submission: `release/submission_final.csv` (five-fold out-of-fold PQ 0.4777 on the 707 training photos; public leaderboard 0.41).
Technical report: `docs/technical_report.pdf`.

## Reproduce the predictions without retraining

1. Environment (Python 3.10, CUDA 12.8, one GPU with 16 GB): `pip install -r requirements.txt`, then install
   detectron2 0.6 (commit `a2f4a87`) and Mask DINO as described in `docs/REPRODUCE.md` (clone `IDEA-Research/MaskDINO` at commit `3831d85` into `external/MaskDINO`, apply `patches/maskdino_filament.patch`, copy `patches/point_mask.py`, build the CUDA ops).
2. Put the competition data where `src/filament_dataset.py` looks for it (`FILAMENT_ROOT`, or `datasets/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026`; `/kaggle/input/...` is detected automatically).
3. Run `notebooks/pipeline_from_weights.ipynb`. It joins the weights in `release/weights/` (10 files in parts below 100 MB, SHA-256 in `MANIFEST.json`), runs the full inference pipeline, writes the CSV and compares it with `release/submission_final.csv`. The default runs 3 test images as a check; set `N_IMAGES = None` for the full run (about 10 h). The notebook's last cell recomputes the OOF PQ of the final setting from `release/oof_swa2_a0.75.json`.

## Layout

| Path | Content |
|---|---|
| `src/` | Modules: data, training, inference (`predict_submit.py`), fusion (`ensemble.py`), vote score (`vote_rescore.py`), SWA (`swa_average.py`), post-processing, PQ metric |
| `configs/` | Training and inference configs (one per fold and seed) |
| `patches/` | Our changes to Mask DINO and the point-sampled mask loss |
| `release/` | Weights, final CSV, OOF predictions, final post-processing settings |
| `scripts/` | Experiment drivers behind every number in the report; `reproduce_final.sh` rebuilds the CSV from the weights on the command line |
| `data/folds.csv` | Photo-level fold assignment |
| `docs/` | `REPRODUCE.md` (training and evaluation commands), decision rules |

Training (about 8 h per run, 5 folds × 2 seeds) is described in `docs/REPRODUCE.md`; the trained weights are provided, so retraining is not needed.

## License

The code, including the Mask DINO patches in `patches/`, is released under the Apache License 2.0 (`LICENSE`; copyright and scope in `NOTICE`). The trained weights in `release/weights/` and the predictions in `release/` (OOF and final CSV) are provided to reproduce the competition results; other uses are subject to the terms of the MAGFiLO dataset and of the competition. Mask DINO and Detectron2 keep their own licenses.
