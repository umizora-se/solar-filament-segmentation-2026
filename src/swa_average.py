"""Stochastic Weight Averaging of detectron2 checkpoints (Izmailov et al., 2018).

Averages the "model" state dicts of several checkpoints of one run. The backbone uses FrozenBN and the
heads GroupNorm, so no batch-norm statistics need to be recomputed after averaging.

    python src/swa_average.py --run-dir work_dirs/r50_full2048_f0 --iters 4249 4499 4749 4999 \
        --out work_dirs/r50_full2048_f0/model_swaA.pth
"""
import argparse
from pathlib import Path

import torch


def average_checkpoints(paths: list[Path]) -> dict:
    acc = None
    for p in paths:
        m = torch.load(p, map_location="cpu", weights_only=False)["model"]
        if acc is None:
            acc = {n: v.double().clone() for n, v in m.items()}
        else:
            for n, v in m.items():
                acc[n] += v.double()
    return {n: (v / len(paths)).to(torch.float32) for n, v in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iters", type=int, nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    paths = [Path(a.run_dir) / f"model_{it:07d}.pth" for it in a.iters]
    torch.save({"model": average_checkpoints(paths), "iteration": a.iters[-1]}, a.out)
    print(f"[swa] averaged {len(paths)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
