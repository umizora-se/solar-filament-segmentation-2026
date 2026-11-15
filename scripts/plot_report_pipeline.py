"""Report figure: end-to-end pipeline diagram -> docs/report/fig_pipeline.png"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(7.0, 2.35)); ax.set_xlim(-1.5, 101.5); ax.set_ylim(0, 34); ax.axis("off")
BL, GR, OR = "#dbe8f7", "#e3f1de", "#fbe7cf"


def box(x, y, w, h, t, c, fs=6.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8", fc=c, ec="#444", lw=0.7))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs, linespacing=1.25)


def arrow(x1, y1, x2, y2):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#333", mutation_scale=7))


ax.text(0, 32.6, "Training (5 folds $\\times$ 2 seeds)", fontsize=7.2, weight="bold")
box(0, 22, 17, 8, "707 photos\n1,154 records, 1 class\nfolds by photo\n(station $\\times$ cycle)", BL, fs=6.2)
box(21, 22, 20, 8, "Mask DINO R50, native\n2048 px, COCO init\n5,000 it, point-sampled\nmask loss", BL)
box(45, 22, 17, 8, "seed 1: SWA of last\n4 checkpoints\nseed 2: final weights", BL)
arrow(17.4, 26, 20.6, 26); arrow(41.4, 26, 44.6, 26)
box(66, 22, 34, 8, "OOF PQ on all 707 photos\n$\\rightarrow$ choose $\\alpha$, threshold, area, keep by nested CV", OR)
arrow(62.4, 26, 65.6, 26)
ax.text(0, 17.8, "Inference (per test image, per fold $k$)", fontsize=7.2, weight="bold")
box(0, 4, 15, 11, "test image\n2048 $\\times$ 2048\n(as is, no\nresizing)", GR)
box(19, 4, 20, 11, "2 models $\\times$ 8 D4 views\n= 16 views\ncluster by IoU>0.5,\nprobability merge", GR)
box(43, 4, 17, 11, "vote score\n$s\\,(v/16)^{\\alpha}$,\n$\\alpha$=0.75", GR)
box(64, 4, 17, 11, "5-fold fusion\nIoU>0.5, $\\geq$2 folds,\npixel majority", GR)
box(85, 4, 15, 11, "paint by score,\nthr 0.32, area $\\geq$200,\nkeep $\\geq$80%\n$\\rightarrow$ RLE CSV", GR)
for a, b in ((15.4, 18.6), (39.4, 42.6), (60.4, 63.6), (81.4, 84.6)):
    arrow(a, 9.5, b, 9.5)
arrow(31, 21.6, 29.5, 15.6)
ax.text(32.5, 17.8, "weights", fontsize=6, style="italic", color="#444")
fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
out = REPO / "docs/report/fig_pipeline.png"; fig.savefig(out, dpi=220); print("[write]", out)
