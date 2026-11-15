"""Figures for the technical report (English): PQ progression and the ablation of every tried change.

All numbers are measured values recorded in work_dirs/*summary.txt (see docs/DECISION_RULES.md).
  python3 scripts/plot_report_figures.py  ->  docs/report/fig_progress.png, docs/report/fig_ablation.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "report"
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, REJ = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.family": "DejaVu Sans"})

# ---------------- Figure 1: progression of the adopted pipeline ----------------
steps = ["Baseline", "+ post-proc\nthresholds", "+ 4-view\nTTA", "+ 8-view\nTTA", "+ verifiers\n(x3)",
         "Native-res\ntraining", "+ 5-fold\nensemble", "+ 2nd seed\n+ vote score", "+ SWA", "+ fine\npost-proc", "+ retain\nfraction"]
dates = ["Sep 1", "Sep 3", "Sep 4", "Sep 5", "Sep 8", "Oct 12-15", "Oct 15", "Nov 10", "Nov 13", "Nov 13", "Nov 16"]
fold0 = [0.3012, 0.3901, 0.4198, 0.4268, 0.4322, 0.4600, None, None, None, None, None]
oof = [None, None, None, 0.4162, 0.4228, 0.4711, None, 0.4724, 0.4726, 0.4748, 0.4777]
lb = [None, 0.35, 0.37, 0.39, 0.40, 0.40, 0.41, 0.41, 0.41, 0.41, 0.41]

fig, ax = plt.subplots(figsize=(12, 5.2), facecolor=SURF)
ax.set_facecolor(SURF)


def series(vals, color, marker, label, dy, fmt, special=None):
    xs = [i for i, v in enumerate(vals) if v is not None]
    ys = [vals[i] for i in xs]
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=7, mec=SURF, mew=1.5, label=label, zorder=3)
    for i, y in zip(xs, ys):
        off, ha = (special or {}).get(i, ((0, dy), "center"))
        ax.annotate(fmt.format(y), (i, y), xytext=off, textcoords="offset points", ha=ha, fontsize=8.5, color=INK2)


series(fold0, BLUE, "o", "fold-0 validation (142 photos)", 8, "{:.4f}",
       {0: ((8, -10), "left"), 5: ((8, -12), "left")})
series(oof, ORANGE, "s", "5-fold OOF (707 photos) - selection metric", 8, "{:.4f}",
       {3: ((0, -15), "center"), 4: ((0, -15), "center"), 5: ((-6, 8), "right"), 7: ((0, -15), "center"),
        8: ((0, 8), "center")})
series(lb, AQUA, "^", "public LB (subset of test, 2 decimals)", -15, "{:.2f}")
ax.axhline(0.340, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(len(steps) - 0.6, 0.341, "inter-annotator agreement 0.340", va="bottom", ha="right", fontsize=8.5, color=MUTED)
ax.set_xticks(range(len(steps)))
ax.set_xticklabels([f"{s}\n{d}" for s, d in zip(steps, dates)], fontsize=8.5)
ax.set_ylim(0.28, 0.50)
ax.set_ylabel("Panoptic Quality")
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_title("Adopted pipeline and PQ over time", loc="left", fontsize=13, fontweight="bold")
fig.savefig(OUT / "fig_progress.png", dpi=170, facecolor=SURF, bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: ablation of every tried change ----------------
items = [  # (name, delta PQ, adopted, basis)
    ("Post-processing thresholds (score, area)", 0.089, True, "fold0"),
    ("Native-resolution training (no 2x downscale)", 0.0549, True, "OOF"),
    ("4-view flip TTA", 0.030, True, "fold0"),
    ("8-view D4 TTA", 0.007, True, "fold0"),
    ("Semantic verifiers x3 (later dropped*)", 0.0066, True, "OOF"),
    ("TTA merge by probability averaging", 0.0029, True, "fold0"),
    ("Retain fraction 0.8 + threshold 0.32 (nested CV)", 0.0029, True, "OOF"),
    ("Fine post-processing grid (nested CV)", 0.0021, True, "OOF"),
    ("Verifier ensemble 3 -> 5", 0.0018, False, "OOF"),
    ("SWA of last 4 checkpoints", 0.0011, True, "OOF"),
    ("Second seed, per-fold prob fusion", 0.0010, True, "OOF"),
    ("TTA vote score", 0.0006, True, "OOF"),
    ("Mask binarisation threshold", 0.0, False, "fold0"),
    ("Shape filter", 0.0, False, "fold0"),
    ("Hole filling of instance masks", 0.0, False, "OOF"),
    ("Removal of small stray parts", 0.0, False, "OOF"),
    ("Low-contrast filter (ring vs. mask intensity)", 0.0, False, "OOF"),
    ("Overlap test against the smaller mask", -0.0001, False, "OOF"),
    ("Keep only the largest connected part", -0.0009, False, "OOF"),
    ("Relative / limb-aware thresholds", 0.0, False, "OOF"),
    ("Size-dependent thresholds", -0.0004, False, "OOF"),
    ("Merging fragmented predictions", -0.0009, False, "OOF"),
    ("Top-k cap per image", -0.0016, False, "OOF"),
    ("Fusion with RF-DETR", -0.0024, False, "fold0"),
    ("Majority-vote verifier targets", -0.0028, False, "OOF"),
    ("Longer schedule 8000 it (native)", -0.0032, False, "fold0"),
    ("Pseudo-labels on unlabeled GONG frames", -0.0044, False, "fold0"),
    ("IoU-aware classification target", -0.0072, False, "fold0"),
    ("Stride-2 mask head (native, 1024 crops)", -0.0078, False, "fold0"),
    ("Swin-L backbone (downscaled)", -0.0102, False, "fold0"),
    ("4x longer schedule (downscaled)", -0.012, False, "fold0"),
    ("Label transfer to neighbouring GONG frames", -0.0121, False, "fold0"),
    ("Learned re-ranking", -0.0134, False, "fold0"),
    ("Uniform mask dilation", -0.016, False, "fold0"),
    ("Focal-Tversky loss", -0.027, False, "fold0"),
    ("MetaSeg-style rescoring", -0.030, False, "fold0"),
    ("Mask replacement by SAM", -0.063, False, "fold0"),
    ("Stride-2 mask head (downscaled)", -0.130, False, "fold0"),
    ("Crop refiner (2nd-stage U-Net)", -0.220, False, "fold0"),
]
CLIP = -0.07
fig, bx = plt.subplots(figsize=(11, 10.5), facecolor=SURF)
bx.set_facecolor(SURF)
y = np.arange(len(items))[::-1]
for yi, (name, d, ok, basis) in zip(y, items):
    shown = max(d, CLIP)
    bx.barh(yi, shown, height=0.62, color=BLUE if ok else REJ, edgecolor=SURF, lw=1.5, zorder=3)
    if d < CLIP:
        bx.text(CLIP - 0.002, yi, f"< {d:+.3f}", va="center", ha="right", fontsize=8, color=INK2)
    else:
        lab = "+/-0.000" if d == 0 else (f"{d:+.4f}" if abs(d) < 0.01 else f"{d:+.3f}")
        bx.text(shown + (0.002 if shown >= 0 else -0.002), yi, lab, va="center",
                ha="left" if shown >= 0 else "right", fontsize=8, color=INK2)
bx.set_yticks(y)
bx.set_yticklabels([f"{n}  [{b}]" for n, _, _, b in items], fontsize=8.5)
for t, (_, _, ok, _) in zip(bx.get_yticklabels(), items):
    t.set_color(INK if ok else MUTED)
bx.axvline(0, color=AXIS, lw=1, zorder=2)
bx.set_xlim(CLIP - 0.025, 0.102)
bx.set_xlabel("Change in PQ relative to the pipeline at that time")
bx.grid(axis="x", color=GRID, lw=0.8)
bx.set_axisbelow(True)
for s in ("top", "right", "left"):
    bx.spines[s].set_visible(False)
bx.tick_params(axis="y", length=0)
bx.legend(handles=[Patch(color=BLUE, label="adopted"), Patch(color=REJ, label="rejected")],
          loc="lower right", frameon=False, fontsize=9)
bx.set_title(f"Every change we tried ({len(items)} experiments)", loc="left", fontsize=13, fontweight="bold")
bx.text(0, -0.085, "[ ] = measurement basis. *Verifiers helped the downscaled model but not the native-resolution "
        "model (OOF 0.4711 without vs 0.4694 with), so the final pipeline does not use them.",
        transform=bx.transAxes, fontsize=8, color=INK2)
fig.savefig(OUT / "fig_ablation.png", dpi=170, facecolor=SURF, bbox_inches="tight")
plt.close(fig)
print("[write]", OUT / "fig_progress.png", OUT / "fig_ablation.png")
