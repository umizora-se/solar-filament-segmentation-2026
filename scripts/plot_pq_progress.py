"""PQ の推移と、試した施策の効果を図にする（docs/figures/pq_progress.png）。

値はすべて work_dirs/*summary.txt と docs/ の測定台帳に記録した実測値。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401  日本語フォント（IPAexGothic）
import numpy as np

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # 検証済み（light, all-pairs PASS）
REJ = "#c3c2b7"

plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig = plt.figure(figsize=(12.5, 13.5), facecolor=SURF)
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.28)

# ───────────── 1. 採用した構成の推移 ─────────────
ax = fig.add_subplot(gs[0]); ax.set_facecolor(SURF)
steps = ["素の予測", "＋後処理の\n閾値調整", "＋4方向\nTTA", "＋8方向\nTTA", "＋検証器\n3種", "原寸学習\n（縮小の除去）", "＋5fold\nアンサンブル", "＋2シード融合\n（改善は未確認）"]
dates = ["9/1", "9/3", "9/4", "9/5", "9/8", "10/12-15", "10/15", "11/9"]
val = [0.3012, 0.3901, 0.4198, 0.4268, 0.4322, 0.4600, None, 0.4565]
oof = [None, None, None, 0.4162, 0.4228, 0.4711, None, 0.4721]
lb  = [None, 0.35, 0.37, 0.39, 0.40, 0.40, 0.41, 0.41]
x = np.arange(len(steps))

def series(vals, color, marker, label, dy, fmt, special=None):
    xs = [i for i, v in enumerate(vals) if v is not None]
    ys = [vals[i] for i in xs]
    ax.plot(xs, ys, color=color, lw=2, marker=marker, ms=8, mec=SURF, mew=2, label=label, zorder=3)
    for i, y in zip(xs, ys):
        off, ha = (special or {}).get(i, ((0, dy), "center"))
        ax.annotate(fmt.format(y), (i, y), xytext=off, textcoords="offset points",
                    ha=ha, fontsize=9, color=INK2)

# ラベルの重なりを避けるため、近接する点だけ位置をずらす
series(val, BLUE, "o", "fold0 の検証（142枚）", 9, "{:.4f}",
       special={0: ((10, -10), "left"), 5: ((10, -14), "left"), 7: ((0, -16), "center")})
series(oof, ORANGE, "s", "OOF（707枚・5fold）← private の推定", -16, "{:.4f}",
       special={5: ((0, 9), "center"), 7: ((0, 9), "center")})
series(lb, AQUA, "^", "public LB（test の一部・小数2桁）", -16, "{:.2f}",
       special={1: ((0, 9), "center")})

ax.axhline(0.340, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
ax.text(len(steps) - 0.55, 0.340, "人間同士の一致度 0.340", va="bottom", ha="right", fontsize=9, color=MUTED)
ax.axhline(0.49, color=MUTED, lw=1, ls=(0, (1, 2)), zorder=1)
ax.text(len(steps) - 0.55, 0.49, "public の正当な最上位とみられる値 0.49（参考）", va="bottom", ha="right", fontsize=9, color=MUTED)

ax.set_xticks(x); ax.set_xticklabels([f"{s}\n{d}" for s, d in zip(steps, dates)], fontsize=9.5)
ax.set_ylim(0.28, 0.51); ax.set_ylabel("PQ")
ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
ax.legend(loc="lower left", bbox_to_anchor=(0.30, 0.02), frameon=False, fontsize=9.5)
ax.set_title("採用した構成と PQ の推移", loc="left", fontsize=14, fontweight="bold", pad=24)
ax.text(0, 1.035, "最大の改善は「後処理の閾値調整」(+0.089) と「原寸学習」(OOF +0.055)。"
        "5fold は推論時のみの手法で OOF・fold0 は測れず、線は前後をつないだだけ",
        transform=ax.transAxes, fontsize=9.5, color=INK2)

# ───────────── 2. 試した施策の効果 ─────────────
bx = fig.add_subplot(gs[1]); bx.set_facecolor(SURF)
items = [  # (名前, Δ, 採用, 測定基盤)
    ("後処理の閾値調整", 0.089, True, "fold0"),
    ("原寸学習（学習時の2倍縮小を除去）", 0.0549, True, "OOF"),
    ("4方向TTA", 0.030, True, "fold0"),
    ("8方向TTA", 0.007, True, "fold0"),
    ("検証器3種（スコアの付け直し）", 0.0066, True, "OOF"),
    ("TTAの確率平均統合", 0.0029, True, "fold0"),
    ("検証器のメンバー追加（3→5種）", 0.0018, False, "OOF"),
    ("2シード融合（確率平均）", 0.0010, False, "OOF"),
    ("TTA の票数でスコア補正（1シード）", 0.0006, False, "OOF"),
    ("TTA の票数でスコア補正（2シード融合）", 0.0003, False, "OOF"),
    ("マスク2値化閾値の探索", 0.000, False, "fold0"),
    ("形状フィルタ", 0.000, False, "fold0"),
    ("予測の統合（過分割対策）", -0.0009, False, "OOF"),
    ("RF-DETR との融合", -0.0024, False, "fold0"),
    ("検証器の教師を多数派に", -0.0028, False, "OOF"),
    ("学習延長 8000 iter（原寸）", -0.0032, False, "fold0"),
    ("疑似ラベル（外部GONG 1277枚）", -0.0044, False, "fold0"),
    ("マスクの高解像度化（原寸1024クロップ）", -0.0078, False, "fold0"),
    ("Swin-L バックボーン", -0.0102, False, "fold0"),
    ("学習4倍延長（縮小下）", -0.012, False, "fold0"),
    ("近傍GONGフレームへのラベル転写", -0.0121, False, "fold0"),
    ("学習ベースの再ランキング", -0.0134, False, "fold0"),
    ("一律のマスク膨張", -0.016, False, "fold0"),
    ("Focal-Tversky 損失", -0.027, False, "fold0"),
    ("MetaSeg 式の再スコアリング", -0.030, False, "fold0"),
    ("SAM でマスクを描き直す", -0.063, False, "fold0"),
    ("マスクの高解像度化（縮小下）", -0.130, False, "fold0"),
    ("crop refiner（2段目U-Net）", -0.220, False, "fold0"),
]
CLIP = -0.07
y = np.arange(len(items))[::-1]
for yi, (name, d, ok, basis) in zip(y, items):
    shown = max(d, CLIP)
    bx.barh(yi, shown, height=0.62, color=BLUE if ok else REJ, edgecolor=SURF, lw=2, zorder=3)
    if d < CLIP:
        bx.text(CLIP - 0.002, yi, f"◀ {d:+.3f}", va="center", ha="right", fontsize=9, color=INK2)
    else:
        bx.text(shown + (0.002 if shown >= 0 else -0.002), yi,
                "±0.000" if d == 0 else (f"{d:+.4f}" if abs(d) < 0.01 else f"{d:+.3f}"),
                va="center", ha="left" if shown >= 0 else "right", fontsize=9, color=INK2)
bx.set_yticks(y)
bx.set_yticklabels([f"{n}  [{b}]" for n, _, _, b in items], fontsize=9.5)
for t, (_, _, ok, _) in zip(bx.get_yticklabels(), items):
    t.set_color(INK if ok else MUTED)
bx.axvline(0, color=AXIS, lw=1, zorder=2)
bx.axvline(0.005, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=2)
bx.text(0.0055, y[0] + 0.7, "目安 +0.005（未満でも汎化の改善が確かなら採用）", fontsize=9, color=MUTED, va="bottom")
bx.set_xlim(CLIP - 0.022, 0.102); bx.set_xlabel("ΔPQ（その時点の構成との差）")
bx.grid(axis="x", color=GRID, lw=0.8); bx.set_axisbelow(True)
for s in ["top", "right", "left"]: bx.spines[s].set_visible(False)
bx.tick_params(axis="y", length=0)
from matplotlib.patches import Patch
bx.legend(handles=[Patch(color=BLUE, label="採用"), Patch(color=REJ, label="不採用")],
          loc="lower right", frameon=False, fontsize=9.5)
bx.set_title(f"試した施策の効果（{len(items)}件）", loc="left", fontsize=14, fontweight="bold", pad=24)
bx.text(0, 1.012, "採用はすべて「モデルの出力の選び方」か「入力の正しさ」に関わるもの。"
        "マスクの作り直し・外部モデル・学習レコードの水増しはすべて悪化。[ ] は測定基盤",
        transform=bx.transAxes, fontsize=9.5, color=INK2)

fig.savefig("docs/figures/pq_progress.png", dpi=150, facecolor=SURF, bbox_inches="tight")
print("[write] docs/figures/pq_progress.png")
