"""Figure: controlled overlap experiment.

Two classes of the Ertam & Kaya firewall dataset, each held at a FIXED training
frequency while its position relative to its nearest competing class is varied.
Overlap is re-measured at every level (N3, the 1-NN leave-one-out error), so the
x axis is a measured quantity rather than the nominal manipulation strength.

Identity is carried by marker shape as well as hue, and the two models by panel,
so the figure survives greyscale printing.

Usage
-----
    python fig_overlap_controlled.py [input.csv] [output_stem]

Both arguments are optional. By default the script reads
`table_overlap_controlled.csv` from its own directory and writes
`fig_overlap_controlled.png` and `.pdf` beside it, so the script and its data
can be kept together in one folder and re-run anywhere.

The input is the per-run output of controlled_experiment.py, filtered to the
overlap experiment and to classes with a valid dose-response. It needs the
columns: target_class, model, alpha, measured_N3, target_f1, train_n_target,
blend_partner_class.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "table_overlap_controlled.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
    HERE, "fig_overlap_controlled")

if not os.path.exists(SRC):
    raise SystemExit(
        f"Input not found: {SRC}\n"
        "Pass the path to the combined overlap results as the first argument, "
        "or put table_overlap_controlled.csv next to this script.")

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"

STYLE = {                      # class -> (hue, marker)
    "drop": (BLUE, "o"),
    "deny": (ORANGE, "s"),
}
LABEL = {
    "drop": "drop  (9,638 training rows)",
    "deny": "deny  (11,240 training rows)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.4,
    "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 7.0, "ytick.labelsize": 7.0,
})

d = pd.read_csv(SRC)
fig, axes = plt.subplots(1, 2, figsize=(6.15, 2.72), sharey=True)

for ax, model, title in zip(axes, ["mlp", "xgb"],
                            ["(a)  Multilayer perceptron",
                             "(b)  XGBoost"]):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
    ax.set_axisbelow(True)
    for cls in ["drop", "deny"]:
        hue, mk = STYLE[cls]
        g = d[(d.target_class == cls) & (d.model == model)]
        cell = g.groupby("alpha").agg(N3=("measured_N3", "mean"),
                                      F1=("target_f1", "mean"),
                                      sd=("target_f1", "std")).reset_index()
        ax.errorbar(cell.N3, cell.F1, yerr=cell.sd, fmt="none",
                    ecolor=hue, elinewidth=0.8, capsize=1.8,
                    capthick=0.8, alpha=0.9, zorder=3)
        ax.plot(cell.N3, cell.F1, linestyle="none", marker=mk,
                markersize=4.2, markerfacecolor=hue, markeredgecolor="white",
                markeredgewidth=0.7, label=LABEL[cls], zorder=4)
        sl, ic, r, p, se = stats.linregress(g.measured_N3, g.target_f1)
        xs = np.linspace(cell.N3.min(), cell.N3.max(), 50)
        ax.plot(xs, ic + sl * xs, color=hue, linewidth=1.1,
                linestyle="--" if cls == "deny" else "-", alpha=0.75, zorder=2)

    # One pooled slope annotation per panel, in ink not series colour. It goes
    # in whichever corner the data vacates: the MLP points fall left-to-right so
    # the top right is clear, the XGBoost points sit along the top so the bottom
    # right is.
    h = d[d.model == model]
    sl, ic, r, p, se = stats.linregress(h.measured_N3, h.target_f1)
    if p < 1e-10:
        ps = "$p$ < 10$^{-10}$"
    elif p < 0.001:
        ps = f"$p$ = {p:.0e}".replace("e-0", r"$\times$10$^{-") + "}$"
    else:
        ps = f"$p$ = {p:.3f}"
    txt = (f"pooled slope {sl:+.3f}\n"
           f"95% CI [{sl-1.96*se:+.3f}, {sl+1.96*se:+.3f}]\n" + ps)
    top = model == "mlp"
    ax.text(0.97, 0.94 if top else 0.06, txt,
            transform=ax.transAxes, ha="right", va="top" if top else "bottom",
            fontsize=6.6, color=INK2, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=GRID, linewidth=0.5), zorder=6)
    ax.set_title(title, fontsize=7.8, color=INK, loc="left", pad=5)
    ax.set_xlabel("Measured class overlap  (N3)", fontsize=7.4, color=INK)
    ax.set_xlim(-0.02, 0.40)
    ax.set_ylim(0.40, 1.04)

axes[0].set_ylabel("$F_1$ of the target class", fontsize=7.4, color=INK)
axes[0].legend(frameon=False, fontsize=6.8, loc="lower left",
               handletextpad=0.4, labelspacing=0.35)

fig.subplots_adjust(left=0.098, right=0.985, top=0.885, bottom=0.175,
                    wspace=0.07)
for ext, dpi in [("png", 600), ("pdf", 600)]:
    fig.savefig(f"{OUT}.{ext}", dpi=dpi, facecolor="white",
                bbox_inches="tight", pad_inches=0.02)
print("written:", OUT + ".png", OUT + ".pdf")
im = plt.imread(f"{OUT}.png")
print("png pixels:", im.shape[1], "x", im.shape[0],
      f"({im.shape[1]/600:.2f} in at 600 dpi)")
