"""
Controlled rarity sweep on Dataset A: the number of training instances of one target
class is reduced by subsampling (training split only; the test split is unchanged),
and the F1 of that class is measured for the MLP (no resampling) and XGBoost. Points
are the mean over three seeds; bars are one standard deviation. The rightmost point of
each panel is the full class.

Reads table_rarity_sweep_from_log.csv from the folder this script sits in and writes
fig_rarity_sweep.png and .pdf (600 DPI) next to it.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "table_rarity_sweep_from_log.csv"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
MODELS = {"xgb": ("#2a78d6", "o", "XGBoost"), "mlp": ("#eb6834", "s", "MLP")}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                     "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
classes = ["drop", "deny", "allow"]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for ax, cls in zip(axes, classes):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
    g = df[df.target_class == cls]
    for m, (hue, mk, lab) in MODELS.items():
        s = g[g.model == m].groupby("train_n_target").target_f1.agg(["mean", "std"])
        ax.errorbar(s.index, s["mean"], yerr=s["std"], color=hue, marker=mk,
                    markersize=4.2, markerfacecolor=hue, markeredgecolor="white",
                    markeredgewidth=0.7, linewidth=1.1, elinewidth=0.8,
                    capsize=1.8, label=lab, zorder=3)
    full = int(g.train_n_target.max())
    ax.set_xscale("log")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(f"Target class: {cls} (full = {full:,})", fontsize=7.8,
                 color=INK, loc="left", pad=5)
    ax.set_xlabel("Training instances of target class", fontsize=7.4, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("$F_1$ of the target class", fontsize=7.4, color=INK)
axes[0].legend(loc="lower right", fontsize=6.6, frameon=False)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_rarity_sweep.{ext}"), dpi=600,
                bbox_inches="tight")
print("written")
