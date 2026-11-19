"""
Reliability diagrams for the three main models on each dataset, from the pooled
confidence bins written by calibration_from_npz.py. Bins with fewer than 30 records
are not drawn (they remain in the CSV). Marker area grows with the log of the bin
count, so the sparsely populated low-confidence bins are visibly lighter evidence.

Reads calibration_reliability_bins.csv and calibration_summary.csv from the folder
this script sits in; writes fig_reliability.png and .pdf (600 DPI) next to it.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
bins = pd.read_csv(os.path.join(HERE, "calibration_reliability_bins.csv"))
summ = pd.read_csv(os.path.join(HERE, "calibration_summary.csv"))
MIN_N = 30
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
MODELS = [("xgb_classweight", "XGBoost, class-weighted", "#2a78d6", "o"),
          ("xgb_smote", "XGBoost, SMOTE", "#1baf7a", "^"),
          ("mlp_baseline_smote", "MLP, SMOTE", "#eb6834", "s")]
DS = [("ertam_kaya", "Dataset A"), ("cicids2017", "Dataset B"),
      ("unsw_nb15", "Dataset C")]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                     "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.75), sharey=True)
for ax, (ds, title) in zip(axes, DS):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=0.8, linestyle="--", zorder=1)
    for var, lab, hue, mk in MODELS:
        g = bins[(bins.dataset == ds) & (bins.variant == var) & (bins["count"] >= MIN_N)]
        ax.plot(g.mean_conf, g.accuracy, color=hue, linewidth=1.0, zorder=2,
                label=lab)
        ax.scatter(g.mean_conf, g.accuracy, s=3 + 0.9 * np.log10(g["count"]) ** 2,
                   marker=mk, color=hue, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=7.8, color=INK, loc="left", pad=5)
    ax.set_xlabel("Mean predicted confidence", fontsize=7.4, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Observed accuracy", fontsize=7.4, color=INK)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3,
           fontsize=6.8, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_reliability.{ext}"), dpi=600,
                bbox_inches="tight")
print("written")
