"""
Macro-F1 of the three headline models under three evaluation protocols: the random
stratified folds of the main comparison, per-class contiguous time-block folds, and the
random folds with the native feature set instead of the common schema. Points are the
mean over five folds (seed 42); bars are 95% confidence intervals (mean +/- 1.96 SD/sqrt(5)).

Reads rerun_metrics.csv, time_blocks_metrics.csv and full_features_metrics.csv from the
folder this script sits in; writes fig_protocol_robustness.png and .pdf (600 DPI).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
rr = pd.read_csv(os.path.join(HERE, "rerun_metrics.csv"))
rr = rr[rr.seed == 42]
tb = pd.read_csv(os.path.join(HERE, "time_blocks_metrics.csv"))
ff = pd.read_csv(os.path.join(HERE, "full_features_metrics.csv"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
MODELS = [("xgb_classweight", "XGBoost, class-weighted", "#2a78d6", "o"),
          ("xgb_smote", "XGBoost, SMOTE", "#1baf7a", "^"),
          ("mlp_baseline_smote", "MLP, SMOTE", "#eb6834", "s")]
DS = [("ertam_kaya", "Dataset A"), ("cicids2017", "Dataset B"), ("unsw_nb15", "Dataset C")]
PROTO = [("random", "Random\nfolds", rr),
         ("time", "Time-block\nfolds", tb),
         ("native", "Native\nfeatures", ff)]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                     "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.7), sharey=True)
for ax, (ds, title) in zip(axes, DS):
    ax.set_facecolor("white")
    ax.grid(True, axis="y", color=GRID, linewidth=0.45, zorder=0)
    for j, (var, lab, hue, mk) in enumerate(MODELS):
        xs, ms, es = [], [], []
        for i, (key, _, df) in enumerate(PROTO):
            g = df[(df.dataset == ds) & (df.variant == var)].macro_f1
            assert len(g) == 5, (ds, var, key, len(g))
            xs.append(i + (j - 1) * 0.22)
            ms.append(g.mean())
            es.append(1.96 * g.std(ddof=1) / np.sqrt(len(g)))
        ax.errorbar(xs, ms, yerr=es, fmt=mk, color=hue, ms=4.5, markeredgecolor="white",
                    markeredgewidth=0.6, capsize=2, elinewidth=0.9, lw=0, label=lab, zorder=3)
        ax.plot(xs, ms, color=hue, lw=0.7, alpha=0.5, zorder=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([p[1] for p in PROTO], fontsize=6.6)
    ax.set_xlim(-0.5, 2.5)
    ax.set_title(title, fontsize=7.8, color=INK, loc="left", pad=5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("Macro-$F_1$ (mean, 95% CI)", fontsize=7.4, color=INK)
axes[0].set_ylim(0.45, 1.0)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.8,
           frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_protocol_robustness.{ext}"), dpi=600, bbox_inches="tight")
print("written")
