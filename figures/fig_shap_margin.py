"""
SHAP attribution margin of class-weighted XGBoost against the composite class overlap
(left) and against class size (right) for the 29 classes. The margin is the mean summed
SHAP evidence for class c on c's own records minus the largest mean evidence for c on the
records of any other class (logit units); a small margin means the features push some
other class towards c almost as strongly as they push c itself. Hollow markers mark
classes whose explained records were partly drawn from the training part (fewer than 20
validation records), for which the margin is partly in-sample.

Reads shap_merged_29.csv from the folder this script sits in; writes fig_shap_margin.png
and .pdf (600 DPI).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "shap_merged_29.csv"))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
DS = {"A": ("#2a78d6", "o", "Dataset A"), "B": ("#eb6834", "s", "Dataset B"),
      "C": ("#1baf7a", "^", "Dataset C")}
LABELS = {"Heartbleed": "Heartbleed", "Sql Injection": "SQL Injection", "Backdoors": "Backdoors",
          "Shellcode": "Shellcode", "Bot": "Bot", "DoS": "DoS", "XSS": "XSS",
          "Brute Force": "Brute Force", "Analysis": "Analysis", "reset-both": "reset-both"}
# label anchor positions in data coordinates (x, y); a thin leader line joins label and marker
POS = {("Overlap", "Heartbleed"): (0.14, 19.3), ("Overlap", "Bot"): (0.13, 16.6),
       ("Overlap", "Shellcode"): (0.15, 11.0), ("Overlap", "Sql Injection"): (0.50, 14.2),
       ("Overlap", "reset-both"): (0.78, 10.6), ("Overlap", "XSS"): (0.70, 3.6),
       ("Overlap", "Brute Force"): (0.46, 2.2), ("Overlap", "Analysis"): (0.74, 5.6),
       ("Overlap", "Backdoors"): (0.60, 0.1), ("Overlap", "DoS"): (0.78, -0.7),
       ("Instances", "Heartbleed"): (25, 19.3), ("Instances", "Bot"): (170, 16.6),
       ("Instances", "Shellcode"): (150, 12.5), ("Instances", "Sql Injection"): (60, 11.1),
       ("Instances", "reset-both"): (130, 8.8), ("Instances", "XSS"): (90, 4.6),
       ("Instances", "Brute Force"): (60, 3.0), ("Instances", "Analysis"): (5000, 4.3),
       ("Instances", "Backdoors"): (150, -0.9), ("Instances", "DoS"): (60000, 2.2)}
HA = {("Overlap", "DoS"): "left", ("Overlap", "Backdoors"): "right", ("Instances", "DoS"): "left",
      ("Instances", "Analysis"): "left"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                     "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
for ax, (xcol, xlabel) in zip(axes, [("Overlap", "Composite class overlap"),
                                     ("Instances", "Class size (instances, log scale)")]):
    ax.set_facecolor("white")
    ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
    for ds, (hue, mk, lab) in DS.items():
        g = df[df.Dataset == ds]
        full = g[g.n_from_training == 0]
        part = g[g.n_from_training > 0]
        ax.scatter(full[xcol], full.attribution_margin, s=28, marker=mk, color=hue,
                   edgecolor="white", linewidth=0.7, zorder=3, label=lab if xcol == "Overlap" else None)
        ax.scatter(part[xcol], part.attribution_margin, s=28, marker=mk, facecolor="white",
                   edgecolor=hue, linewidth=1.1, zorder=3)
    for _, r in df[df.Class.isin(LABELS)].iterrows():
        ax.annotate(LABELS[r.Class], (r[xcol], r.attribution_margin),
                    xytext=POS[(xcol, r.Class)], textcoords="data",
                    ha=HA.get((xcol, r.Class), "left"), va="center",
                    fontsize=6.0, color=INK2, zorder=4,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=2.5))
    if xcol == "Instances":
        ax.set_xscale("log")
    else:
        b, a = np.polyfit(df[xcol], df.attribution_margin, 1)
        xs = np.linspace(df[xcol].min(), df[xcol].max(), 50)
        ax.plot(xs, a + b * xs, color=INK2, lw=0.9, ls="--", zorder=2)
    ax.set_xlabel(xlabel, fontsize=7.4, color=INK)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("SHAP attribution margin (logits)", fontsize=7.4, color=INK)
axes[0].set_ylim(-1.5, 20.5)
fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=6.8,
           frameon=False, bbox_to_anchor=(0.5, 1.03))
fig.tight_layout(rect=(0, 0, 1, 0.95))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_shap_margin.{ext}"), dpi=600, bbox_inches="tight")
print("written")
