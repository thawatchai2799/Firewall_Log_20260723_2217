"""
Per-class F1 against class size (left) and measured class overlap (right) for the
29 classes of the three datasets, for class-weighted XGBoost (top) and the SMOTE MLP
(bottom). F1 values come from the seeded re-run (mean over folds); overlap is the
composite of five data-complexity measures (min-max scaled, then averaged).

Reads table_overlap_measures_v2.csv from the folder this script sits in and writes
fig_difficulty_vs_rarity_overlap.png and .pdf (600 DPI) next to it.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(HERE, "table_overlap_measures_v2.csv"))

INK, INK2, GRID = "#0b0b0b", "#52514e", "#d8d7d2"
DS = {"A": ("#2a78d6", "o", "Dataset A (firewall)"),
      "B": ("#eb6834", "s", "Dataset B (CIC-IDS2017)"),
      "C": ("#1baf7a", "^", "Dataset C (UNSW-NB15)")}
MODELS = [("xgb_classweight", "XGBoost, class-weighted"),
          ("mlp_baseline_smote", "MLP, SMOTE")]
LABELS = {"Heartbleed": "Heartbleed", "Sql Injection": "SQL injection",
          "Backdoors": "Backdoors", "Shellcode": "Shellcode",
          "reset-both": "reset-both"}

# label anchors in data coordinates, keyed by (row, col, class); a thin leader joins label and marker
POS = {
    (0, 0, "Heartbleed"): (25, 0.90), (0, 0, "Sql Injection"): (45, 0.44), (0, 0, "reset-both"): (75, 0.56),
    (0, 0, "Shellcode"): (400, 0.80), (0, 0, "Backdoors"): (6000, 0.14),
    (0, 1, "Heartbleed"): (0.12, 0.73), (0, 1, "Shellcode"): (0.27, 0.80), (0, 1, "reset-both"): (0.70, 0.76),
    (0, 1, "Sql Injection"): (0.66, 0.44), (0, 1, "Backdoors"): (0.72, 0.16),
    (1, 0, "Heartbleed"): (25, 0.93), (1, 0, "Sql Injection"): (12, 0.28), (1, 0, "reset-both"): (130, 0.34),
    (1, 0, "Shellcode"): (400, 0.80), (1, 0, "Backdoors"): (6000, 0.22),
    (1, 1, "Heartbleed"): (0.12, 0.78), (1, 1, "Shellcode"): (0.23, 0.62), (1, 1, "reset-both"): (0.80, 0.30),
    (1, 1, "Sql Injection"): (0.66, 0.24), (1, 1, "Backdoors"): (0.62, 0.01),
}
LABELS["Sql Injection"] = "SQL Injection"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                     "axes.linewidth": 0.6, "axes.edgecolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.4), sharey=True)
for r, (col, mname) in enumerate(MODELS):
    for c, (xcol, xlabel) in enumerate([("Instances", "Class size (instances, log scale)"),
                                        ("Overlap", "Composite class overlap")]):
        ax = axes[r][c]
        ax.set_facecolor("white")
        ax.grid(True, color=GRID, linewidth=0.45, zorder=0)
        for ds, (hue, mk, lab) in DS.items():
            g = df[df.Dataset == ds]
            ax.scatter(g[xcol], g[col], s=26, marker=mk, color=hue,
                       edgecolor="white", linewidth=0.7, zorder=3,
                       label=lab if (r == 0 and c == 0) else None)
        for _, row in df[df.Class.isin(LABELS)].iterrows():
            ax.annotate(LABELS[row.Class], (row[xcol], row[col]),
                        xytext=POS[(r, c, row.Class)], textcoords="data",
                        ha="left", va="center", fontsize=6.2, color=INK2, zorder=4,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4, shrinkA=0, shrinkB=2.5))
        if xcol == "Instances":
            ax.set_xscale("log")
        else:
            x = df[xcol].values
            b, a = np.polyfit(x, df[col].values, 1)
            xs = np.linspace(x.min(), x.max(), 50)
            ax.plot(xs, a + b * xs, color=INK2, linewidth=0.9, linestyle="--",
                    zorder=2)
        ax.set_ylim(-0.04, 1.06)
        if r == 1:
            ax.set_xlabel(xlabel, fontsize=7.4, color=INK)
        if c == 0:
            ax.set_ylabel(f"{mname}\nper-class $F_1$", fontsize=7.4, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center",
           ncol=3, fontsize=6.8, frameon=False, handletextpad=0.3,
           bbox_to_anchor=(0.5, 1.02))
fig.tight_layout(rect=(0, 0, 1, 0.97))
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_difficulty_vs_rarity_overlap.{ext}"),
                dpi=600, bbox_inches="tight")
print("written")
