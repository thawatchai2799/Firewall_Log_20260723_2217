"""
collect_results.py

Gathers every per-fold result already on disk into tidy tables, and runs the
statistical tests used in the paper. Nothing here re-trains a model: it
reads the existing checkpoint and result files only.

What it produces
----------------
1. per_fold_macro_f1.csv
   One row per (dataset, model, fold) with that fold's macro-F1, covering both
   the MLP techniques (from run/checkpoints/) and the tree baselines (from
   run/results/). Because both were run over the same StratifiedKFold splits
   with random_state=42, these columns are paired fold-for-fold and can be
   compared directly.

2. per_class_f1.csv
   One row per (dataset, model, class) with mean per-class F1, its standard
   deviation across folds, and a bootstrap confidence interval. The standard
   deviation matters for tiny classes: where a class
   has only a handful of test samples per fold, the spread will be large, and
   reporting it makes that visible instead of hiding it behind a mean.

3. cross_family_tests.csv
   Paired comparisons BETWEEN model families (MLP vs XGBoost vs LightGBM), not
   only within them. Each comparison is tested three ways:
     - Wilcoxon signed-rank, which with five folds cannot return a p below
       0.0625 no matter how large the difference is;
     - an exact paired permutation test over the 2^n sign flips;
     - a paired bootstrap over folds, reported as a confidence interval on the
       mean difference.
   The last two do not suffer from the discrete floor, which is what makes the
   model-family comparison testable with five folds.

Usage:
    python collect_results.py --run_dir ../run --output analysis_tables
"""

import argparse
import glob
import itertools
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RNG = np.random.default_rng(42)
N_BOOT = 20000

TECHNIQUES = ["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"]


# --------------------------------------------------------------------------
# Reading what is already on disk
# --------------------------------------------------------------------------

def read_mlp_checkpoints(run_dir):
    """Per-fold MLP results from run/checkpoints/unit__cv__*.json."""
    rows, per_class = [], []
    pattern = os.path.join(run_dir, "checkpoints", "unit__cv__*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no checkpoint files matched {pattern}")
    for path in files:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        ds, tech, fold = d["dataset_key"], d["technique"], int(d["fold_idx"])
        model = f"mlp_{tech}"
        rows.append({"dataset": ds, "model": model, "family": "MLP",
                     "fold": fold, "macro_f1": float(d["macro_f1"])})
        for cname, f1 in zip(d["class_names"], d["per_class_f1"]):
            per_class.append({"dataset": ds, "model": model, "family": "MLP",
                              "fold": fold, "class_name": cname,
                              "f1": float(f1)})
    print(f"  MLP: {len(files)} checkpoint files")
    return rows, per_class


def read_tree_results(run_dir):
    """Per-fold tree results from run/results/*_tree_baseline_results.json."""
    rows, per_class = [], []
    pattern = os.path.join(run_dir, "results", "*_tree_baseline_results.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"  WARNING: no tree result files matched {pattern}")
    for path in files:
        ds = os.path.basename(path).replace("_tree_baseline_results.json", "")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for model, res in d.get("summary", {}).items():
            family = "Tree-XGB" if model.startswith("xgb") else "Tree-LGBM"
            for fold, v in enumerate(res.get("per_fold_macro_f1", [])):
                rows.append({"dataset": ds, "model": model, "family": family,
                             "fold": fold, "macro_f1": float(v)})
            # Tree files store only the across-fold mean per class.
            for cname, f1 in zip(res.get("class_names", []),
                                 res.get("mean_per_class_f1", [])):
                per_class.append({"dataset": ds, "model": model, "family": family,
                                  "fold": -1, "class_name": cname,
                                  "f1": float(f1)})
        print(f"  trees: {os.path.basename(path)}")
    return rows, per_class


def read_tabnet_results(run_dir):
    """Per-fold TabNet results, where they exist."""
    rows = []
    pattern = os.path.join(run_dir, "results", "*_tabnet_baseline_results.json")
    for path in sorted(glob.glob(pattern)):
        ds = os.path.basename(path).replace("_tabnet_baseline_results.json", "")
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        for model, res in d.get("summary", {}).items():
            for fold, v in enumerate(res.get("per_fold_macro_f1", [])):
                rows.append({"dataset": ds, "model": model, "family": "TabNet",
                             "fold": fold, "macro_f1": float(v)})
        print(f"  tabnet: {os.path.basename(path)}")
    return rows


# --------------------------------------------------------------------------
# Tests that do not hit the five-fold p-value floor
# --------------------------------------------------------------------------

def exact_permutation_p(diff):
    """Two-sided exact paired permutation test over all 2^n sign flips.

    With five folds there are only 32 sign assignments, so the test is computed
    exhaustively rather than sampled. The smallest attainable p is 2/32 = 0.0625
    for n=5 as well, so this is reported alongside the bootstrap rather than as a
    replacement for it; it is exact where Wilcoxon is a rank approximation.
    """
    n = len(diff)
    obs = float(np.mean(diff))
    count = 0
    for signs in itertools.product([1, -1], repeat=n):
        if abs(float(np.mean(diff * np.array(signs)))) >= abs(obs) - 1e-12:
            count += 1
    return count / (2 ** n)


def paired_bootstrap(diff, n_boot=N_BOOT):
    """Bootstrap over folds: CI on the mean difference, and a two-sided p.

    The p-value is the proportion of resamples whose mean falls on the opposite
    side of zero from the observed mean, doubled. Unlike the rank tests this is
    continuous in the data, so a large consistent difference can reach a small p
    even with few folds -- at the cost of assuming the folds are exchangeable,
    which is itself an assumption that the paper states explicitly rather than
    relies on silently.
    """
    n = len(diff)
    obs = float(np.mean(diff))
    idx = RNG.integers(0, n, size=(n_boot, n))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    # When no resample crosses zero the estimate is not "p = 0": it is only
    # "p below the resolution of this many resamples". Report that floor instead
    # of a zero that the data cannot support.
    p = max(p, 1.0 / n_boot)
    return obs, float(lo), float(hi), float(min(p, 1.0))


def compare_pair(df, dataset, model_a, model_b):
    a = df[(df.dataset == dataset) & (df.model == model_a)].sort_values("fold")
    b = df[(df.dataset == dataset) & (df.model == model_b)].sort_values("fold")
    folds = sorted(set(a.fold) & set(b.fold))
    folds = [f for f in folds if f >= 0]
    if len(folds) < 3:
        return None
    va = a[a.fold.isin(folds)].macro_f1.to_numpy()
    vb = b[b.fold.isin(folds)].macro_f1.to_numpy()
    diff = va - vb
    if np.allclose(diff, 0):
        return None
    try:
        w_stat, w_p = wilcoxon(va, vb)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    obs, lo, hi, boot_p = paired_bootstrap(diff)
    perm_p = exact_permutation_p(diff) if len(diff) <= 20 else np.nan
    pooled = np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2)
    return {
        "dataset": dataset, "model_a": model_a, "model_b": model_b,
        "n_folds": len(folds),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "mean_difference": obs,
        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
        "bootstrap_p": boot_p,
        "permutation_p": perm_p,
        "wilcoxon_stat": float(w_stat) if w_stat == w_stat else np.nan,
        "wilcoxon_p": float(w_p) if w_p == w_p else np.nan,
        "paired_cohens_d": float(obs / pooled) if pooled > 0 else np.nan,
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def bootstrap_ci_mean(values, n_boot=N_BOOT):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (float(v[0]) if len(v) else np.nan), np.nan, np.nan
    idx = RNG.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run_dir", required=True,
                    help="The 'run' directory holding checkpoints/ and results/")
    ap.add_argument("--output", default="analysis_tables")
    args = ap.parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("[1/4] Reading existing results (no model is retrained)")
    mlp_rows, mlp_cls = read_mlp_checkpoints(args.run_dir)
    tree_rows, tree_cls = read_tree_results(args.run_dir)
    tabnet_rows = read_tabnet_results(args.run_dir)

    folds_df = pd.DataFrame(mlp_rows + tree_rows + tabnet_rows)
    cls_df = pd.DataFrame(mlp_cls + tree_cls)
    if folds_df.empty:
        raise SystemExit("No results found. Check --run_dir.")

    folds_path = os.path.join(args.output, "per_fold_macro_f1.csv")
    folds_df.sort_values(["dataset", "family", "model", "fold"]).to_csv(
        folds_path, index=False)
    print(f"      {len(folds_df)} (dataset, model, fold) rows -> {folds_path}")

    print("[2/4] Per-class F1 with spread and bootstrap CI across folds")
    recs = []
    mlp_only = cls_df[cls_df.fold >= 0]
    for (ds, model, cname), g in mlp_only.groupby(["dataset", "model", "class_name"]):
        mean, lo, hi = bootstrap_ci_mean(g.f1.to_numpy())
        recs.append({"dataset": ds, "model": model, "class_name": cname,
                     "n_folds": len(g), "mean_f1": mean,
                     "sd_f1": float(g.f1.std(ddof=1)) if len(g) > 1 else np.nan,
                     "min_f1": float(g.f1.min()), "max_f1": float(g.f1.max()),
                     "ci_low": lo, "ci_high": hi})
    for (ds, model, cname), g in cls_df[cls_df.fold < 0].groupby(
            ["dataset", "model", "class_name"]):
        recs.append({"dataset": ds, "model": model, "class_name": cname,
                     "n_folds": np.nan, "mean_f1": float(g.f1.mean()),
                     "sd_f1": np.nan, "min_f1": np.nan, "max_f1": np.nan,
                     "ci_low": np.nan, "ci_high": np.nan})
    cls_out = pd.DataFrame(recs)
    cls_path = os.path.join(args.output, "per_class_f1.csv")
    cls_out.sort_values(["dataset", "model", "class_name"]).to_csv(
        cls_path, index=False)
    print(f"      {len(cls_out)} (dataset, model, class) rows -> {cls_path}")

    unstable = cls_out[(cls_out.sd_f1 > 0.25)].sort_values(
        "sd_f1", ascending=False)
    if len(unstable):
        print("\n      Classes whose F1 swings by more than 0.25 SD across folds")
        print("      (these are the ones no firm conclusion should rest on):")
        print(unstable[["dataset", "model", "class_name", "mean_f1",
                        "sd_f1", "min_f1", "max_f1"]].head(15).to_string(index=False))

    print("\n[3/4] Cross-family comparisons on the same folds")
    results = []
    for ds in sorted(folds_df.dataset.unique()):
        sub = folds_df[folds_df.dataset == ds]
        has = set(sub.model.unique())
        best_mlp = "mlp_baseline_smote"
        candidates = []
        for tree in ["xgb_classweight", "xgb_smote", "xgb_plain"]:
            if tree in has and best_mlp in has:
                candidates.append((best_mlp, tree))
        for lg in ["lgbm_plain", "lgbm_smote", "lgbm_classweight"]:
            if lg in has and "xgb_plain" in has:
                candidates.append(("xgb_plain", lg))
        for tb in ["tabnet_plain", "tabnet_smote", "tabnet_classweight"]:
            if tb in has and best_mlp in has:
                candidates.append((best_mlp, tb))
        for a, b in candidates:
            r = compare_pair(sub, ds, a, b)
            if r:
                results.append(r)

    tests = pd.DataFrame(results)
    if len(tests):
        tests_path = os.path.join(args.output, "cross_family_tests.csv")
        tests.to_csv(tests_path, index=False)
        print(f"      {len(tests)} paired comparisons -> {tests_path}")
        show = tests[["dataset", "model_a", "model_b", "mean_a", "mean_b",
                      "mean_difference", "wilcoxon_p", "permutation_p",
                      "bootstrap_p", "ci_excludes_zero"]]
        print(show.to_string(index=False))
    else:
        print("      no comparable pairs found")

    print("\n[4/4] Summary table of macro-F1 by dataset and model")
    summ = folds_df.groupby(["dataset", "family", "model"]).macro_f1.agg(
        ["count", "mean", "std"]).reset_index()
    summ_path = os.path.join(args.output, "macro_f1_summary.csv")
    summ.to_csv(summ_path, index=False)
    print(summ.to_string(index=False))
    print(f"\nSaved to {summ_path}")


if __name__ == "__main__":
    main()
