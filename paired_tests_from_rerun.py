"""
paired_tests_from_rerun.py

Paired comparison of the best XGBoost technique against the SMOTE MLP on the same folds,
computed from the summary tables of the seeded re-run and of the protocol checks.

For every dataset and every protocol (random folds, time-block folds, native features) it
reports the mean paired difference in macro-F1 with a paired-bootstrap 95% confidence
interval (20,000 resamples), the bootstrap p-value, the exact sign-flip permutation p-value
(2^-(n-1) is its floor; 0.0625 for five folds) and whether XGBoost wins every fold.
Seeds are averaged within each fold first, so that n equals the number of folds.

Usage (from the repository root):
    python paired_tests_from_rerun.py
    python paired_tests_from_rerun.py --results results --output paired_tests.csv

Reads:
    results/main_comparison/rerun_metrics.csv
    results/protocol_checks/time_blocks/metrics.csv
    results/protocol_checks/full_features/metrics.csv
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

MLP = ("mlp", "baseline_smote")
XGB = [("xgb", "classweight"), ("xgb", "smote")]
DATASETS = ["ertam_kaya", "cicids2017", "unsw_nb15"]


def fold_means(df, family, technique):
    sel = df[(df.family == family) & (df.technique_or_mode == technique)]
    return sel.groupby("fold").macro_f1.mean().sort_index()


def paired_stats(diff, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(diff)
    boots = np.array([rng.choice(diff, n, replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    boot_p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    # exact sign-flip permutation test on the mean difference
    obs = abs(diff.mean())
    count = 0
    total = 0
    for signs in itertools.product([1, -1], repeat=n):
        total += 1
        if abs((diff * np.array(signs)).mean()) >= obs - 1e-12:
            count += 1
    return diff.mean(), lo, hi, boot_p, count / total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results")
    ap.add_argument("--output", default=None, help="optional CSV to write")
    args = ap.parse_args()

    sources = {
        "random folds": os.path.join(args.results, "main_comparison", "rerun_metrics.csv"),
        "time-block folds": os.path.join(args.results, "protocol_checks", "time_blocks", "metrics.csv"),
        "native features": os.path.join(args.results, "protocol_checks", "full_features", "metrics.csv"),
    }
    rows = []
    for protocol, path in sources.items():
        if not os.path.exists(path):
            print(f"[skip] {path} not found")
            continue
        df = pd.read_csv(path)
        for ds in DATASETS:
            d = df[df.dataset == ds]
            if d.empty:
                continue
            mlp = fold_means(d, *MLP)
            best = max(XGB, key=lambda ft: fold_means(d, *ft).mean())
            xgb = fold_means(d, *best)
            common = xgb.index.intersection(mlp.index)
            diff = (xgb.loc[common] - mlp.loc[common]).values
            mean, lo, hi, boot_p, perm_p = paired_stats(diff)
            rows.append({
                "protocol": protocol, "dataset": ds, "best_xgb_technique": best[1],
                "n_folds": len(diff), "xgb_mean": xgb.loc[common].mean(), "mlp_mean": mlp.loc[common].mean(),
                "paired_difference": mean, "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                "bootstrap_p": boot_p, "permutation_p": perm_p, "xgb_wins_every_fold": bool((diff > 0).all()),
            })
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.4f}".format):
        print(out.to_string(index=False))
    if args.output:
        out.to_csv(args.output, index=False)
        print(f"written {args.output}")


if __name__ == "__main__":
    main()
