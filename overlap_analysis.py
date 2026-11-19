"""
overlap_analysis.py

Measures class-level feature overlap directly, so that the claim "classification
difficulty is governed by feature overlap rather than class rarity" can be tested
statistically instead of asserted descriptively.

For every class in every dataset this script computes five established
data-complexity measures, each of which quantifies, from a different angle, how
much a class overlaps the others in feature space:

  N1  Fraction of the class's points that lie on a boundary with another class,
      measured as the fraction that are endpoints of a minimum-spanning-tree edge
      joining two different classes (Ho & Basu, 2002). Higher = more overlap.
  N2  Ratio of the mean distance to the nearest same-class neighbour to the mean
      distance to the nearest different-class neighbour, expressed as
      intra / (intra + inter) (Ho & Basu, 2002). Higher = more overlap.
  N3  Leave-one-out error rate of a 1-nearest-neighbour classifier, restricted to
      the class's own points (Ho & Basu, 2002). Higher = more overlap.
  R   Fraction of the class's points whose k nearest neighbours are mostly points
      of other classes (Oh, 2011). Higher = more overlap.
  F1  Fisher discriminant ratio of this class against all others, taken as the
      maximum over single features (Ho & Basu, 2002). REPORTED INVERTED as
      1/(1+F1) so that, like the others, higher = more overlap.

Because these measures are all nearest-neighbour or spanning-tree based, they are
quadratic in the number of points and cannot be computed on millions of rows
directly. Each measure is therefore computed on a stratified subsample and
repeated over several independent draws; the script reports the mean and standard
deviation across draws, so every overlap value carries its own uncertainty.

Output: one CSV row per (dataset, class) with the overlap measures, the class
frequency, and the class size, plus a JSON file with the full per-draw values.
These are then joined with the per-class F1 scores from the main experiments to
test whether overlap predicts difficulty after class frequency is controlled for.

Usage:
    python overlap_analysis.py --dataset ertam_kaya
    python overlap_analysis.py --dataset cicids2017 --class_cap 1500 --n_draws 5
    python overlap_analysis.py --all
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances

from data_loader import load_dataset

try:
    from dataset_config import DATASET_PATHS
except ImportError:
    DATASET_PATHS = {}

GLOBAL_SEED = 42

# Defaults chosen so that one draw fits comfortably in memory on a 16 GB machine:
# the pairwise distance matrix for the MST is (n x n) float64, so n is kept in the
# low thousands.
DEFAULT_CLASS_CAP = 2000     # max points sampled per class, per draw
DEFAULT_TOTAL_CAP = 12000    # max points in one draw across all classes
DEFAULT_N_DRAWS = 5          # independent subsamples, for mean +/- SD
DEFAULT_K = 5                # neighbourhood size for the R measure


# --------------------------------------------------------------------------
# Subsampling
# --------------------------------------------------------------------------

def stratified_draw(X, y, class_cap, total_cap, rng):
    """Draw a stratified subsample, keeping small classes whole.

    Classes are filled smallest-first, so rare classes (the ones the study is
    about) are never reduced; only the largest classes are capped. The result is
    a subsample in which every class is represented as fully as the budget allows.
    """
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)  # smallest first
    budget = total_cap
    remaining = len(classes)
    keep = []
    for pos in order:
        c = classes[pos]
        idx_c = np.where(y == c)[0]
        quota = min(class_cap, max(1, budget // remaining))
        if len(idx_c) <= quota:
            keep.append(idx_c)
            budget -= len(idx_c)
        else:
            keep.append(rng.choice(idx_c, size=quota, replace=False))
            budget -= quota
        remaining -= 1
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


# --------------------------------------------------------------------------
# Complexity measures, computed per class
# --------------------------------------------------------------------------

def measure_N1(X, y, classes):
    """Fraction of each class's points that touch a cross-class MST edge."""
    D = pairwise_distances(X, metric="euclidean")
    mst = minimum_spanning_tree(D).tocoo()
    touching = {c: set() for c in classes}
    for i, j in zip(mst.row, mst.col):
        if y[i] != y[j]:
            touching[y[i]].add(i)
            touching[y[j]].add(j)
    out = {}
    for c in classes:
        n_c = int((y == c).sum())
        out[c] = len(touching[c]) / n_c if n_c else np.nan
    return out


def measure_N2(X, y, classes):
    """intra / (intra + inter) nearest-neighbour distance, per class."""
    nn = NearestNeighbors(n_neighbors=min(len(X), 50)).fit(X)
    dist, idx = nn.kneighbors(X)
    out = {}
    for c in classes:
        rows = np.where(y == c)[0]
        intra, inter = [], []
        for r in rows:
            same = None
            other = None
            for d, j in zip(dist[r][1:], idx[r][1:]):   # skip self
                if same is None and y[j] == c:
                    same = d
                if other is None and y[j] != c:
                    other = d
                if same is not None and other is not None:
                    break
            if same is not None and other is not None:
                intra.append(same)
                inter.append(other)
        if intra:
            a, b = float(np.mean(intra)), float(np.mean(inter))
            out[c] = a / (a + b) if (a + b) > 0 else np.nan
        else:
            out[c] = np.nan
    return out


def measure_N3(X, y, classes):
    """Leave-one-out 1NN error rate, per class."""
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    _, idx = nn.kneighbors(X)
    nearest_label = y[idx[:, 1]]          # column 0 is the point itself
    out = {}
    for c in classes:
        rows = np.where(y == c)[0]
        out[c] = float((nearest_label[rows] != c).mean()) if len(rows) else np.nan
    return out


def measure_R(X, y, classes, k):
    """Fraction of points whose k neighbours are mostly other classes."""
    kk = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=kk).fit(X)
    _, idx = nn.kneighbors(X)
    neigh_labels = y[idx[:, 1:]]          # drop self
    out = {}
    for c in classes:
        rows = np.where(y == c)[0]
        if not len(rows):
            out[c] = np.nan
            continue
        frac_other = (neigh_labels[rows] != c).mean(axis=1)
        out[c] = float((frac_other > 0.5).mean())
    return out


def measure_F1_inverted(X, y, classes):
    """1/(1+Fisher discriminant ratio) for class vs rest, max over features."""
    out = {}
    for c in classes:
        a = X[y == c]
        b = X[y != c]
        if len(a) < 2 or len(b) < 2:
            out[c] = np.nan
            continue
        num = (a.mean(axis=0) - b.mean(axis=0)) ** 2
        den = a.var(axis=0) + b.var(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(den > 0, num / den, 0.0)
        f1 = float(np.nanmax(ratio))
        out[c] = 1.0 / (1.0 + f1)
    return out


MEASURES = {
    "N1_boundary_fraction": measure_N1,
    "N2_intra_inter_ratio": measure_N2,
    "N3_1nn_loo_error": measure_N3,
    "R_knn_majority_other": measure_R,
    "F1inv_fisher": measure_F1_inverted,
}


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def analyse_dataset(dataset_key, path, feature_names_csv, class_cap, total_cap,
                    n_draws, k, output_dir):
    print(f"\n=== {dataset_key} ===")
    print(f"[1/3] Loading from {path}")
    if feature_names_csv:
        X_df, y_raw, _ = load_dataset(dataset_key, path,
                                      feature_names_csv=feature_names_csv)
    else:
        X_df, y_raw, _ = load_dataset(dataset_key, path)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    encoder = LabelEncoder()
    y_all = encoder.fit_transform(y_raw)
    class_names = encoder.classes_.tolist()
    n_total = len(y_all)

    # Class frequency is computed on the FULL dataset, not the subsample, so that
    # rarity is the true rarity and only the overlap measures are subsampled.
    counts_full = {c: int((y_all == c).sum()) for c in range(len(class_names))}
    print(f"      {n_total:,} rows, {len(class_names)} classes")

    X_all = StandardScaler().fit_transform(X_df.values).astype(np.float32)

    print(f"[2/3] Computing overlap measures over {n_draws} independent draws "
          f"(class cap {class_cap}, total cap {total_cap})")
    per_draw = {name: {c: [] for c in range(len(class_names))} for name in MEASURES}
    for d in range(n_draws):
        t0 = time.time()
        rng = np.random.default_rng(GLOBAL_SEED + d)
        Xs, ys = stratified_draw(X_all, y_all, class_cap, total_cap, rng)
        classes_present = np.unique(ys)
        for name, fn in MEASURES.items():
            vals = fn(Xs, ys, classes_present, k) if name == "R_knn_majority_other" \
                else fn(Xs, ys, classes_present)
            for c, v in vals.items():
                if not (v is None or (isinstance(v, float) and np.isnan(v))):
                    per_draw[name][int(c)].append(float(v))
        print(f"      draw {d+1}/{n_draws}: {len(ys):,} points, "
              f"{time.time()-t0:.1f}s")

    print("[3/3] Writing results")
    rows = []
    for c, cname in enumerate(class_names):
        row = {
            "dataset": dataset_key,
            "class_index": c,
            "class_name": cname,
            "class_count": counts_full[c],
            "class_proportion": counts_full[c] / n_total,
            "log10_proportion": float(np.log10(max(counts_full[c] / n_total, 1e-12))),
        }
        for name in MEASURES:
            vals = per_draw[name][c]
            row[name] = float(np.mean(vals)) if vals else np.nan
            row[name + "_sd"] = float(np.std(vals)) if len(vals) > 1 else np.nan
            row[name + "_n_draws"] = len(vals)
        rows.append(row)

    df = pd.DataFrame(rows)
    # A single composite overlap score: the mean of the five measures after each is
    # scaled to [0, 1] across the classes of this dataset. Every component is
    # oriented so that higher means more overlap.
    comp = df[list(MEASURES)].copy()
    for col in comp.columns:
        lo, hi = comp[col].min(), comp[col].max()
        comp[col] = 0.5 if hi == lo else (comp[col] - lo) / (hi - lo)
    df["overlap_composite"] = comp.mean(axis=1)

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{dataset_key}_overlap_measures.csv")
    df.to_csv(csv_path, index=False)

    json_path = os.path.join(output_dir, f"{dataset_key}_overlap_per_draw.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_key": dataset_key,
            "n_rows_full": n_total,
            "class_names": class_names,
            "protocol": {
                "class_cap": class_cap, "total_cap": total_cap,
                "n_draws": n_draws, "k_for_R": k, "global_seed": GLOBAL_SEED,
                "note": ("Class frequencies come from the full dataset; only the "
                         "overlap measures are computed on stratified subsamples, "
                         "repeated over independent draws."),
            },
            "per_draw_values": {m: {str(c): v for c, v in d.items()}
                                for m, d in per_draw.items()},
        }, f, indent=2)

    print(f"      saved {csv_path}")
    print(f"      saved {json_path}")
    print("\n" + df[["class_name", "class_count", "class_proportion",
                     "N3_1nn_loo_error", "R_knn_majority_other",
                     "overlap_composite"]].to_string(index=False))
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Run every dataset in dataset_config.DATASET_PATHS")
    ap.add_argument("--path", default=None)
    ap.add_argument("--feature_names_csv", default=None)
    ap.add_argument("--class_cap", type=int, default=DEFAULT_CLASS_CAP)
    ap.add_argument("--total_cap", type=int, default=DEFAULT_TOTAL_CAP)
    ap.add_argument("--n_draws", type=int, default=DEFAULT_N_DRAWS)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--output", default="overlap_results")
    args = ap.parse_args()

    targets = list(DATASET_PATHS) if args.all else [args.dataset]
    if not targets or targets == [None]:
        raise SystemExit("Give --dataset NAME or --all")

    frames = []
    for key in targets:
        cfg = DATASET_PATHS.get(key, {})
        path = args.path or cfg.get("path")
        fnames = args.feature_names_csv or cfg.get("feature_names_csv")
        if not path:
            raise SystemExit(f"No path for {key}; set it in dataset_config.py")
        frames.append(analyse_dataset(key, path, fnames, args.class_cap,
                                      args.total_cap, args.n_draws, args.k,
                                      args.output))

    if len(frames) > 1:
        combined = pd.concat(frames, ignore_index=True)
        out = os.path.join(args.output, "all_datasets_overlap_measures.csv")
        combined.to_csv(out, index=False)
        print(f"\nCombined table for all datasets: {out}")
        print(f"{len(combined)} classes in total, ready to be joined with "
              f"per-class F1 for the regression.")


if __name__ == "__main__":
    main()
