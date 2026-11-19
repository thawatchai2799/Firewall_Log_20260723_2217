"""
controlled_experiment.py

The observational analysis can only show that overlap correlates with difficulty
better than rarity does. A correlation across
classes cannot by itself establish which of the two actually causes the difficulty, because
in real data the two are entangled: a class that is rare is often also the one
sitting in a crowded region of feature space.

This script separates them by manipulating each one directly while holding the
other fixed, and measuring what happens to the class's F1.

Experiment 1 - vary rarity, hold overlap fixed
    A target class is randomly subsampled in the TRAINING set only, to
    n = 25, 50, 100, 250, 500, 1000, ... up to its full size. Random subsampling
    leaves the class's feature distribution unchanged, so its overlap with other
    classes is untouched while its frequency falls by orders of magnitude. The
    TEST set is never modified, so F1 stays comparable across conditions.

    If rarity were the cause of difficulty, every class would recover as n grows.
    If overlap were the cause, classes sitting in clean regions should recover
    quickly while classes buried inside other classes should stay flat no matter
    how many training examples they are given.

Experiment 2 - vary overlap, hold rarity fixed
    A target class's points are moved a fraction alpha of the way towards the
    centroid of the nearest other class, for alpha = 0, 0.2, 0.4, 0.6, 0.8. The
    number of examples never changes; only how deeply the class sits inside its
    neighbour does. The same transformation is applied to train and test, so this
    genuinely makes the class more overlapped rather than creating a train/test
    mismatch.

    Crucially, the overlap measures are RE-MEASURED at every alpha, so the script
    verifies that the manipulation did what it claims rather than assuming it.

Both experiments are run with two model families (the MLP used throughout the
study, and XGBoost) so that any conclusion does not rest on one inductive bias.

Usage:
    python controlled_experiment.py --dataset ertam_kaya
    python controlled_experiment.py --dataset ertam_kaya --classes reset-both deny
    python controlled_experiment.py --dataset unsw_nb15 --max_rows 400000 --seeds 3
"""

import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.neighbors import NearestNeighbors

from data_loader import load_dataset

try:
    from dataset_config import (DATASET_PATHS, EPOCHS, HIDDEN_SIZE,
                                BATCH_SIZE, LEARNING_RATE)
except ImportError:
    DATASET_PATHS, EPOCHS, HIDDEN_SIZE = {}, 100, 200
    BATCH_SIZE, LEARNING_RATE = 512, 1e-3

GLOBAL_SEED = 42

# torch is imported only when the MLP is actually requested, so that
# "--models xgb" runs on a machine without PyTorch installed.
_TORCH = None


def _torch():
    global _TORCH
    if _TORCH is None:
        import torch
        _TORCH = torch
    return _TORCH


def get_device():
    return "cuda" if _torch().cuda.is_available() else "cpu"

RARITY_LEVELS = [25, 50, 100, 250, 500, 1000, 2500, 5000]
# alpha = 1.0 is deliberately excluded. There the target points become exact
# copies of rows drawn from the partner pool, so many of them also duplicate
# each other: a point's nearest neighbour is then another target point at
# distance zero, the measured N3 falls back instead of saturating, and the level
# is not comparable with the rest of the sweep. 0.9 already drives the measured
# overlap close to its ceiling. The remaining levels match the grid on which
# the usable 'drop' curve was measured, so classes run separately stay directly
# comparable with it.
ALPHA_LEVELS = [0.0, 0.3, 0.5, 0.65, 0.8, 0.9]


def set_seeds(seed, with_torch=False):
    random.seed(seed)
    np.random.seed(seed)
    if with_torch:
        t = _torch()
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def train_mlp(X_train, y_train, n_classes, seed):
    """Train the study's MLP with NO resampling and NO class weighting.

    This is deliberate and is the point on which the rarity experiment depends.
    SMOTE would synthesise the target class back up to the size of the majority,
    which would undo the very manipulation being applied: a class cut down to 25
    training rows would be re-inflated to hundreds of thousands, and the
    experiment would measure nothing. Training on the imbalanced data as it
    stands is therefore the only way to let class frequency vary.

    A consequence worth stating in the write-up: F1 values here are not directly
    comparable with the main results, which do use resampling. The experiment is
    internally comparative -- what matters is how F1 moves across conditions, not
    its absolute level.
    """
    torch = _torch()
    import torch.nn as nn
    import torch.utils.data  # not always pulled in by `import torch` alone
    from model import FirewallMLP
    set_seeds(seed, with_torch=True)
    DEVICE = get_device()
    model = FirewallMLP(n_features=X_train.shape[1], n_classes=n_classes,
                        hidden_size=HIDDEN_SIZE).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    Xt = torch.as_tensor(X_train, dtype=torch.float32)
    yt = torch.as_tensor(y_train, dtype=torch.long)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_mlp(model, X):
    torch = _torch()
    DEVICE = get_device()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 8192):
            xb = torch.as_tensor(X[i:i + 8192], dtype=torch.float32).to(DEVICE)
            out.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(out)


def train_predict_xgb(X_train, y_train, X_test, n_classes, seed):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, tree_method="hist",
        objective="multi:softmax", num_class=n_classes,
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# --------------------------------------------------------------------------
# Overlap re-measurement, so the manipulation can be verified
# --------------------------------------------------------------------------

def quick_overlap(X, y, target_idx, cap=2000, seed=GLOBAL_SEED):
    """1NN leave-one-out error and k-NN majority-other rate for one class.

    These are the two cheapest of the five measures used in the observational
    analysis, and both move in the same direction as the rest, so they serve as a
    verification that an overlap manipulation actually changed overlap.
    """
    rng = np.random.default_rng(seed)
    idx_t = np.where(y == target_idx)[0]
    idx_o = np.where(y != target_idx)[0]
    if len(idx_t) < 2:
        return np.nan, np.nan
    if len(idx_t) > cap:
        idx_t = rng.choice(idx_t, cap, replace=False)
    if len(idx_o) > cap * 4:
        idx_o = rng.choice(idx_o, cap * 4, replace=False)
    idx = np.concatenate([idx_t, idx_o])
    Xs, ys = X[idx], (y[idx] == target_idx).astype(int)
    k = min(6, len(Xs))
    nn = NearestNeighbors(n_neighbors=k).fit(Xs)
    _, nbr = nn.kneighbors(Xs)
    rows = np.where(ys == 1)[0]
    n3 = float((ys[nbr[rows, 1]] != 1).mean())
    frac_other = (ys[nbr[rows, 1:]] != 1).mean(axis=1)
    r = float((frac_other > 0.5).mean())
    return n3, r


def pick_nearest_other_class(X_tr, y_tr, target_idx, min_partner_rows=500):
    """Return the index of the class whose centroid is closest to the target's.

    Decided on the training split alone and then held fixed for the whole sweep,
    so every level of the manipulation moves the target class towards the same
    competitor.

    Candidates are restricted to classes with at least `min_partner_rows`
    training rows. Partner points are drawn with replacement, so a pool of a few
    dozen rows would place the whole target class onto a few dozen locations:
    the class would break into tight internally-pure knots and the MEASURED
    overlap would fall as alpha rose, which is the opposite of the intended
    manipulation. If no class qualifies the nearest class is used anyway and the
    caller is warned.
    """
    mask = y_tr == target_idx
    if mask.sum() == 0:
        return None, False
    c_t = X_tr[mask].mean(axis=0)
    best, best_d, ok = None, np.inf, True
    for c in np.unique(y_tr):
        if c == target_idx or (y_tr == c).sum() < min_partner_rows:
            continue
        d = float(np.linalg.norm(X_tr[y_tr == c].mean(axis=0) - c_t))
        if d < best_d:
            best_d, best = d, int(c)
    if best is None:                      # nothing big enough; fall back
        ok = False
        for c in np.unique(y_tr):
            if c == target_idx:
                continue
            d = float(np.linalg.norm(X_tr[y_tr == c].mean(axis=0) - c_t))
            if d < best_d:
                best_d, best = d, int(c)
    return best, ok


def blend_class_towards(X, y, target_idx, neighbour_pool, alpha, seed=0):
    """Pull every point of the target class a fraction alpha towards a randomly
    matched point of the neighbouring class.

    Two properties matter here, and the first version of this function had
    neither.

    First, the manipulation has to raise the measured overlap. Translating the
    whole class rigidly so that its centroid slides onto the neighbour's does
    not: a dense class stays a dense blob, every one of its points still has
    another target point as its nearest neighbour, and N3 barely moves even
    once the two centroids coincide. Blending each point individually towards a
    different partner disperses the class into the neighbour's region, so the
    two become progressively inseparable and N3 rises with alpha.

    Second, `neighbour_pool` must come from the TRAINING split and the same pool
    must be used for the training and the test split. Recomputing the
    transformation separately on each split displaces them by different vectors,
    and the F1 that results measures a train/test mismatch rather than class
    overlap.

    A third refinement was tried and REJECTED, and is recorded here so it is not
    attempted again. A convex combination of two independent draws contracts the
    variance -- Var[(1-a)x + a*p] = (1-a)^2 Var(x) + a^2 Var(p) -- shrinking the
    class by up to a quarter, and a more compact class is an easier class. The
    obvious correction is to rescale each feature back to the class's original
    standard deviation. It does not work. These features are standardised
    globally, so a class is typically near-constant in several of them (a port
    that hardly varies within the class) while its competitor is not. In those
    features the blend raises the standard deviation by two orders of magnitude,
    the correction factor comes out around 0.005, and rescaling crushes the
    blended points back onto a near-degenerate hyperplane -- undoing the blend
    exactly where it mattered. Measured overlap then COLLAPSES as alpha rises
    (N3 0.765 -> 0.003 in a direct comparison) instead of rising, and the class
    becomes so thin that ordinary sampling differences between the splits exceed
    its own spread, so F1 falls to zero for reasons that have nothing to do with
    overlap.

    The variance contraction is therefore left in place and reported as a
    measured confound. It biases the class towards being EASIER, so it works
    against an observed decline in F1: a decline measured under it is a lower
    bound on the effect of overlap. It does dominate for a class of a few dozen
    rows, where it can make F1 rise with alpha; such classes are excluded from
    the dose-response analysis rather than corrected.
    """
    if alpha <= 0 or neighbour_pool is None or len(neighbour_pool) == 0:
        return X
    mask = y == target_idx
    if mask.sum() == 0:
        return X
    rng = np.random.default_rng(seed)
    partner = neighbour_pool[rng.integers(0, len(neighbour_pool), int(mask.sum()))]
    Xn = X.copy()
    Xn[mask] = ((1.0 - alpha) * X[mask] + alpha * partner).astype(X.dtype)
    return Xn


# --------------------------------------------------------------------------
# The two experiments
# --------------------------------------------------------------------------

def run_rarity_sweep(X_tr, y_tr, X_te, y_te, target_idx, target_name,
                     n_classes, seeds, models):
    rows = []
    full_n = int((y_tr == target_idx).sum())
    levels = [n for n in RARITY_LEVELS if n < full_n] + [full_n]
    for n_keep in levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx_t = np.where(y_tr == target_idx)[0]
            keep = rng.choice(idx_t, size=min(n_keep, len(idx_t)), replace=False)
            sel = np.concatenate([np.where(y_tr != target_idx)[0], keep])
            rng.shuffle(sel)
            Xs, ys = X_tr[sel], y_tr[sel]
            n3, rr = quick_overlap(Xs, ys, target_idx, seed=seed)
            for mname in models:
                t0 = time.time()
                if mname == "mlp":
                    pred = predict_mlp(train_mlp(Xs, ys, n_classes, seed), X_te)
                else:
                    pred = train_predict_xgb(Xs, ys, X_te, n_classes, seed)
                _, _, f1s, _ = precision_recall_fscore_support(
                    y_te, pred, labels=range(n_classes), zero_division=0)
                rows.append({
                    "experiment": "rarity", "target_class": target_name,
                    "model": mname, "seed": seed,
                    "train_n_target": int(min(n_keep, len(idx_t))),
                    "train_prop_target": float(min(n_keep, len(idx_t)) / len(ys)),
                    "alpha": 0.0,
                    "target_f1": float(f1s[target_idx]),
                    "macro_f1": float(f1_score(y_te, pred, average="macro",
                                               zero_division=0)),
                    "measured_N3": n3, "measured_R": rr,
                    "seconds": round(time.time() - t0, 1),
                })
                print(f"      rarity n={n_keep:<6d} seed={seed} {mname:4s} "
                      f"F1({target_name})={f1s[target_idx]:.4f} "
                      f"N3={n3:.3f} ({rows[-1]['seconds']}s)")
    return rows


def run_overlap_sweep(X_tr, y_tr, X_te, y_te, target_idx, target_name,
                      n_classes, seeds, models, class_names,
                      min_partner_rows=500):
    rows = []
    nb_idx, pool_ok = pick_nearest_other_class(X_tr, y_tr, target_idx,
                                               min_partner_rows)
    pool = X_tr[y_tr == nb_idx] if nb_idx is not None else None
    nb_name = class_names[nb_idx] if nb_idx is not None else "none"
    print(f"      nearest competing class: '{nb_name}' "
          f"({0 if pool is None else len(pool):,} training rows)"
          + ("" if pool_ok else
             f"  ** WARNING: below --min_partner_rows {min_partner_rows}; "
             "the blend will collapse onto a few locations and the measured "
             "overlap may fall instead of rise **"))
    for alpha in ALPHA_LEVELS:
        # The pool is the same for both splits, and the partner draw is seeded
        # per split, so train and test are mapped into the same region.
        Xtr_a = blend_class_towards(X_tr, y_tr, target_idx, pool, alpha,
                                    seed=1000)
        Xte_a = blend_class_towards(X_te, y_te, target_idx, pool, alpha,
                                    seed=2000)
        n3, rr = quick_overlap(Xtr_a, y_tr, target_idx)
        for seed in seeds:
            for mname in models:
                t0 = time.time()
                if mname == "mlp":
                    pred = predict_mlp(train_mlp(Xtr_a, y_tr, n_classes, seed),
                                       Xte_a)
                else:
                    pred = train_predict_xgb(Xtr_a, y_tr, Xte_a, n_classes, seed)
                _, _, f1s, _ = precision_recall_fscore_support(
                    y_te, pred, labels=range(n_classes), zero_division=0)
                rows.append({
                    "experiment": "overlap", "target_class": target_name,
                    "model": mname, "seed": seed,
                    "train_n_target": int((y_tr == target_idx).sum()),
                    "train_prop_target": float((y_tr == target_idx).mean()),
                    "alpha": alpha, "blend_partner_class": nb_name,
                    "target_f1": float(f1s[target_idx]),
                    "macro_f1": float(f1_score(y_te, pred, average="macro",
                                               zero_division=0)),
                    "measured_N3": n3, "measured_R": rr,
                    "seconds": round(time.time() - t0, 1),
                })
                print(f"      overlap alpha={alpha:<4} seed={seed} {mname:4s} "
                      f"F1({target_name})={f1s[target_idx]:.4f} "
                      f"N3={n3:.3f} ({rows[-1]['seconds']}s)")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="ertam_kaya")
    ap.add_argument("--path", default=None)
    ap.add_argument("--feature_names_csv", default=None)
    ap.add_argument("--classes", nargs="+", default=None,
                    help="Target classes; default is every class with >= 60 "
                         "training rows")
    ap.add_argument("--min_partner_rows", type=int, default=500,
                    help="Minimum training rows for a class to be eligible as "
                         "the blend partner in the overlap sweep")
    ap.add_argument("--min_rows", type=int, default=60,
                    help="Minimum training rows for a class to be targeted "
                         "automatically. The rarity sweep needs headroom to "
                         "subsample, but the overlap sweep does not, so lower "
                         "this (with --skip rarity) to reach the rarest classes")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--models", nargs="+", default=["mlp", "xgb"])
    ap.add_argument("--max_rows", type=int, default=0,
                    help="Cap total rows (stratified) to keep memory in bounds; "
                         "0 uses everything")
    ap.add_argument("--skip", nargs="+", default=[],
                    help="Skip an experiment: rarity and/or overlap")
    ap.add_argument("--output", default="controlled_results")
    args = ap.parse_args()

    cfg = DATASET_PATHS.get(args.dataset, {})
    path = args.path or cfg.get("path")
    fnames = args.feature_names_csv or cfg.get("feature_names_csv")
    if not path:
        raise SystemExit(f"No path for {args.dataset}; set dataset_config.py")
    os.makedirs(args.output, exist_ok=True)
    set_seeds(GLOBAL_SEED)
    device = get_device() if "mlp" in args.models else "cpu (xgboost only)"

    print(f"[1/3] Loading {args.dataset} (device: {device})")
    if fnames:
        X_df, y_raw, _ = load_dataset(args.dataset, path, feature_names_csv=fnames)
    else:
        X_df, y_raw, _ = load_dataset(args.dataset, path)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    enc = LabelEncoder()
    y_all = enc.fit_transform(y_raw)
    class_names = enc.classes_.tolist()
    n_classes = len(class_names)
    X_all = X_df.values

    if args.max_rows and len(y_all) > args.max_rows:
        X_all, _, y_all, _ = train_test_split(
            X_all, y_all, train_size=args.max_rows, stratify=y_all,
            random_state=GLOBAL_SEED)
        print(f"      capped to {len(y_all):,} rows (stratified)")

    X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.25, stratify=y_all, random_state=GLOBAL_SEED)
    sc = StandardScaler()
    X_tr = sc.fit_transform(X_tr_raw).astype(np.float32)
    X_te = sc.transform(X_te_raw).astype(np.float32)
    print(f"      train {len(y_tr):,}  test {len(y_te):,}  classes {n_classes}")

    if args.classes:
        targets = []
        for c in args.classes:
            if c not in class_names:
                raise SystemExit(f"Class '{c}' not found. Available: {class_names}")
            targets.append(c)
    else:
        targets = [c for i, c in enumerate(class_names)
                   if (y_tr == i).sum() >= args.min_rows]
        print(f"      targeting {len(targets)} classes with "
              f">= {args.min_rows} training rows")

    seeds = [GLOBAL_SEED + i for i in range(args.seeds)]
    all_rows = []
    for t in targets:
        ti = class_names.index(t)
        print(f"\n[2/3] Target class '{t}' "
              f"({int((y_tr == ti).sum()):,} training rows, "
              f"{(y_tr == ti).mean()*100:.4f}% of training data)")
        if "rarity" not in args.skip:
            all_rows += run_rarity_sweep(X_tr, y_tr, X_te, y_te, ti, t,
                                         n_classes, seeds, args.models)
        if "overlap" not in args.skip:
            all_rows += run_overlap_sweep(X_tr, y_tr, X_te, y_te, ti, t,
                                          n_classes, seeds, args.models,
                                          class_names, args.min_partner_rows)

    print("\n[3/3] Saving")
    df = pd.DataFrame(all_rows)
    df["dataset"] = args.dataset
    csv_path = os.path.join(args.output,
                            f"{args.dataset}_controlled_results.csv")
    df.to_csv(csv_path, index=False)

    meta = {
        "dataset": args.dataset, "device": device,
        "n_classes": n_classes, "class_names": class_names,
        "train_rows": int(len(y_tr)), "test_rows": int(len(y_te)),
        "seeds": seeds, "models": args.models,
        "rarity_levels": RARITY_LEVELS, "alpha_levels": ALPHA_LEVELS,
        "mlp": {"hidden_size": HIDDEN_SIZE, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "lr": LEARNING_RATE},
        "total_seconds": float(df.seconds.sum()) if len(df) else 0.0,
        "note": ("Rarity is varied by subsampling the target class in the "
                 "training set only; the test set is identical across all "
                 "conditions. Note that subsampling also lowers the local "
                 "density of the class, so the overlap measured at small n is "
                 "not comparable with the overlap measured on the full class; "
                 "the measured_N3 column records this rather than holding it "
                 "fixed. Overlap is varied by blending every point of the "
                 "target class a fraction alpha towards a randomly matched "
                 "point of its nearest competing class. The class frequency is "
                 "identical at every level. The blend contracts the class's "
                 "own variance by up to a quarter, which is reported as a "
                 "confound rather than corrected: it makes the class easier, so "
                 "it works against an observed decline in F1, and correcting it "
                 "by per-feature rescaling was tried and found to destroy the "
                 "manipulation (see blend_class_towards). The pool of partner "
                 "points is "
                 "drawn from the training split and the same pool is used for "
                 "the training and the test split, so both are mapped into the "
                 "same region of feature space and the resulting drop in F1 "
                 "reflects class overlap rather than a train/test mismatch. "
                 "Partner classes below --min_partner_rows are not eligible, "
                 "because drawing with replacement from a small pool breaks the "
                 "target class into tight internally-pure knots and lowers the "
                 "measured overlap. Overlap is re-measured at every level so "
                 "the manipulation is verified rather than assumed."),
    }
    with open(os.path.join(args.output,
                           f"{args.dataset}_controlled_meta.json"),
              "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"      {len(df)} runs -> {csv_path}")
    if len(df):
        print(f"      total compute: {df.seconds.sum()/60:.1f} minutes on {device}")
        for exp in df.experiment.unique():
            print(f"\n   {exp} experiment, mean F1 of the target class:")
            key = "train_n_target" if exp == "rarity" else "alpha"
            piv = df[df.experiment == exp].pivot_table(
                index=["target_class", key], columns="model",
                values=["target_f1", "measured_N3"])
            print(piv.to_string())


if __name__ == "__main__":
    main()
