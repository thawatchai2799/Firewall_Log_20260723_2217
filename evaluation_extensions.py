"""
evaluation_extensions.py

Two extensions of the main comparison, each changing exactly one thing about the
evaluation protocol of rerun_with_predictions.py and keeping everything else identical
(same loader, same models and hyper-parameters, same seeds, same metrics, same output
layout). The training functions are imported from rerun_with_predictions.py rather than
copied, so there is one implementation of each model.

  --experiment time_blocks
      Replaces the random stratified folds by per-class contiguous time blocks. For every
      class, its records are put in chronological order and cut into --folds contiguous
      blocks; block k is the validation part of fold k and the other blocks are the
      training part. Every class therefore appears in both parts of every fold, but
      records that are adjacent in time (the same session or attack episode) are on the
      same side of the split except at the two block boundaries. The last fold is a pure
      forward split: the latest fifth of every class is predicted from everything
      earlier. Comparing these folds with the random folds measures how much the random
      split benefits from temporal proximity.

      The ordering key per dataset is recorded in <dataset>_split_info.json:
        unsw_nb15   capture file order, then the flow start time (Stime), then row order.
        cicids2017  the MachineLearningCVE files carry no timestamp; capture file order
                    (one file per day or half-day) then row order within the file, which
                    is the order in which CICFlowMeter emitted the flows. This is a proxy
                    for time and is labelled as such.
        ertam_kaya  the file carries no timestamp; row order is used and the split is
                    labelled "row-order blocks, chronology unknown".

  --experiment full_features
      Keeps the random stratified folds of the main comparison (same seed, so the same
      fold assignment) and replaces the common feature schema by every native feature
      of the dataset, excluding the label, identifiers (IP addresses) and absolute time
      stamps. Nominal columns (UNSW-NB15 proto, state, service) are integer-encoded the
      way the main pipeline encodes protocol. Infinite values become missing, and missing
      values are filled with the column median, exactly as in the main pipeline. The
      feature list and its length are recorded in <dataset>_feature_info.json.

Both experiments write one .json (metrics, per-class metrics, confusion matrix) and one
.npz (y_true, y_pred, proba) per unit under
    <output>/<experiment>/<dataset>/<variant>/seed<S>_fold<F>.{json,npz}
and are resumable: a unit with a .json is skipped. --summarize writes
<output>/<experiment>/metrics.csv, per_class.csv and confusion_matrices.json in the same
columns as the main re-run, plus the columns experiment and n_features.

Defaults are the three headline variants: xgb_classweight, xgb_smote, mlp_baseline_smote.

Usage (from the folder that holds rerun_with_predictions.py and the pipeline modules):
  python evaluation_extensions.py --synthetic --experiment time_blocks --folds 2
  python evaluation_extensions.py --experiment time_blocks --datasets ertam_kaya cicids2017 unsw_nb15
  python evaluation_extensions.py --experiment full_features --datasets unsw_nb15
  python evaluation_extensions.py --experiment full_features --summarize
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    import rerun_with_predictions as R
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "This script must sit in the SAME folder as rerun_with_predictions.py, "
        "data_loader.py, dataset_config.py, imbalance_methods.py, model.py and "
        "tree_baselines.py, and be run from there.\n")

HEADLINE = ["xgb_classweight", "xgb_smote", "mlp_baseline_smote"]
EXPERIMENTS = ["time_blocks", "full_features"]


def _hms(s):
    return R._hms(s)


# ---------------------------------------------------------------------------
# Raw data access (for ordering keys and native features)

def load_raw(dataset_key):
    """Return (raw DataFrame, file index per row, row index within file)."""
    import data_loader as DL
    cfg = R.DATASET_PATHS[dataset_key]
    path = cfg["path"]
    if dataset_key == "ertam_kaya":
        raw = DL._load_ertam_kaya(path)
        return raw, np.zeros(len(raw), int), np.arange(len(raw))
    if dataset_key == "cicids2017":
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "*.csv")))
            frames, fidx, ridx = [], [], []
            for i, f in enumerate(files):
                d = DL._strip_columns(pd.read_csv(f, low_memory=False))
                frames.append(d)
                fidx.append(np.full(len(d), i))
                ridx.append(np.arange(len(d)))
            raw = pd.concat(frames, ignore_index=True)
            return raw, np.concatenate(fidx), np.concatenate(ridx)
        raw = DL._strip_columns(pd.read_csv(path, low_memory=False))
        return raw, np.zeros(len(raw), int), np.arange(len(raw))
    if dataset_key == "unsw_nb15":
        # replicate _load_unsw_nb15 but keep the file index
        feat_csv = cfg.get("feature_names_csv")
        col_names = None
        if feat_csv and os.path.exists(feat_csv):
            fd = pd.read_csv(feat_csv, encoding="latin1")
            name_col = "Name" if "Name" in fd.columns else fd.columns[1]
            col_names = fd[name_col].str.strip().tolist()
        if os.path.isdir(path):
            files = sorted(glob.glob(os.path.join(path, "UNSW-NB15_*.csv")))
            frames, fidx, ridx = [], [], []
            for i, f in enumerate(files):
                d = pd.read_csv(f, header=None, low_memory=False)
                if col_names and len(col_names) == d.shape[1]:
                    d.columns = col_names
                frames.append(d)
                fidx.append(np.full(len(d), i))
                ridx.append(np.arange(len(d)))
            raw = pd.concat(frames, ignore_index=True)
            fi, ri = np.concatenate(fidx), np.concatenate(ridx)
        else:
            raw = DL._strip_columns(pd.read_csv(path, low_memory=False))
            fi, ri = np.zeros(len(raw), int), np.arange(len(raw))
        if "attack_cat" in raw.columns:
            raw["attack_cat"] = raw["attack_cat"].fillna("Normal").str.strip()
        return raw, fi, ri
    raise ValueError(dataset_key)


def raw_labels(dataset_key, raw):
    from feature_schema import ALL_SPECS
    y = raw[ALL_SPECS[dataset_key].label_column].astype(str).str.strip()
    if dataset_key == "unsw_nb15":
        y = y.replace({"Backdoor": "Backdoors"})
    return y.values


def check_alignment(dataset_key, y_canon_names, raw):
    yr = raw_labels(dataset_key, raw)
    if len(yr) != len(y_canon_names):
        raise RuntimeError(f"{dataset_key}: raw has {len(yr)} rows, canonical has "
                           f"{len(y_canon_names)}; row alignment is broken")
    mism = int((yr != np.asarray(y_canon_names)).sum())
    if mism:
        raise RuntimeError(f"{dataset_key}: {mism} rows whose raw label differs from the "
                           "canonical label; row alignment is broken")


def find_col(raw, *cands):
    low = {c.lower(): c for c in raw.columns}
    for c in cands:
        if c.lower() in low:
            return low[c.lower()]
    return None


# ---------------------------------------------------------------------------
# Experiment 1: per-class contiguous time blocks

def order_key(dataset_key, raw, fidx, ridx):
    """Return (sortable key array, description dict)."""
    n = len(raw)
    if dataset_key == "unsw_nb15":
        col = find_col(raw, "Stime")
        if col is None:
            return np.lexsort((ridx, fidx)), {"order": "file order, then row order",
                                               "chronology_known": False,
                                               "note": "Stime column not found"}
        st = pd.to_numeric(raw[col], errors="coerce").fillna(-1).values
        return np.lexsort((ridx, st, fidx)), {
            "order": "file order, then Stime (flow start, epoch seconds), then row order",
            "chronology_known": True,
            "stime_missing": int((st < 0).sum()),
            "stime_min": float(st[st >= 0].min()) if (st >= 0).any() else None,
            "stime_max": float(st.max())}
    if dataset_key == "cicids2017":
        return np.lexsort((ridx, fidx)), {
            "order": "capture file order (one file per day or half-day), then row order "
                     "within the file (CICFlowMeter emission order)",
            "chronology_known": "proxy",
            "n_files": int(fidx.max() + 1),
            "note": "MachineLearningCVE files carry no timestamp column"}
    return np.arange(n), {"order": "row order of the published file",
                          "chronology_known": False,
                          "note": "no timestamp in the Ertam & Kaya file"}


def time_block_folds(y, order, n_folds):
    """order: permutation that sorts rows chronologically. Returns list of (train, val)
    index arrays: for every class, its chronologically ordered rows are cut into n_folds
    contiguous blocks and block k is the validation part of fold k."""
    rank = np.empty(len(order), int)
    rank[order] = np.arange(len(order))          # rank[i] = chronological position of row i
    val_sets = [[] for _ in range(n_folds)]
    for c in np.unique(y):
        rows = np.where(y == c)[0]
        rows = rows[np.argsort(rank[rows], kind="stable")]
        for k, chunk in enumerate(np.array_split(rows, n_folds)):
            val_sets[k].append(chunk)
    folds = []
    all_idx = np.arange(len(y))
    for k in range(n_folds):
        va = np.sort(np.concatenate(val_sets[k]))
        mask = np.ones(len(y), bool)
        mask[va] = False
        folds.append((all_idx[mask], va))
    return folds


# ---------------------------------------------------------------------------
# Experiment 2: native feature set

def native_features(dataset_key, raw):
    """Return (X float64 array, feature names, info dict)."""
    from feature_schema import ALL_SPECS
    label_col = ALL_SPECS[dataset_key].label_column
    drop = {label_col}
    if dataset_key == "unsw_nb15":
        for c in ("srcip", "dstip", "Stime", "Ltime", "Label", "attack_cat"):
            col = find_col(raw, c)
            if col:
                drop.add(col)
    if dataset_key == "cicids2017":
        for c in ("Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"):
            col = find_col(raw, c)
            if col:
                drop.add(col)
    cols = [c for c in raw.columns if c not in drop]
    X = raw[cols].copy()
    encoded, dropped_constant = [], []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            num = pd.to_numeric(X[c], errors="coerce")
            # a column that is mostly numeric text (e.g. ports with a few hex values):
            # keep it numeric, as the main pipeline does for the canonical ports
            if num.notna().mean() >= 0.9:
                X[c] = num
            else:
                codes, uniq = pd.factorize(X[c].astype(str).fillna("unknown"))
                X[c] = codes
                encoded.append({"column": c, "n_levels": int(len(uniq))})
    X = X.astype(np.float64)
    X = X.replace([np.inf, -np.inf], np.nan)
    n_missing = int(X.isna().sum().sum())
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    nunique = X.nunique()
    const = nunique[nunique <= 1].index.tolist()
    if const:
        dropped_constant = const
        X = X.drop(columns=const)
    info = {"n_features": int(X.shape[1]), "features": X.columns.tolist(),
            "dropped_columns": sorted(drop), "integer_encoded": encoded,
            "dropped_constant_columns": dropped_constant,
            "n_values_imputed": n_missing}
    return X.values, X.columns.tolist(), info


# ---------------------------------------------------------------------------
# Synthetic data for the smoke test

def make_synthetic_raw(seed=0):
    X_df, y_raw, names = R.make_synthetic(seed)
    rng = np.random.default_rng(seed)
    raw = X_df.copy()
    raw["extra_num"] = rng.normal(size=len(raw))
    raw["extra_cat"] = rng.choice(["tcp", "udp", "icmp"], size=len(raw))
    raw["const"] = 1.0
    raw["label"] = y_raw.values
    stime = np.cumsum(rng.exponential(1.0, size=len(raw)))
    return X_df, y_raw, raw, stime


# ---------------------------------------------------------------------------
# Driver

def unit_paths(root, ds, variant, seed, fold):
    d = os.path.join(root, ds, variant)
    stem = os.path.join(d, f"seed{seed}_fold{fold}")
    return d, stem + ".json", stem + ".npz"


def run(args):
    root = os.path.join(args.output, args.experiment)
    os.makedirs(root, exist_ok=True)
    log = R.Log(os.path.join(root, "progress.log"))
    variants = [v for v in R.variant_list(["mlp", "xgb", "lgbm"]) if v[1] in args.variants]
    missing = set(args.variants) - {v[1] for v in variants}
    if missing:
        raise SystemExit(f"unknown variants: {sorted(missing)}")
    seeds = [R.SEED_BASE + i for i in range(args.seeds)]
    datasets = ["synthetic"] if args.synthetic else args.datasets
    folds_to_run = args.only_folds if args.only_folds else list(range(args.folds))
    log(f"experiment={args.experiment} device={R.DEVICE} datasets={datasets} "
        f"variants={[v[1] for v in variants]} seeds={seeds} folds={args.folds} "
        f"run_folds={folds_to_run}")

    total = len(datasets) * len(variants) * len(seeds) * len(folds_to_run)
    done = 0
    per_kind = {}       # (ds, variant) -> list of seconds, for the ETA
    t_all = time.time()

    for ds in datasets:
        log(f"loading {ds}")
        if args.synthetic:
            X_df, y_raw, raw, stime = make_synthetic_raw()
            X_df = X_df.replace([np.inf, -np.inf], np.nan)
            X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
            enc = LabelEncoder()
            y = enc.fit_transform(y_raw)
            class_names = enc.classes_.tolist()
            X = X_df.values.astype(np.float64)
            feature_names = X_df.columns.tolist()
            fidx, ridx = np.zeros(len(y), int), np.arange(len(y))
        else:
            X, y, class_names, feature_names = R.load_prepared(ds, False)
            raw, fidx, ridx = load_raw(ds)
            check_alignment(ds, np.asarray(class_names)[y], raw)
        n_classes = len(class_names)
        log(f"  {X.shape[0]:,} rows, {n_classes} classes, canonical features: {X.shape[1]}")

        if args.experiment == "time_blocks":
            if args.synthetic:
                order = np.argsort(stime, kind="stable")
                desc = {"order": "synthetic stime", "chronology_known": True}
            else:
                order, desc = order_key(ds, raw, fidx, ridx)
            folds = time_block_folds(y, order, args.folds)
            feats = feature_names
            info = {"dataset": ds, "experiment": "time_blocks", "n_folds": args.folds,
                    "ordering": desc, "class_names": class_names,
                    "val_sizes": [int(len(v)) for _, v in folds],
                    "val_class_counts": [np.bincount(y[v], minlength=n_classes).tolist()
                                         for _, v in folds]}
            with open(os.path.join(root, f"{ds}_split_info.json"), "w",
                      encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            log(f"  ordering: {desc['order']}")
        else:
            if args.synthetic:
                Xn, feats, finfo = native_features_synthetic(raw)
            else:
                Xn, feats, finfo = native_features(ds, raw)
            if len(Xn) != len(y):
                raise RuntimeError("native feature matrix is misaligned with labels")
            X = Xn
            skf = StratifiedKFold(n_splits=args.folds, shuffle=True,
                                  random_state=R.SEED_BASE)
            folds = list(skf.split(X, y))
            finfo.update({"dataset": ds, "experiment": "full_features",
                          "canonical_n_features": len(feature_names)})
            with open(os.path.join(root, f"{ds}_feature_info.json"), "w",
                      encoding="utf-8") as f:
                json.dump(finfo, f, indent=2)
            log(f"  native features: {len(feats)} "
                f"(encoded: {[e['column'] for e in finfo['integer_encoded']]}, "
                f"constant dropped: {finfo['dropped_constant_columns']})")
        del raw

        with open(os.path.join(root, f"{ds}_class_names.json"), "w", encoding="utf-8") as f:
            json.dump({"class_names": class_names, "feature_names": feats,
                       "class_counts": np.bincount(y, minlength=n_classes).tolist()}, f,
                      indent=2)

        for fold_idx, (tr, va) in enumerate(folds):
            if fold_idx not in folds_to_run:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            y_tr, y_va = y[tr], y[va]
            if len(np.unique(y_tr)) != n_classes:
                log(f"  NOTE fold {fold_idx}: {n_classes - len(np.unique(y_tr))} class(es) "
                    "absent from the training part")

            for family, variant, spec in variants:
                for seed in seeds:
                    d, jp, npz = unit_paths(root, ds, variant, seed, fold_idx)
                    uid = f"{args.experiment}/{ds}/{variant}/seed{seed}/fold{fold_idx}"
                    if os.path.exists(jp):
                        done += 1
                        continue
                    os.makedirs(d, exist_ok=True)
                    kind = per_kind.setdefault((ds, variant), [])
                    eta = ""
                    if kind:
                        remaining = sum(
                            np.mean(per_kind[k]) * cnt for k, cnt in _remaining(
                                root, datasets, variants, seeds, folds_to_run).items()
                            if k in per_kind)
                        eta = f"  ~{_hms(remaining)} left for timed kinds"
                    log(f"[{done + 1}/{total}] {uid}{eta}")
                    t0 = time.time()
                    try:
                        def hb(k, n, el, u=uid):
                            left = el / k * (n - k) if k else 0
                            log(f"    {u}: epoch {k}/{n} (~{_hms(left)} left)")
                        if family == "mlp":
                            y_pred, proba, info = R.run_mlp(spec, X_tr, y_tr, X_va,
                                                            n_classes, seed, heartbeat=hb)
                        else:
                            y_pred, proba, info = R.run_tree(family, spec, X_tr, y_tr,
                                                             X_va, n_classes, seed)
                    except Exception as e:
                        import traceback
                        log(f"    FAILED {uid}: {e}\n{traceback.format_exc()}")
                        continue
                    secs = time.time() - t0
                    kind.append(secs)
                    m = R.compute_metrics(y_va, y_pred, n_classes, class_names)
                    m.update({"dataset": ds, "variant": variant, "family": family,
                              "technique_or_mode": spec, "seed": seed, "fold": fold_idx,
                              "experiment": args.experiment,
                              "n_features": int(X_tr.shape[1]),
                              "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                              "n_classes": n_classes, "class_names": class_names,
                              "train_seconds": round(secs, 2), "device": R.DEVICE, **info})
                    np.savez_compressed(npz, y_true=y_va.astype(np.int16),
                                        y_pred=y_pred.astype(np.int16),
                                        proba=proba.astype(np.float16))
                    with open(jp, "w", encoding="utf-8") as f:
                        json.dump(m, f)
                    done += 1
                    log(f"    done in {_hms(secs)}  macro-F1 {m['macro_f1']:.4f}  "
                        f"total elapsed {_hms(time.time() - t_all)}")
    log(f"finished {done}/{total} units in {_hms(time.time() - t_all)}")
    summarize(root, log)


def _remaining(root, datasets, variants, seeds, folds_to_run):
    out = {}
    for ds in datasets:
        for _, variant, _ in variants:
            n = 0
            for seed in seeds:
                for k in folds_to_run:
                    if not os.path.exists(unit_paths(root, ds, variant, seed, k)[1]):
                        n += 1
            out[(ds, variant)] = n
    return out


def native_features_synthetic(raw):
    cols = [c for c in raw.columns if c != "label"]
    X = raw[cols].copy()
    enc = []
    for c in cols:
        if not pd.api.types.is_numeric_dtype(X[c]):
            codes, uniq = pd.factorize(X[c].astype(str))
            X[c] = codes
            enc.append({"column": c, "n_levels": int(len(uniq))})
    X = X.astype(np.float64)
    nun = X.nunique()
    const = nun[nun <= 1].index.tolist()
    X = X.drop(columns=const)
    return X.values, X.columns.tolist(), {
        "n_features": int(X.shape[1]), "features": X.columns.tolist(),
        "dropped_columns": ["label"], "integer_encoded": enc,
        "dropped_constant_columns": const, "n_values_imputed": 0}


def summarize(root, log=print):
    rows, cls_rows, cms = [], [], {}
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("seed") and fn.endswith(".json"):
                with open(os.path.join(r, fn), encoding="utf-8") as f:
                    m = json.load(f)
                base = {k: m[k] for k in ["experiment", "dataset", "variant", "family",
                                          "technique_or_mode", "seed", "fold", "n_features",
                                          "n_train", "n_train_after_resampling", "n_val",
                                          "train_seconds", "device"]}
                rows.append({**base, **{k: m[k] for k in [
                    "accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                    "macro_recall", "weighted_f1", "n_classes_predicted"]}})
                for pc in m["per_class"]:
                    cls_rows.append({**base, **pc})
                cms[f"{m['dataset']}/{m['variant']}/seed{m['seed']}/fold{m['fold']}"] = {
                    "class_names": m["class_names"], "confusion_matrix": m["confusion_matrix"]}
    if not rows:
        log("summarize: no finished units")
        return
    df = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed", "fold"])
    df.to_csv(os.path.join(root, "metrics.csv"), index=False)
    pd.DataFrame(cls_rows).to_csv(os.path.join(root, "per_class.csv"), index=False)
    with open(os.path.join(root, "confusion_matrices.json"), "w", encoding="utf-8") as f:
        json.dump(cms, f)
    agg = df.groupby(["dataset", "variant"]).macro_f1.agg(["count", "mean", "std"])
    log("macro-F1 by (dataset, variant)\n" + agg.round(4).to_string())
    log(f"wrote metrics.csv ({len(df)} units), per_class.csv, confusion_matrices.json "
        f"in {os.path.abspath(root)}")


def selftest_folds():
    """Checks of the time-block fold logic that need no model."""
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [100, 23, 7])
    t = rng.random(len(y))
    order = np.argsort(t)
    folds = time_block_folds(y, order, 5)
    ok = True
    rank = np.empty(len(y), int); rank[order] = np.arange(len(y))
    for k, (tr, va) in enumerate(folds):
        ok &= len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(y)
        for c in [0, 1, 2]:
            vc = va[y[va] == c]
            tc = tr[y[tr] == c]
            # validation block is contiguous in time within the class
            r = np.sort(rank[vc])
            others = rank[y == c]
            inside = others[(others >= r.min()) & (others <= r.max())]
            ok &= len(inside) == len(r)
            ok &= len(tc) > 0
    allv = np.sort(np.concatenate([va for _, va in folds]))
    ok &= np.array_equal(allv, np.arange(len(y)))
    # last fold is the latest block of every class
    _, va_last = folds[-1]
    for c in [0, 1, 2]:
        ok &= rank[va_last[y[va_last] == c]].min() > rank[np.setdiff1d(np.where(y == c)[0], va_last)].max()
    print("  fold logic:", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    ap.add_argument("--datasets", nargs="+", default=["ertam_kaya", "cicids2017", "unsw_nb15"])
    ap.add_argument("--variants", nargs="+", default=HEADLINE)
    ap.add_argument("--output", default="extension_results")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only_folds", type=int, nargs="+", default=None,
                    help="run only these fold indices (e.g. 0 1 2) to fit a time budget")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize(os.path.join(args.output, args.experiment))
        return 0
    if args.synthetic:
        if not selftest_folds():
            return 1
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
