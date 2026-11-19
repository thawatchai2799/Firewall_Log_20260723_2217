"""
rerun_with_predictions.py

Re-run the main cross-validated comparison and KEEP the predictions this time.

Why this script exists
----------------------
The original run (run_full_experiment.py / tree_baselines.py) stored only macro-F1 and
per-class F1 for every fold. Per-class precision and recall, confusion matrices,
false-positive rates, calibration, and confidence intervals on the headline numbers cannot be
recovered from an F1 score, so the models have to be trained again with the predictions
written to disk.

What is held identical to the original run
------------------------------------------
  * data loading and imputation       : data_loader.load_dataset, inf -> NaN -> median -> 0
  * fold assignment                   : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  * scaling                           : StandardScaler fitted on the training fold only
  * the four MLP techniques           : imbalance_methods.TECHNIQUES (capped SMOTE etc.)
  * the MLP                           : model.FirewallMLP, 200 hidden, dropout 0.25,
                                        Adam lr 1e-3, 100 epochs, batch 512, no early stopping
  * the tree models                   : tree_baselines._make_xgb / _make_lgbm, unchanged
  * the TabNet model (optional)       : tabnet_baseline._make_tabnet and its fit protocol

All of these are IMPORTED from the existing modules, not copied, so this script cannot drift
from the pipeline it re-runs. It must therefore sit in the same folder as those modules.

What is different, and why
--------------------------
  * Every unit is seeded explicitly (python, numpy, torch, CUDA). The original MLP loop drew
    its initialisation and batch order from an unseeded torch RNG, so its numbers were not
    exactly reproducible. The new numbers will therefore differ from the published table by
    seed noise; that is expected, and the across-seed spread is now reported instead of hidden.
  * Several seeds per fold (default 3) so that confidence intervals have more than five points.
    The SMOTE random_state stays fixed at 42, as in the original: the seed varies the model,
    not the resampled data.
  * Predicted labels, predicted probabilities and true labels are saved for every unit, so
    confusion matrices, FPR and calibration can be computed afterwards without retraining.

Resumability
------------
Each (dataset, variant, seed, fold) unit writes its own files and is skipped on the next run
if its metrics file exists. The arrays are written first and the metrics file last, so a unit
interrupted by Ctrl+C or a power cut is simply re-run next time. Re-run the same command to
continue; finished units are never recomputed.

Known behaviour, verified on synthetic data before release
----------------------------------------------------------
  * Same seed -> bit-identical predictions on CPU for every family (MLP, XGBoost, LightGBM,
    TabNet). On CUDA the MLP is identical in practice for this architecture (Linear/ReLU/
    Dropout only); cudnn.deterministic is set, but a different GPU driver can in principle
    change the last decimal.
  * A class absent from a training fold: LightGBM and TabNet are handled (its probability
    column is filled with zeros so the array is always n_classes wide); XGBoost raises on
    non-contiguous labels and that unit is logged as FAILED while the run continues. With
    StratifiedKFold this cannot happen for any class with at least two rows, which is every
    class in all three datasets.
  * Memory: the data are held in float64 exactly as the original pipeline held them. For
    CIC-IDS2017 (2.8M rows x 78 features) that is ~1.8 GB for X plus per-fold copies, which
    fitted on the 32 GB machine the original run used; on a 16 GB machine run that dataset
    last and alone.
  * Disk: predictions are stored as compressed float16 and cost about 6 bytes per validation
    row per 6 classes; the whole three-dataset run stays under ~2 GB. --no_proba drops the
    probabilities (and with them calibration) if space is tight.

Usage
-----
    # 2-minute smoke test on synthetic data, no real data needed (do this first):
    python rerun_with_predictions.py --synthetic --output smoke_test

    # the real thing, one dataset at a time, cheapest first:
    python rerun_with_predictions.py --datasets ertam_kaya --output rerun_results
    python rerun_with_predictions.py --datasets unsw_nb15  --output rerun_results
    python rerun_with_predictions.py --datasets cicids2017 --output rerun_results

    # from a second terminal while it runs: how far along is it?
    python rerun_with_predictions.py --status --output rerun_results

    # rebuild the CSV summaries from whatever units are finished:
    python rerun_with_predictions.py --summarize --output rerun_results

Progress reporting
------------------
Before and after every unit the log shows: units done / total and the percentage, time
elapsed in this session, estimated time remaining, and the expected finish as a local
date and time. The estimate is per (dataset, variant), so it does not assume an MLP on
CIC-IDS2017 costs what an MLP on the firewall log cost; until each kind of unit has been
timed once it is reported as a lower bound. Inside a long MLP unit a line every 10
epochs shows that unit's own percentage and time left.
"""

import argparse
import json
import os
import random
import sys
import time
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

# --------------------------------------------------------------------------------------
# Imports from the existing pipeline. Failing loudly here is better than silently
# re-implementing something slightly differently.
# --------------------------------------------------------------------------------------
try:
    from dataset_config import (BATCH_SIZE, DATASET_PATHS, EPOCHS, HIDDEN_SIZE,
                                LEARNING_RATE, N_FOLDS)
    from imbalance_methods import TECHNIQUES, _safe_smote
    from model import FirewallMLP
    from tree_baselines import _HAS_LGBM, _HAS_XGB, _make_lgbm, _make_xgb
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "This script must sit in the SAME folder as dataset_config.py, data_loader.py,\n"
        "imbalance_methods.py, model.py and tree_baselines.py, and be run from there.\n")

try:
    from tabnet_baseline import _HAS_TABNET, _make_tabnet
except Exception:  # tabnet_baseline.py absent or pytorch-tabnet not installed
    _HAS_TABNET = False
    _make_tabnet = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED_BASE = 42

MLP_TECHNIQUES = ["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"]
TREE_MODES = ["plain", "smote", "classweight"]


# ======================================================================================
# Seeding
# ======================================================================================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================================
# Data
# ======================================================================================
def make_synthetic(seed: int = 0):
    """A small, severely imbalanced multi-class problem for smoke-testing every code path.

    One class has 12 rows so the rare-class branches (k_neighbors clamp in SMOTE, a class
    that may be absent from a prediction, zero-division guards) are all exercised.
    """
    rng = np.random.default_rng(seed)
    sizes = [6000, 1500, 400, 90, 30, 12]
    n_features = 12
    X, y = [], []
    for c, n in enumerate(sizes):
        centre = rng.normal(0, 2.0, n_features)
        X.append(rng.normal(centre, 1.0, (n, n_features)))
        y.append(np.full(n, c))
    X = np.vstack(X)
    y = np.concatenate(y)
    names = [f"class_{c}" for c in range(len(sizes))]
    X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    return X_df, pd.Series([names[i] for i in y]), names


def load_prepared(dataset_key: str, synthetic: bool):
    if synthetic:
        X_df, y_raw, _ = make_synthetic()
    else:
        from data_loader import load_dataset
        cfg = DATASET_PATHS[dataset_key]
        X_df, y_raw, _ = load_dataset(dataset_key, cfg["path"],
                                      feature_names_csv=cfg.get("feature_names_csv"))
    # identical to run_full_experiment / tree_baselines
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    enc = LabelEncoder()
    y = enc.fit_transform(y_raw)
    class_names = enc.classes_.tolist()
    X = X_df.values.astype(np.float64)
    return X, y, class_names, X_df.columns.tolist()


# ======================================================================================
# Model runners. Each returns (y_pred, proba, info) where proba is (n_val, n_classes).
# ======================================================================================
def _full_width_proba(model, X_val, n_classes):
    """predict_proba with columns aligned to 0..n_classes-1 even if a class was absent
    from the training fold (model.classes_ would then be shorter)."""
    p = model.predict_proba(X_val)
    classes = np.asarray(getattr(model, "classes_", np.arange(p.shape[1])), dtype=int)
    if p.shape[1] == n_classes and np.array_equal(classes, np.arange(n_classes)):
        return p
    full = np.zeros((p.shape[0], n_classes), dtype=p.dtype)
    full[:, classes] = p
    return full


def run_mlp(technique: str, X_train, y_train, X_val, n_classes, seed, heartbeat=None):
    """Exactly train_and_eval_fold() from run_full_experiment.py, plus probabilities."""
    set_all_seeds(seed)
    tech_fn = TECHNIQUES[technique]
    X_res, y_res, loss_fn = tech_fn(X_train, y_train, n_classes)   # SMOTE rs=42 inside

    model = FirewallMLP(n_features=X_train.shape[1], n_classes=n_classes,
                        hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(DEVICE)

    X_t = torch.as_tensor(np.asarray(X_res), dtype=torch.float32)
    y_t = torch.as_tensor(np.asarray(y_res), dtype=torch.long)
    n = X_t.shape[0]

    t_train = time.time()          # resampling is done; time only the epochs for the ETA
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = X_t[idx].to(DEVICE)
            yb = y_t[idx].to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        if heartbeat is not None and epoch % 10 == 0:
            heartbeat(epoch + 1, EPOCHS, time.time() - t_train)

    model.eval()
    Xv = torch.as_tensor(np.asarray(X_val), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for start in range(0, Xv.shape[0], 8192):
            xb = Xv[start:start + 8192].to(DEVICE)
            out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    proba = np.concatenate(out, axis=0)
    y_pred = proba.argmax(axis=1)
    return y_pred, proba, {"n_train_after_resampling": int(n)}


def run_tree(family: str, mode: str, X_train, y_train, X_val, n_classes, seed):
    """Exactly _fit_predict() from tree_baselines.py, plus probabilities."""
    set_all_seeds(seed)
    factory = _make_xgb if family == "xgb" else _make_lgbm
    model = factory(n_classes, random_state=seed)

    if mode == "smote":
        X_fit, y_fit = _safe_smote(X_train, y_train, random_state=SEED_BASE)
        sw = None
    elif mode == "classweight":
        X_fit, y_fit = X_train, y_train
        sw = compute_sample_weight(class_weight="balanced", y=y_fit)
    else:
        X_fit, y_fit = X_train, y_train
        sw = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sw is not None:
            model.fit(X_fit, y_fit, sample_weight=sw)
        else:
            model.fit(X_fit, y_fit)
        proba = _full_width_proba(model, X_val, n_classes)
    y_pred = proba.argmax(axis=1)
    return y_pred, proba, {"n_train_after_resampling": int(len(y_fit))}


def run_tabnet(mode: str, X_train, y_train, X_val, n_classes, seed):
    """Exactly _fit_predict() from tabnet_baseline.py, plus probabilities."""
    from sklearn.model_selection import train_test_split
    set_all_seeds(seed)
    if mode == "smote":
        X_fit, y_fit = _safe_smote(X_train, y_train, random_state=SEED_BASE)
        weights = 0
    elif mode == "classweight":
        X_fit, y_fit = X_train, y_train
        weights = 1
    else:
        X_fit, y_fit = X_train, y_train
        weights = 0

    clf = _make_tabnet(random_state=seed)
    try:
        X_in, X_es, y_in, y_es = train_test_split(
            X_fit, y_fit, test_size=0.1, random_state=seed, stratify=y_fit)
        eval_set = [(X_es.astype(np.float32), y_es)]
    except ValueError:
        X_in, y_in, eval_set = X_fit, y_fit, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_in.astype(np.float32), y_in, eval_set=eval_set,
                max_epochs=100, patience=15, batch_size=1024,
                virtual_batch_size=256, weights=weights, drop_last=False)
        proba = _full_width_proba(clf, X_val.astype(np.float32), n_classes)
    y_pred = proba.argmax(axis=1)
    return y_pred, proba, {"n_train_after_resampling": int(len(y_fit))}


# ======================================================================================
# Metrics from (y_true, y_pred): everything derivable from one confusion matrix.
# ======================================================================================
def compute_metrics(y_true, y_pred, n_classes, class_names):
    labels = list(range(n_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(cm.sum())
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)

    per_class = []
    for c in labels:
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = total - tp - fp - fn
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        per_class.append({
            "class_index": c, "class_name": class_names[c],
            "precision": float(prec[c]), "recall": float(rec[c]), "f1": float(f1[c]),
            "support": int(sup[c]), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "fpr": float(fpr), "predicted_count": int(cm[:, c].sum()),
        })

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(np.mean(prec)),
        "macro_recall": float(np.mean(rec)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "n_classes_predicted": int(len(np.unique(y_pred))),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


# ======================================================================================
# Units and bookkeeping
# ======================================================================================
def variant_list(models):
    out = []
    if "mlp" in models:
        out += [("mlp", f"mlp_{t}", t) for t in MLP_TECHNIQUES]
    if "xgb" in models:
        out += [("xgb", f"xgb_{m}", m) for m in TREE_MODES]
    if "lgbm" in models:
        out += [("lgbm", f"lgbm_{m}", m) for m in TREE_MODES]
    if "tabnet" in models:
        out += [("tabnet", f"tabnet_{m}", m) for m in TREE_MODES]
    return out


def unit_paths(output, dataset_key, variant, seed, fold):
    d = os.path.join(output, dataset_key, variant)
    stem = os.path.join(d, f"seed{seed}_fold{fold}")
    return d, stem + ".json", stem + ".npz"


class Log:
    def __init__(self, path):
        self.path = path
        self.t0 = time.time()

    def __call__(self, msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _hms(seconds):
    return str(timedelta(seconds=int(max(0, seconds))))


class Eta:
    """Time-remaining estimate for a job whose units differ in cost by two orders of
    magnitude (an MLP unit on CIC-IDS2017 takes minutes; an XGBoost unit on the firewall
    log takes seconds).

    Each unit is keyed by (dataset, variant). The estimate is the sum, over the units still
    to do, of the mean time already observed for that key. A key with no observation yet
    is NOT guessed from other keys -- an MLP on a 2.8M-row dataset tells you nothing about
    an MLP on a 65k-row one -- so such units are counted separately and the figure is
    reported as a lower bound until every key has been seen once. After the first fold of
    a dataset every key on that dataset has been seen and the estimate is complete.

    On resume, the train_seconds stored in finished units' json files seed the means, so
    the first line printed after a restart is already informed.
    """

    def __init__(self, remaining_keys, output):
        self.remaining = list(remaining_keys)            # (dataset, variant) per pending unit
        self.times = {}                                  # (dataset, variant) -> [seconds]
        self.t_start = time.time()
        for root, _, files in os.walk(output):
            for fn in files:
                if fn.startswith("seed") and fn.endswith(".json"):
                    try:
                        with open(os.path.join(root, fn), encoding="utf-8") as f:
                            m = json.load(f)
                        self.times.setdefault((m["dataset"], m["variant"]), []) \
                            .append(float(m["train_seconds"]))
                    except Exception:
                        pass

    def record(self, key, seconds):
        self.times.setdefault(key, []).append(float(seconds))
        if key in self.remaining:
            self.remaining.remove(key)

    def seconds_left(self):
        """(seconds for units whose key has been timed, number of units not yet timed)."""
        known, untimed = 0.0, 0
        for k in self.remaining:
            if self.times.get(k):
                known += float(np.mean(self.times[k]))
            else:
                untimed += 1
        return known, untimed

    def line(self, done, total):
        pct = 100.0 * done / total if total else 100.0
        elapsed = time.time() - self.t_start
        head = f"progress {done}/{total} ({pct:.1f}%)  elapsed {_hms(elapsed)}"
        if not self.remaining:
            return head + "  remaining 0:00:00"
        known, untimed = self.seconds_left()
        if untimed == len(self.remaining):
            return head + "  remaining: not yet estimable (no unit of this kind timed yet)"
        finish = time.strftime("%a %d %b %Y %H:%M", time.localtime(time.time() + known))
        if untimed:
            return (head + f"  remaining >{_hms(known)} for the {len(self.remaining) - untimed} "
                    f"timed units, plus {untimed} not yet timed  "
                    f"expected finish later than ~{finish}")
        return head + f"  remaining ~{_hms(known)}  expected finish ~{finish}"


def run(args):
    os.makedirs(args.output, exist_ok=True)
    log = Log(os.path.join(args.output, "progress.log"))
    seeds = [SEED_BASE + i for i in range(args.seeds)]
    variants = variant_list(args.models)

    missing = []
    if "xgb" in args.models and not _HAS_XGB:
        missing.append("xgboost")
    if "lgbm" in args.models and not _HAS_LGBM:
        missing.append("lightgbm")
    if "tabnet" in args.models and not _HAS_TABNET:
        missing.append("pytorch-tabnet")
    if missing:
        raise SystemExit(f"[setup error] not installed: {', '.join(missing)}")

    datasets = ["synthetic"] if args.synthetic else args.datasets
    log(f"device={DEVICE}  datasets={datasets}  models={args.models}  "
        f"seeds={seeds}  folds={args.folds}  save_proba={not args.no_proba}")

    # Enumerate every unit up front so progress and ETA refer to the whole job, and so
    # the ETA knows how many slow and fast units are still ahead.
    total_units = len(datasets) * len(variants) * len(seeds) * args.folds
    pending_keys = []
    done_units = 0
    for dataset_key in datasets:
        for fold_idx in range(args.folds):
            for family, variant, _ in variants:
                for seed in seeds:
                    _, jp, _ = unit_paths(args.output, dataset_key, variant, seed, fold_idx)
                    if os.path.exists(jp):
                        done_units += 1
                    else:
                        pending_keys.append((dataset_key, variant))
    eta = Eta(pending_keys, args.output)
    failed = []
    if done_units:
        log(f"resuming: {done_units}/{total_units} units already finished")
    log(eta.line(done_units, total_units))

    try:
        _run_units(args, log, datasets, variants, seeds, eta, failed, done_units, total_units)
    except KeyboardInterrupt:
        log("INTERRUPTED (Ctrl+C). Every finished unit is saved; the unit in progress was "
            "discarded. Run the same command again to continue from here.")
        summarize(args.output, log)
        return


def _run_units(args, log, datasets, variants, seeds, eta, failed, done_units, total_units):
    for dataset_key in datasets:
        log(f"loading {dataset_key}")
        X, y, class_names, feature_names = load_prepared(dataset_key, args.synthetic)
        n_classes = len(class_names)
        log(f"  {X.shape[0]:,} rows, {X.shape[1]} features, {n_classes} classes")
        with open(os.path.join(args.output, f"{dataset_key}_class_names.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"class_names": class_names, "feature_names": feature_names,
                       "class_counts": np.bincount(y, minlength=n_classes).tolist()},
                      f, indent=2)

        # SAME fold object as the original pipeline.
        skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED_BASE)
        folds = list(skf.split(X, y))

        for fold_idx, (tr, va) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_va = scaler.transform(X[va])
            y_tr, y_va = y[tr], y[va]

            for family, variant, spec in variants:
                for seed in seeds:
                    d, json_path, npz_path = unit_paths(args.output, dataset_key,
                                                        variant, seed, fold_idx)
                    if os.path.exists(json_path):
                        continue            # counted in done_units already
                    os.makedirs(d, exist_ok=True)
                    uid = f"{dataset_key}/{variant}/seed{seed}/fold{fold_idx}"
                    log(f"[{done_units + 1}/{total_units}] {uid}")

                    t0 = time.time()
                    try:
                        def hb(k, n, el, u=uid):
                            # k epochs of n done in el seconds (resampling excluded)
                            left = el / k * (n - k) if k else 0
                            log(f"    {u}: epoch {k}/{n} ({100 * k / n:.0f}% of this "
                                f"unit, ~{_hms(left)} left)")
                        hb = hb if family == "mlp" else None
                        if family == "mlp":
                            y_pred, proba, info = run_mlp(spec, X_tr, y_tr, X_va, n_classes,
                                                          seed, heartbeat=hb)
                        elif family in ("xgb", "lgbm"):
                            y_pred, proba, info = run_tree(family, spec, X_tr, y_tr, X_va,
                                                           n_classes, seed)
                        else:
                            y_pred, proba, info = run_tabnet(spec, X_tr, y_tr, X_va,
                                                             n_classes, seed)
                    except Exception as e:
                        import traceback
                        log(f"    FAILED {uid}: {e}\n{traceback.format_exc()}")
                        failed.append(uid)
                        continue
                    secs = time.time() - t0

                    m = compute_metrics(y_va, y_pred, n_classes, class_names)
                    m.update({
                        "dataset": dataset_key, "variant": variant, "family": family,
                        "technique_or_mode": spec, "seed": seed, "fold": fold_idx,
                        "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                        "n_classes": n_classes, "class_names": class_names,
                        "train_seconds": round(secs, 2), "device": DEVICE,
                        **info,
                    })

                    # npz first, json last: the json is the "done" marker.
                    arrays = {"y_true": y_va.astype(np.int16),
                              "y_pred": y_pred.astype(np.int16),
                              "val_index": va.astype(np.int32)}
                    if not args.no_proba:
                        arrays["proba"] = proba.astype(np.float16)
                    np.savez_compressed(npz_path, **arrays)
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(m, f, indent=1)

                    done_units += 1
                    eta.record((dataset_key, variant), secs)
                    log(f"    macro-F1={m['macro_f1']:.4f}  bal-acc={m['balanced_accuracy']:.4f}"
                        f"  {secs:.0f}s")
                    log("    " + eta.line(done_units, total_units))

    log(f"finished: {done_units}/{total_units} units present  "
        f"(this session ran for {_hms(time.time() - eta.t_start)})")
    if failed:
        log(f"{len(failed)} unit(s) FAILED and are not saved; re-run the same command to "
            f"retry them, and see progress.log for each traceback:\n    " + "\n    ".join(failed))
    summarize(args.output, log)


# ======================================================================================
# Summaries (safe to re-run any time; reads only the json files)
# ======================================================================================
def summarize(output, log=print):
    rows, cls_rows, cms = [], [], {}
    for root, _, files in os.walk(output):
        for fn in files:
            if not (fn.startswith("seed") and fn.endswith(".json")):
                continue
            with open(os.path.join(root, fn), encoding="utf-8") as f:
                m = json.load(f)
            base = {k: m[k] for k in ["dataset", "variant", "family", "technique_or_mode",
                                      "seed", "fold", "n_train", "n_train_after_resampling",
                                      "n_val", "train_seconds", "device"]}
            rows.append({**base, **{k: m[k] for k in [
                "accuracy", "balanced_accuracy", "macro_f1", "macro_precision",
                "macro_recall", "weighted_f1", "n_classes_predicted"]}})
            for pc in m["per_class"]:
                cls_rows.append({**base, **pc})
            cms[f"{m['dataset']}/{m['variant']}/seed{m['seed']}/fold{m['fold']}"] = {
                "class_names": m["class_names"], "confusion_matrix": m["confusion_matrix"]}

    if not rows:
        log("summarize: no finished units found")
        return
    df = pd.DataFrame(rows).sort_values(["dataset", "variant", "seed", "fold"])
    df.to_csv(os.path.join(output, "rerun_metrics.csv"), index=False)
    pd.DataFrame(cls_rows).sort_values(["dataset", "variant", "class_index", "seed", "fold"]) \
        .to_csv(os.path.join(output, "rerun_per_class.csv"), index=False)
    with open(os.path.join(output, "rerun_confusion_matrices.json"), "w",
              encoding="utf-8") as f:
        json.dump(cms, f)

    agg = df.groupby(["dataset", "variant"]).macro_f1.agg(["count", "mean", "std"])
    log("summarize: macro-F1 by (dataset, variant)\n" + agg.round(4).to_string())
    log(f"wrote rerun_metrics.csv ({len(df)} units), rerun_per_class.csv "
        f"({len(cls_rows)} rows), rerun_confusion_matrices.json")


def status(output):
    n = 0
    for root, _, files in os.walk(output):
        n += sum(1 for f in files if f.startswith("seed") and f.endswith(".json"))
    print(f"{n} finished units under {output}")
    p = os.path.join(output, "progress.log")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            print("".join(f.readlines()[-8:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(DATASET_PATHS),
                    choices=list(DATASET_PATHS))
    ap.add_argument("--models", nargs="+", default=["mlp", "xgb", "lgbm"],
                    choices=["mlp", "xgb", "lgbm", "tabnet"])
    ap.add_argument("--seeds", type=int, default=3, help="seeds 42, 43, ... per fold")
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--output", default="rerun_results")
    ap.add_argument("--no_proba", action="store_true",
                    help="do not save predicted probabilities (saves disk; loses calibration)")
    ap.add_argument("--synthetic", action="store_true",
                    help="smoke test on a built-in synthetic dataset; needs no data files")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.status:
        status(args.output)
    elif args.summarize:
        summarize(args.output)
    else:
        run(args)


if __name__ == "__main__":
    main()
