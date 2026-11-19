"""
lgbm_investigation.py

Follow-up experiment probing WHY LightGBM collapses on the two multi-million-row datasets
(CIC-IDS2017, UNSW-NB15) while XGBoost stays stable. It tests whether the collapse is a fixable
hyperparameter artifact or an intrinsic failure of leaf-wise growth under extreme class imbalance.

The hypothesis is that LightGBM's leaf-wise tree growth, combined with its default minimum-samples
constraints, lets the model satisfy its objective by collapsing onto the majority class on these
heavily imbalanced datasets. We test this directly by sweeping the parameters most relevant to that
mechanism, one axis at a time, from a fixed base configuration:

  * min_child_samples : default 20 -> also try 5, 50, 200 (controls how eagerly leaves specialise)
  * num_leaves        : default 63 -> also try 15, 31, 127
  * min_split_gain    : default 0.0 -> also try 0.01, 0.1 (regularises splitting)
  * is_unbalance      : default off -> also try on (LightGBM's built-in imbalance handling)
  * class weighting   : the "smote" and "classweight" modes from tree_baselines, for reference

Everything else matches the pipeline exactly: same StratifiedKFold(random_state=42) folds, same
per-fold StandardScaler, same capped SMOTE (for the smote reference), same macro-F1 / per-class-F1.
The output records, for every configuration, whether the model escaped majority-collapse - defined
here as reaching macro-F1 above a small threshold AND assigning non-zero F1 to more than one class.

Run from the pipeline folder (next to data_loader.py). CIC-IDS2017 is the default because its
collapse is the clearest; it is also the slowest (millions of rows x many configs), so a --folds 3
quick pass and a single dataset are recommended first:

    python lgbm_investigation.py --dataset cicids2017 --path "C:\\path\\to\\MachineLearningCVE" --folds 3 --output results

Dataset A (small) finishes in minutes and is a good smoke test:

    python lgbm_investigation.py --dataset ertam_kaya --path "C:\\path\\to\\log2.csv" --folds 5 --output results

Dependency: lightgbm (already installed for tree_baselines.py).
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

try:
    from data_loader import load_dataset, report_data_quality
    from imbalance_methods import _safe_smote
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "Run this script from the folder that contains pipeline.py / data_loader.py.\n")

try:
    from lightgbm import LGBMClassifier
except ModuleNotFoundError:
    raise SystemExit("[setup error] lightgbm is not installed. Run: pip install lightgbm")


# Base configuration = the exact fixed config from tree_baselines.py (so "base" reproduces the
# collapse already reported). Each swept configuration changes ONE axis from this base.
BASE = dict(n_estimators=400, max_depth=-1, num_leaves=63, learning_rate=0.1,
            subsample=0.9, subsample_freq=1, colsample_bytree=0.9,
            n_jobs=-1, random_state=42, verbose=-1)


def _configs():
    """Yield (config_name, extra_params, mode) tuples. mode in {plain, smote, classweight}.

    'extra_params' overrides BASE. 'mode' controls resampling/weighting exactly as in
    tree_baselines.py so the reference points line up with the numbers already reported.
    """
    yield ("base_plain", {}, "plain")                                  # reproduces the reported collapse
    # min_child_samples sweep (leaf specialisation)
    yield ("min_child_5", {"min_child_samples": 5}, "plain")
    yield ("min_child_50", {"min_child_samples": 50}, "plain")
    yield ("min_child_200", {"min_child_samples": 200}, "plain")
    # num_leaves sweep (tree complexity)
    yield ("num_leaves_15", {"num_leaves": 15}, "plain")
    yield ("num_leaves_31", {"num_leaves": 31}, "plain")
    yield ("num_leaves_127", {"num_leaves": 127}, "plain")
    # min_split_gain sweep (split regularisation)
    yield ("min_split_gain_0p01", {"min_split_gain": 0.01}, "plain")
    yield ("min_split_gain_0p1", {"min_split_gain": 0.1}, "plain")
    # built-in imbalance handling
    yield ("is_unbalance", {"is_unbalance": True}, "plain")
    # imbalance-handling references (same as tree_baselines LightGBM variants)
    yield ("ref_smote", {}, "smote")
    yield ("ref_classweight", {}, "classweight")


def _fit_predict(extra, mode, Xtr, ytr, Xte, random_state=42):
    params = dict(BASE); params.update(extra)
    model = LGBMClassifier(**params)
    if mode == "smote":
        Xf, yf = _safe_smote(Xtr, ytr, random_state=random_state); sw = None
    elif mode == "classweight":
        Xf, yf = Xtr, ytr; sw = compute_sample_weight(class_weight="balanced", y=yf)
    else:
        Xf, yf = Xtr, ytr; sw = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(Xf, yf, sample_weight=sw) if sw is not None else model.fit(Xf, yf)
        return model.predict(Xte)


def run(dataset_key, path, n_folds=5, output_dir="results", feature_names_csv=None,
        collapse_threshold=0.30, random_state=42):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/4] Loading dataset: {dataset_key}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)
    report_data_quality(X_df, name=dataset_key).to_csv(
        os.path.join(output_dir, f"{dataset_key}_lgbm_investigation_data_quality.csv"))
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    le = LabelEncoder(); y = le.fit_transform(y_raw)
    n_classes = len(le.classes_); class_names = le.classes_.tolist()
    X = X_df.values
    print(f"[2/4] classes = {n_classes}")

    configs = list(_configs())
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    results = {name: {"per_fold_macro_f1": [], "per_fold_per_class_f1": []} for name, _, _ in configs}

    print(f"[3/4] {n_folds}-fold CV for {len(configs)} LightGBM configurations")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        for name, extra, mode in configs:
            t0 = time.time()
            pred = _fit_predict(extra, mode, Xtr, ytr, Xte, random_state=random_state)
            dt = time.time() - t0
            mf = f1_score(yte, pred, average="macro", zero_division=0)
            _, _, pcf, _ = precision_recall_fscore_support(yte, pred, labels=range(n_classes), zero_division=0)
            results[name]["per_fold_macro_f1"].append(mf)
            results[name]["per_fold_per_class_f1"].append(pcf.tolist())
            print(f"  fold {fold_idx+1}/{n_folds} | {name:22s} | macro-F1={mf:.4f} | {dt:6.1f}s")

    print("[4/4] summarising")
    summary = {}
    for name, _, _ in configs:
        f1s = results[name]["per_fold_macro_f1"]
        mean_pcf = np.mean(results[name]["per_fold_per_class_f1"], axis=0)
        n_nonzero = int(np.sum(mean_pcf > 0.01))
        escaped = bool(np.mean(f1s) > collapse_threshold and n_nonzero > 1)
        summary[name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "per_fold_macro_f1": [float(x) for x in f1s],
            "mean_per_class_f1": mean_pcf.tolist(),
            "n_classes_with_nonzero_f1": n_nonzero,
            "escaped_majority_collapse": escaped,
            "class_names": class_names,
        }

    out = {"dataset": dataset_key, "collapse_threshold": collapse_threshold, "summary": summary}
    out_path = os.path.join(output_dir, f"{dataset_key}_lgbm_investigation_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {out_path}")

    print("\n=== macro-F1 (mean +/- std) | classes-with-signal | escaped-collapse? ===")
    for name, _, _ in configs:
        s = summary[name]
        flag = "YES" if s["escaped_majority_collapse"] else "no"
        print(f"  {name:22s} {s['mean_macro_f1']:.4f} +/- {s['std_macro_f1']:.4f} "
              f"| {s['n_classes_with_nonzero_f1']:2d}/{n_classes} | escaped={flag}")
    escaped_any = [n for n, _, _ in configs if summary[n]["escaped_majority_collapse"]]
    print(f"\nConfigurations that escaped collapse: {escaped_any if escaped_any else 'NONE'}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ertam_kaya", "cicids2017", "unsw_nb15"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--feature_names_csv", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", default="results")
    ap.add_argument("--collapse_threshold", type=float, default=0.30)
    args = ap.parse_args()
    run(args.dataset, args.path, n_folds=args.folds, output_dir=args.output,
        feature_names_csv=args.feature_names_csv, collapse_threshold=args.collapse_threshold)
