"""
tune_xgb_datasetA.py

Lightly tuned XGBoost on Dataset A (Ertam & Kaya). It checks whether hyperparameter tuning
changes the tree-versus-MLP picture relative to the fixed-configuration baselines.

Methodology: NESTED cross-validation so that tuning never sees the outer test fold.
  * Outer loop: the SAME StratifiedKFold(n_splits=5, shuffle=True, random_state=42) folds used by
    pipeline.py and tree_baselines.py, so results pair fold-for-fold with the other experiments.
  * Inner loop: on each outer training fold only, RandomizedSearchCV (n_iter=25, 3-fold stratified,
    scoring='f1_macro') selects hyperparameters; the selected model is then refit on the full outer
    training fold and evaluated once on the outer test fold.
  * Mode: inverse-frequency class weighting (the best XGBoost variant on Dataset A in the untuned
    comparison), plus the fixed-configuration result recomputed in the same run for a paired
    tuned-vs-fixed comparison with a Wilcoxon signed-rank test.

Runtime estimate on Dataset A (65,532 rows): each fit took roughly 4-6 s in the untuned run, and the
search performs 25 candidates x 3 inner folds + 1 refit per outer fold (~380 fits total), so expect
roughly 25-45 minutes on a multi-core CPU. All cores are used (n_jobs=-1).

Run from the pipeline folder (next to data_loader.py):
    python tune_xgb_datasetA.py --path "C:\\path\\to\\log2.csv" --output results

Output: results/ertam_kaya_xgb_tuned_results.json  (same summary shape as the other result files,
plus the chosen hyperparameters per fold).
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
from scipy.stats import wilcoxon, loguniform
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

try:
    from data_loader import load_dataset
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "Run this script from the folder that contains pipeline.py / data_loader.py.\n")

try:
    from xgboost import XGBClassifier
except ModuleNotFoundError:
    raise SystemExit("[setup error] xgboost is not installed. Run: pip install xgboost")


SEARCH_SPACE = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": loguniform(0.03, 0.3),
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5, 10],
    "gamma": [0, 0.1, 0.5, 1.0],
}

FIXED_PARAMS = {  # the exact fixed configuration used in tree_baselines.py, for the paired baseline
    "n_estimators": 400, "max_depth": 6, "learning_rate": 0.1,
    "subsample": 0.9, "colsample_bytree": 0.9,
}


def make_xgb(random_state=42, **params):
    return XGBClassifier(tree_method="hist", eval_metric="mlogloss", n_jobs=-1,
                         random_state=random_state, verbosity=0, **params)


def run(path, output_dir="results", n_iter=25, random_state=42):
    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Loading Dataset A (ertam_kaya)")
    X_df, y_raw, _ = load_dataset("ertam_kaya", path)
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    class_names = le.classes_.tolist()
    X = X_df.values
    print(f"[2/4] classes = {class_names}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    res = {"tuned": {"per_fold_macro_f1": [], "per_fold_per_class_f1": [], "chosen_params": []},
           "fixed": {"per_fold_macro_f1": [], "per_fold_per_class_f1": []}}

    print(f"[3/4] nested CV: outer 5-fold (random_state=42), inner RandomizedSearchCV "
          f"(n_iter={n_iter}, 3-fold, f1_macro), class-weighted XGBoost")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]
        sw = compute_sample_weight(class_weight="balanced", y=ytr)

        # --- fixed configuration (paired baseline, same as tree_baselines.py) ---
        t0 = time.time()
        fixed = make_xgb(random_state=random_state, **FIXED_PARAMS)
        fixed.fit(Xtr, ytr, sample_weight=sw)
        pred = fixed.predict(Xte)
        mf = f1_score(yte, pred, average="macro", zero_division=0)
        _, _, pcf, _ = precision_recall_fscore_support(yte, pred, labels=range(n_classes), zero_division=0)
        res["fixed"]["per_fold_macro_f1"].append(float(mf))
        res["fixed"]["per_fold_per_class_f1"].append(pcf.tolist())
        print(f"  fold {fold_idx+1}/5 | fixed  | macro-F1={mf:.4f} | {time.time()-t0:5.1f}s")

        # --- randomized search on the outer TRAINING fold only ---
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search = RandomizedSearchCV(
                make_xgb(random_state=random_state),
                SEARCH_SPACE, n_iter=n_iter, scoring="f1_macro",
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
                random_state=random_state, n_jobs=1, refit=True, verbose=0)
            # n_jobs=1 at search level because each XGBClassifier already uses all cores (n_jobs=-1);
            # nesting parallelism would oversubscribe the CPU and can slow the search down.
            search.fit(Xtr, ytr, sample_weight=sw)
        best = search.best_estimator_
        pred = best.predict(Xte)
        mf = f1_score(yte, pred, average="macro", zero_division=0)
        _, _, pcf, _ = precision_recall_fscore_support(yte, pred, labels=range(n_classes), zero_division=0)
        def _clean(v):
            if isinstance(v, (np.integer,)): return int(v)
            if isinstance(v, (float, np.floating)):
                return int(v) if float(v).is_integer() else float(v)
            return v
        chosen = {k: _clean(v) for k, v in search.best_params_.items()}
        res["tuned"]["per_fold_macro_f1"].append(float(mf))
        res["tuned"]["per_fold_per_class_f1"].append(pcf.tolist())
        res["tuned"]["chosen_params"].append(chosen)
        print(f"  fold {fold_idx+1}/5 | tuned  | macro-F1={mf:.4f} | {time.time()-t0:5.1f}s "
              f"| params={chosen}")

    print("[4/4] summarising")
    summary = {}
    for name in ("fixed", "tuned"):
        f1s = res[name]["per_fold_macro_f1"]
        summary[name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "per_fold_macro_f1": f1s,
            "mean_per_class_f1": np.mean(res[name]["per_fold_per_class_f1"], axis=0).tolist(),
            "class_names": class_names,
        }
    summary["tuned"]["chosen_params_per_fold"] = res["tuned"]["chosen_params"]

    a, b = res["tuned"]["per_fold_macro_f1"], res["fixed"]["per_fold_macro_f1"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stat, pval = wilcoxon(a, b)
        stat, pval = float(stat), float(pval)
        if not np.isfinite(stat): stat = None
        if not np.isfinite(pval): pval = None
    except ValueError:
        stat, pval = None, None

    diffs = [x - y for x, y in zip(a, b)]
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
    cohens_d = float(np.mean(diffs) / sd) if sd > 0 else None

    out = {"summary": summary,
           "tuned_vs_fixed": {"wilcoxon_statistic": stat, "wilcoxon_p": pval,
                              "mean_gain_macro_f1": float(np.mean(diffs)),
                              "paired_cohens_d": cohens_d}}
    out_path = os.path.join(output_dir, "ertam_kaya_xgb_tuned_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"\nsaved -> {out_path}")
    print(f"fixed : {summary['fixed']['mean_macro_f1']:.4f} +/- {summary['fixed']['std_macro_f1']:.4f}")
    print(f"tuned : {summary['tuned']['mean_macro_f1']:.4f} +/- {summary['tuned']['std_macro_f1']:.4f}")
    print(f"gain  : {np.mean(diffs):+.4f} (Wilcoxon p={pval})")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="Path to the Ertam & Kaya CSV (log2.csv)")
    ap.add_argument("--output", default="results")
    ap.add_argument("--n_iter", type=int, default=25)
    args = ap.parse_args()
    run(args.path, args.output, args.n_iter)
