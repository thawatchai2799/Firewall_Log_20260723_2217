"""
tree_baselines.py

Non-neural (gradient-boosted decision tree) baselines for the imbalance-handling comparison.
They evaluate how the MLP compares against XGBoost and LightGBM, which are usually the strongest
models on tabular data.

Design goal: DIRECT comparability with the four MLP-based techniques already reported
(baseline_smote, cost_sensitive, focal_loss, hybrid_resample) in pipeline.py. To achieve this,
this module reuses EXACTLY the same experimental scaffolding as pipeline.run_comparison():

  * same data loading / imputation path (data_loader.load_dataset + median impute)
  * same StratifiedKFold(n_splits=5, shuffle=True, random_state=42) fold splits
  * same per-fold StandardScaler fitted on the training fold only (no leakage)
  * same capped SMOTE resampling (imbalance_methods._safe_smote) for the resampling variants
  * same metrics: macro-F1 and per-class F1 via sklearn, identical call signatures
  * same pairwise Wilcoxon signed-rank test between methods

Because a boosted tree does not train through a PyTorch loss function, the "technique" axis is
realised differently for trees than for the MLP:

  * xgb_plain / lgbm_plain          : tree trained on the raw (scaled) imbalanced data, no
                                       resampling and no class weighting -> the honest floor.
  * xgb_smote / lgbm_smote          : tree trained on capped-SMOTE-resampled data (the tree
                                       analogue of the MLP "baseline_smote" technique).
  * xgb_classweight / lgbm_classwt  : tree trained with per-sample weights inversely proportional
                                       to class frequency (the tree analogue of "cost_sensitive").

This lets this study report, on the very same folds, whether the ranking "resampling > loss-based"
found for the MLP also holds for the strongest tabular baselines, and whether any tree model finally
moves the security-critical minority classes that every MLP technique left essentially unclassifiable.

Run (mirrors pipeline.py CLI):
    python tree_baselines.py --dataset ertam_kaya --path data/raw/ertam_kaya_firewall.csv \
        --models xgb lgbm --folds 5 --output results/

Dependencies: xgboost, lightgbm (add to requirements). Both are CPU-friendly; GPU optional.
"""

import argparse
import json
import os
import time
import warnings

import numpy as np
from scipy.stats import wilcoxon
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
        "This script must sit in the SAME folder as the existing pipeline files\n"
        "(data_loader.py, imbalance_methods.py, model.py), because it reuses their\n"
        "data loading and capped-SMOTE code.\n\n"
        "Fix: copy tree_baselines.py (and make_tree_vs_mlp_table.py) into the folder\n"
        "that contains pipeline.py, then run it from that folder, e.g.:\n\n"
        "    cd C:\\path\\to\\your\\pipeline\\folder\n"
        "    python tree_baselines.py --dataset ertam_kaya --path <your csv> "
        "--models xgb lgbm --folds 5 --output results\n"
    )

# Optional imports guarded so the module still loads (and reports a clear message) if one
# library is missing; the other can still run.
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False


# ----------------------------------------------------------------------------------------------
# Model factories. Hyperparameters are deliberately modest, fixed, and identical across datasets,
# mirroring the MLP protocol (a single fixed architecture, not per-dataset tuned) so that the
# comparison isolates model family rather than tuning budget. random_state=42 throughout.
# ----------------------------------------------------------------------------------------------
def _make_xgb(n_classes: int, random_state: int = 42):  # n_classes kept for a uniform factory signature; inferred in fit()
    # NOTE: objective and num_class are intentionally NOT set. The scikit-learn wrapper infers
    # them from the labels seen in fit() (binary:logistic for 2 classes, multi:softprob for >2),
    # which is the version-stable, portable path. Passing num_class explicitly alongside the
    # sklearn API is redundant and has triggered conflicts/warnings in some XGBoost versions,
    # so we leave it out to keep the script robust across environments (incl. Windows installs).
    return XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",       # fast on large real datasets (CIC-IDS2017 / UNSW-NB15)
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=random_state,
        verbosity=0,
    )


def _make_lgbm(n_classes: int, random_state: int = 42):  # n_classes kept for a uniform factory signature; inferred in fit()
    # As with XGBoost above, objective and num_class are left unset so the sklearn wrapper infers
    # them from the labels in fit(). subsample<1.0 requires bagging_freq>0 in LightGBM, so we set
    # it explicitly; otherwise LightGBM silently ignores subsample and prints a warning.
    return LGBMClassifier(
        n_estimators=400,
        max_depth=-1,
        num_leaves=63,
        learning_rate=0.1,
        subsample=0.9,
        subsample_freq=1,        # required for subsample<1.0 to actually take effect in LightGBM
        colsample_bytree=0.9,
        n_jobs=-1,
        random_state=random_state,
        verbose=-1,
    )


def _build_variants(model_key: str, n_classes: int, random_state: int = 42):
    """Return an ordered dict {variant_name: (factory_fn, mode)} for one model family.

    mode is one of: 'plain' (raw imbalanced), 'smote' (capped SMOTE), 'classweight'
    (per-sample inverse-frequency weights). These three parallel the MLP techniques so the
    resulting names line up conceptually with baseline_smote / cost_sensitive in the MLP table.
    """
    if model_key == "xgb":
        mk = lambda: _make_xgb(n_classes, random_state)
        return {
            "xgb_plain": (mk, "plain"),
            "xgb_smote": (mk, "smote"),
            "xgb_classweight": (mk, "classweight"),
        }
    elif model_key == "lgbm":
        mk = lambda: _make_lgbm(n_classes, random_state)
        return {
            "lgbm_plain": (mk, "plain"),
            "lgbm_smote": (mk, "smote"),
            "lgbm_classweight": (mk, "classweight"),
        }
    else:
        raise ValueError(f"unknown model_key={model_key} (expected 'xgb' or 'lgbm')")


def _fit_predict(factory, mode, X_train_scaled, y_train, X_val_scaled, n_classes,
                 random_state: int = 42):
    """Train one tree variant on one fold and return validation predictions.

    Uses the same capped SMOTE as the MLP baseline for the 'smote' mode, and sklearn's
    inverse-frequency sample weights for 'classweight'. 'plain' trains on the raw scaled data.
    """
    model = factory()

    if mode == "smote":
        X_fit, y_fit = _safe_smote(X_train_scaled, y_train, random_state=random_state)
        sample_weight = None
    elif mode == "classweight":
        X_fit, y_fit = X_train_scaled, y_train
        # 'balanced' => weight_c = n_samples / (n_classes * count_c): the same inverse-frequency
        # principle the MLP cost_sensitive technique applies inside the loss, expressed here as
        # per-row weights the tree can consume directly.
        sample_weight = compute_sample_weight(class_weight="balanced", y=y_fit)
    elif mode == "plain":
        X_fit, y_fit = X_train_scaled, y_train
        sample_weight = None
    else:
        raise ValueError(f"unknown mode={mode}")

    if sample_weight is not None:
        model.fit(X_fit, y_fit, sample_weight=sample_weight)
    else:
        model.fit(X_fit, y_fit)

    return model.predict(X_val_scaled)


def run_tree_comparison(dataset_key: str, path: str, model_keys, n_folds: int = 5,
                        output_dir: str = "results", feature_names_csv=None,
                        random_state: int = 42):
    """Mirror of pipeline.run_comparison() but for gradient-boosted tree baselines.

    Produces the same JSON shape (summary + pairwise_wilcoxon_tests) so downstream table-building
    code can consume tree results exactly like the MLP results.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[1/5] Loading dataset: {dataset_key}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)

    dq = report_data_quality(X_df, name=dataset_key)
    dq.to_csv(os.path.join(output_dir, f"{dataset_key}_tree_data_quality.csv"))

    # identical imputation to pipeline.run_comparison (median then 0 for all-NaN columns).
    # The inf->NaN replace mirrors data_loader's own cleaning; it is a no-op on already-clean
    # data but protects against a stray Inf (CIC-IDS2017 is known to contain Inf in rate
    # columns) crashing StandardScaler/SMOTE deep into a long run.
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()

    print(f"[2/5] number of classes = {n_classes}: {class_names}")

    X = X_df.values

    # Assemble the full list of variants across the requested model families.
    variants = {}
    for mk in model_keys:
        if mk == "xgb" and not _HAS_XGB:
            print("[warn] xgboost not installed - skipping xgb variants. `pip install xgboost`.")
            continue
        if mk == "lgbm" and not _HAS_LGBM:
            print("[warn] lightgbm not installed - skipping lgbm variants. `pip install lightgbm`.")
            continue
        variants.update(_build_variants(mk, n_classes, random_state))

    if not variants:
        raise RuntimeError("no tree models available to run (install xgboost and/or lightgbm)")

    variant_names = list(variants)

    # SAME fold object as the MLP pipeline: n_splits=5, shuffle=True, random_state=42.
    # This is what makes the tree macro-F1 values paired with the MLP ones fold-for-fold.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    all_results = {v: {"per_fold_macro_f1": [], "per_fold_per_class_f1": []} for v in variant_names}

    print(f"[3/5] {n_folds}-fold CV for {len(variant_names)} tree variants: {variant_names}")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # per-fold scaler fitted on train only (kept for parity with the MLP path; trees are
        # scale-invariant so this does not change tree results, but it guarantees identical inputs)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        for v_name in variant_names:
            factory, mode = variants[v_name]
            t0 = time.time()
            val_pred = _fit_predict(factory, mode, X_train_scaled, y_train, X_val_scaled,
                                    n_classes, random_state=random_state)
            elapsed = time.time() - t0

            macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                y_val, val_pred, labels=range(n_classes), zero_division=0)

            all_results[v_name]["per_fold_macro_f1"].append(macro_f1)
            all_results[v_name]["per_fold_per_class_f1"].append(per_class_f1.tolist())

            print(f"  fold {fold_idx+1}/{n_folds} | {v_name:18s} | macro-F1={macro_f1:.4f} "
                  f"| {elapsed:5.1f}s")

    print("[4/5] summarising + pairwise Wilcoxon signed-rank")
    summary = {}
    for v_name in variant_names:
        f1s = all_results[v_name]["per_fold_macro_f1"]
        summary[v_name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "per_fold_macro_f1": [float(x) for x in f1s],
            "mean_per_class_f1": np.mean(
                all_results[v_name]["per_fold_per_class_f1"], axis=0).tolist(),
            "class_names": class_names,
        }

    pairwise_tests = {}
    for i, a_name in enumerate(variant_names):
        for b_name in variant_names[i + 1:]:
            a = all_results[a_name]["per_fold_macro_f1"]
            b = all_results[b_name]["per_fold_macro_f1"]
            try:
                # Identical or near-identical fold scores make scipy emit a harmless
                # "invalid value encountered in scalar divide" RuntimeWarning (division by a
                # zero standard error). Suppress it locally so a tie between two variants does
                # not look like a bug on the console; the resulting p-value is handled below.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stat, p_value = wilcoxon(a, b)
                stat, p_value = float(stat), float(p_value)
            except ValueError:  # identical vectors -> Wilcoxon undefined (raised by some scipy versions)
                stat, p_value = None, None
            # Non-finite values (nan/inf) are not valid strict JSON ("NaN" literal breaks jq,
            # JavaScript, and other consumers), so map them to null instead.
            if stat is not None and not np.isfinite(stat):
                stat = None
            if p_value is not None and not np.isfinite(p_value):
                p_value = None
            pairwise_tests[f"{a_name}_vs_{b_name}"] = {
                "statistic": stat, "p_value": p_value}

    output = {"summary": summary, "pairwise_wilcoxon_tests": pairwise_tests}
    out_path = os.path.join(output_dir, f"{dataset_key}_tree_baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"[5/5] saved -> {out_path}")
    # concise stdout table for quick reading
    print("\n=== macro-F1 (mean +/- std) ===")
    for v_name in variant_names:
        s = summary[v_name]
        print(f"  {v_name:18s} {s['mean_macro_f1']:.4f} +/- {s['std_macro_f1']:.4f}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ertam_kaya", "cicids2017", "unsw_nb15"])
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    parser.add_argument("--models", nargs="+", default=["xgb", "lgbm"], choices=["xgb", "lgbm"])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    run_tree_comparison(
        dataset_key=args.dataset,
        path=args.path,
        model_keys=args.models,
        n_folds=args.folds,
        output_dir=args.output,
        feature_names_csv=args.feature_names_csv,
    )
