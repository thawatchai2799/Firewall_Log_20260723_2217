"""
tabnet_baseline.py

Deep tabular baseline (TabNet) for the model-family comparison. It evaluates how a modern
attention-based tabular network compares against the MLP and the gradient-boosted tree baselines.

Design goal is identical to tree_baselines.py: DIRECT comparability with the MLP techniques and the
gradient-boosted baselines. This script reuses exactly the same experimental
scaffolding as pipeline.run_comparison() and tree_baselines.run_tree_comparison():

  * same data loading / imputation path (data_loader.load_dataset + inf->NaN + median impute)
  * same StratifiedKFold(n_splits=5, shuffle=True, random_state=42) fold splits (use --folds to change)
  * same per-fold StandardScaler fitted on the training fold only (no leakage)
  * same capped SMOTE resampling (imbalance_methods._safe_smote) for the resampling variant
  * same metrics: macro-F1 and per-class F1, identical call signatures
  * same pairwise Wilcoxon signed-rank test and the same result-JSON shape

TabNet is a supervised attention-based network for tabular data (Arik & Pfister, 2021). Because it
trains through its own fit()/predict() API rather than a PyTorch loss we pass in, the "technique"
axis is realised in three modes paralleling the other families:

  * tabnet_plain       : trained on the raw (scaled) imbalanced data, no resampling / no weighting.
  * tabnet_smote       : trained on capped-SMOTE-resampled data (the resampling analogue).
  * tabnet_classweight : trained with per-class weights inversely proportional to frequency,
                         passed via TabNet's `weights` argument.

Run from the pipeline folder (next to data_loader.py):
    python tabnet_baseline.py --dataset ertam_kaya --path "C:\\path\\to\\log2.csv" --folds 5 --output results

Dependencies: pytorch-tabnet (pip install pytorch-tabnet), which pulls in torch. GPU is used
automatically if available; otherwise CPU. Dataset A finishes in minutes; Datasets B and C are
substantially slower (deep network over millions of rows) and are optional.
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

try:
    from data_loader import load_dataset, report_data_quality
    from imbalance_methods import _safe_smote
except ModuleNotFoundError as e:
    raise SystemExit(
        f"\n[setup error] Could not import '{e.name}'.\n"
        "Run this script from the folder that contains pipeline.py / data_loader.py "
        "(the same folder where tree_baselines.py runs).\n")

try:
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    _HAS_TABNET = True
except Exception:  # pragma: no cover
    _HAS_TABNET = False


def _make_tabnet(random_state: int = 42):
    """Fixed, modest TabNet configuration, identical across datasets.

    As with the MLP and the trees, hyperparameters are fixed rather than tuned so the comparison
    isolates the model family, not the tuning budget. n_d=n_a=16, n_steps=4 is a small,
    widely-used default that trains quickly and avoids overfitting the tiny minority classes.
    """
    device = "cuda" if (torch.cuda.is_available()) else "cpu"
    return TabNetClassifier(
        n_d=16, n_a=16, n_steps=4, gamma=1.5,
        n_independent=2, n_shared=2,
        seed=random_state,
        verbose=0,
        device_name=device,
        optimizer_params=dict(lr=2e-2),
    )


# The three modes parallel tree_baselines / the MLP techniques.
_MODES = ("plain", "smote", "classweight")


def _fit_predict(mode, X_train_scaled, y_train, X_val_scaled, n_classes, random_state=42):
    """Train one TabNet variant on one fold and return validation predictions.

    max_epochs is kept modest with early stopping on a small internal validation slice so runtime
    stays reasonable; the eval split is carved from the training fold only (never the outer val fold).
    """
    if mode == "smote":
        X_fit, y_fit = _safe_smote(X_train_scaled, y_train, random_state=random_state)
        weights = 0
    elif mode == "classweight":
        X_fit, y_fit = X_train_scaled, y_train
        weights = 1  # TabNet: weights=1 enables inverse-frequency class weighting internally
    elif mode == "plain":
        X_fit, y_fit = X_train_scaled, y_train
        weights = 0
    else:
        raise ValueError(f"unknown mode={mode}")

    clf = _make_tabnet(random_state=random_state)
    # Carve a small internal validation slice from the TRAINING data only, for early stopping.
    # This never touches the outer validation fold. Stratified so rare classes appear in both parts.
    from sklearn.model_selection import train_test_split
    try:
        X_in, X_es, y_in, y_es = train_test_split(
            X_fit, y_fit, test_size=0.1, random_state=random_state, stratify=y_fit)
        eval_set = [(X_es.astype(np.float32), y_es)]
    except ValueError:
        # a class too small to stratify-split -> skip early stopping, train the full max_epochs
        X_in, y_in, eval_set = X_fit, y_fit, None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(
            X_in.astype(np.float32), y_in,
            eval_set=eval_set,
            max_epochs=100, patience=15, batch_size=1024, virtual_batch_size=256,
            weights=weights, drop_last=False,
        )
        pred = clf.predict(X_val_scaled.astype(np.float32))
    return pred


def run_tabnet_comparison(dataset_key, path, n_folds=5, output_dir="results",
                          feature_names_csv=None, random_state=42):
    if not _HAS_TABNET:
        raise SystemExit("[setup error] pytorch-tabnet is not installed. Run: pip install pytorch-tabnet")

    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/5] Loading dataset: {dataset_key}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)

    dq = report_data_quality(X_df, name=dataset_key)
    dq.to_csv(os.path.join(output_dir, f"{dataset_key}_tabnet_data_quality.csv"))

    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)
    class_names = le.classes_.tolist()
    X = X_df.values
    print(f"[2/5] number of classes = {n_classes}: {class_names}")

    variant_names = [f"tabnet_{m}" for m in _MODES]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    all_results = {v: {"per_fold_macro_f1": [], "per_fold_per_class_f1": []} for v in variant_names}

    print(f"[3/5] {n_folds}-fold CV for {len(variant_names)} TabNet variants: {variant_names}")
    for fold_idx, (tr, te) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr]); Xte = scaler.transform(X[te])
        ytr, yte = y[tr], y[te]

        for mode in _MODES:
            v_name = f"tabnet_{mode}"
            t0 = time.time()
            pred = _fit_predict(mode, Xtr, ytr, Xte, n_classes, random_state=random_state)
            elapsed = time.time() - t0
            macro_f1 = f1_score(yte, pred, average="macro", zero_division=0)
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                yte, pred, labels=range(n_classes), zero_division=0)
            all_results[v_name]["per_fold_macro_f1"].append(macro_f1)
            all_results[v_name]["per_fold_per_class_f1"].append(per_class_f1.tolist())
            print(f"  fold {fold_idx+1}/{n_folds} | {v_name:18s} | macro-F1={macro_f1:.4f} | {elapsed:6.1f}s")

    print("[4/5] summarising + pairwise Wilcoxon signed-rank")
    summary = {}
    for v_name in variant_names:
        f1s = all_results[v_name]["per_fold_macro_f1"]
        summary[v_name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "per_fold_macro_f1": [float(x) for x in f1s],
            "mean_per_class_f1": np.mean(all_results[v_name]["per_fold_per_class_f1"], axis=0).tolist(),
            "class_names": class_names,
        }

    pairwise = {}
    for i, a in enumerate(variant_names):
        for b in variant_names[i + 1:]:
            va, vb = all_results[a]["per_fold_macro_f1"], all_results[b]["per_fold_macro_f1"]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    stat, pval = wilcoxon(va, vb)
                stat, pval = float(stat), float(pval)
                if not np.isfinite(stat): stat = None
                if not np.isfinite(pval): pval = None
            except ValueError:
                stat, pval = None, None
            pairwise[f"{a}_vs_{b}"] = {"statistic": stat, "p_value": pval}

    out = {"summary": summary, "pairwise_wilcoxon_tests": pairwise}
    out_path = os.path.join(output_dir, f"{dataset_key}_tabnet_baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"[5/5] saved -> {out_path}")
    print("\n=== macro-F1 (mean +/- std) ===")
    for v_name in variant_names:
        s = summary[v_name]
        print(f"  {v_name:18s} {s['mean_macro_f1']:.4f} +/- {s['std_macro_f1']:.4f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ertam_kaya", "cicids2017", "unsw_nb15"])
    ap.add_argument("--path", required=True)
    ap.add_argument("--feature_names_csv", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--output", default="results")
    args = ap.parse_args()
    run_tabnet_comparison(args.dataset, args.path, n_folds=args.folds,
                          output_dir=args.output, feature_names_csv=args.feature_names_csv)
