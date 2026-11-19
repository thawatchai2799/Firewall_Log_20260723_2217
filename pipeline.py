"""
pipeline.py

Example end-to-end pipeline: load dataset -> preprocess -> stratified k-fold ->
train an MLP with each technique in imbalance_methods.py -> report per-class F1 / macro-F1 with
paired statistical tests between techniques (answers RQ1 in the main experiment).

Example run:
    python pipeline.py --dataset ertam_kaya --path data/raw/ertam_kaya_firewall.csv \
        --techniques baseline_smote cost_sensitive focal_loss hybrid_resample \
        --folds 10 --epochs 100 --output results/
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

from data_loader import load_dataset, report_data_quality
from model import FirewallMLP
from imbalance_methods import TECHNIQUES


def train_one_fold(X_train, y_train, X_val, y_val, n_features, n_classes,
                    loss_fn, epochs=100, lr=1e-3, hidden_size=200, device="cpu"):
    model = FirewallMLP(n_features=n_features, n_classes=n_classes, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long).to(device)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(device)

    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(device)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train_t)
        loss = loss_fn(logits, y_train_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t)
        val_pred = val_logits.argmax(dim=1).cpu().numpy()

    return val_pred


def run_comparison(dataset_key: str, path: str, technique_names, n_folds: int = 10,
                    epochs: int = 100, output_dir: str = "results", feature_names_csv=None):
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[1/5] Loading dataset: {dataset_key}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)

    dq = report_data_quality(X_df, name=dataset_key)
    dq.to_csv(os.path.join(output_dir, f"{dataset_key}_data_quality.csv"))

    # simple median imputation - adjust based on the actual data-quality report
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    X_df = X_df.fillna(0)  # guard against an all-NaN column (median is also NaN)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_.tolist()

    print(f"[2/5] number of classes = {n_classes}: {class_names}")
    print(f"      class proportions:\n{pd.Series(y_raw).value_counts(normalize=True)}")

    X = X_df.values
    n_features = X.shape[1]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_results = {tech: {"per_fold_macro_f1": [], "per_fold_per_class_f1": []}
                   for tech in technique_names}

    print(f"[3/5] starting {n_folds}-fold cross-validation for {len(technique_names)} techniques")
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_raw, X_val_raw = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # scale only after the split (fit on the train fold only, to prevent data leakage)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_val_scaled = scaler.transform(X_val_raw)

        for tech_name in technique_names:
            tech_fn = TECHNIQUES[tech_name]
            X_res, y_res, loss_fn = tech_fn(X_train_scaled, y_train, n_classes)

            val_pred = train_one_fold(
                X_res, y_res, X_val_scaled, y_val,
                n_features=n_features, n_classes=n_classes,
                loss_fn=loss_fn, epochs=epochs, device=device,
            )

            macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
            _, _, per_class_f1, _ = precision_recall_fscore_support(
                y_val, val_pred, labels=range(n_classes), zero_division=0)

            all_results[tech_name]["per_fold_macro_f1"].append(macro_f1)
            all_results[tech_name]["per_fold_per_class_f1"].append(per_class_f1.tolist())

            print(f"  fold {fold_idx+1}/{n_folds} | {tech_name:16s} | macro-F1={macro_f1:.4f}")

    print("[4/5] summarising and testing statistical significance (Wilcoxon signed-rank, pairwise)")
    summary = {}
    for tech_name in technique_names:
        f1s = all_results[tech_name]["per_fold_macro_f1"]
        summary[tech_name] = {
            "mean_macro_f1": float(np.mean(f1s)),
            "std_macro_f1": float(np.std(f1s)),
            "mean_per_class_f1": np.mean(
                all_results[tech_name]["per_fold_per_class_f1"], axis=0).tolist(),
            "class_names": class_names,
        }

    pairwise_tests = {}
    for i, tech_a in enumerate(technique_names):
        for tech_b in technique_names[i + 1:]:
            a = all_results[tech_a]["per_fold_macro_f1"]
            b = all_results[tech_b]["per_fold_macro_f1"]
            try:
                stat, p_value = wilcoxon(a, b)
            except ValueError:
                stat, p_value = float("nan"), float("nan")
            pairwise_tests[f"{tech_a}_vs_{tech_b}"] = {"statistic": float(stat), "p_value": float(p_value)}

    print("[5/5] saving results")
    output = {"summary": summary, "pairwise_wilcoxon_tests": pairwise_tests}
    with open(os.path.join(output_dir, f"{dataset_key}_comparison_results.json"), "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(pairwise_tests, indent=2, ensure_ascii=False))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["ertam_kaya", "cicids2017", "unsw_nb15"])
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    parser.add_argument("--techniques", nargs="+",
                         default=["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"])
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output", default="results")
    args = parser.parse_args()

    run_comparison(
        dataset_key=args.dataset,
        path=args.path,
        technique_names=args.techniques,
        n_folds=args.folds,
        epochs=args.epochs,
        output_dir=args.output,
        feature_names_csv=args.feature_names_csv,
    )
