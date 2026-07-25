"""
explanation_robustness_experiment.py

Tests what this study flagged as "not yet done": how stable SHAP explanations
are under (1) adversarial attack and (2) adversarial training.

Compares three conditions on the same background/explain samples throughout (for a fair
comparison, free of variance from different random samples):

  1. standard_clean      : standard model, clean input (unperturbed)
  2. standard_perturbed  : the same standard model, input attacked with FGSM at epsilon=0.3
                           (the same as used during adversarial training in the main experiment)
  3. adversarial_clean   : adversarially trained model, clean input (unperturbed)

Comparison (1)-(2) answers "does the explanation change under attack?" (explainability
drift, per Rajhans & Khawarey [25]).
Comparison (1)-(3) answers "does robust training change what the model relies on?"

Drift is measured with three metrics: Spearman rank correlation of feature-importance
rankings, cosine similarity of the importance vectors, and top-1/top-3 feature overlap.

Usage:
    python explanation_robustness_experiment.py --dataset ertam_kaya --path /path/to/log2.csv
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import shap
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 1200,
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.linewidth": 1.1,
    "savefig.bbox": "tight",
})

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_loader import load_dataset
from model import FirewallMLP
from imbalance_methods import baseline_smote
from adversarial_robustness import fgsm_attack, PERTURBABLE_FEATURES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_SIZE = 200
LEARNING_RATE = 1e-3
EPOCHS = 150
EPSILON = 0.3           # same perturbation level used during adversarial training in the main experiment
N_BACKGROUND = 50
N_EXPLAIN = 100


def train_model(X_train, y_train, n_features, n_classes, adversarial, perturb_mask):
    """Train a standard model (adversarial=False) or an adversarially trained model
    (adversarial=True). The structure is identical to run_adversarial_unit in
    run_full_experiment.py, so the resulting models are directly comparable with the main results."""
    model = FirewallMLP(n_features=n_features, n_classes=n_classes, hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.as_tensor(y_train, dtype=torch.long).to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        if adversarial:
            X_adv = fgsm_attack(model, loss_fn, X_train, y_train, EPSILON, perturb_mask)
            X_adv_t = torch.as_tensor(X_adv, dtype=torch.float32).to(DEVICE)
            loss = 0.5 * loss + 0.5 * loss_fn(model(X_adv_t), y_t)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def make_predict_proba(model):
    def predict_proba(x_numpy):
        with torch.no_grad():
            logits = model(torch.as_tensor(x_numpy, dtype=torch.float32).to(DEVICE))
            return torch.softmax(logits, dim=1).cpu().numpy()
    return predict_proba


def global_importance_vector(shap_values, n_classes):
    """Sum mean(|SHAP value|) across all classes into a single vector of length n_features
    (like the total bar lengths in summary_plot type='bar'), used as a proxy for the
    'overall feature-importance ranking' when comparing the three conditions."""
    if isinstance(shap_values, list):
        stacked = np.stack(shap_values, axis=0)  # (n_classes, n_samples, n_features)
    else:
        stacked = np.transpose(shap_values, (2, 0, 1))  # -> (n_classes, n_samples, n_features)
    per_class_mean_abs = np.mean(np.abs(stacked), axis=1)  # (n_classes, n_features)
    return per_class_mean_abs.sum(axis=0)  # (n_features,) summed over all classes


def compare_importance(vec_a, vec_b, feature_names, label_a, label_b):
    rho, pval = spearmanr(vec_a, vec_b)
    cos_sim = float(np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b) + 1e-12))
    top3_a = set(np.array(feature_names)[np.argsort(vec_a)[::-1][:3]])
    top3_b = set(np.array(feature_names)[np.argsort(vec_b)[::-1][:3]])
    top1_a = feature_names[int(np.argmax(vec_a))]
    top1_b = feature_names[int(np.argmax(vec_b))]
    return {
        "comparison": f"{label_a}_vs_{label_b}",
        "spearman_rho": float(rho), "spearman_p": float(pval),
        "cosine_similarity": cos_sim,
        "top1_feature_a": top1_a, "top1_feature_b": top1_b,
        "top1_same": bool(top1_a == top1_b),
        "top3_overlap_count": len(top3_a & top3_b),
        "top3_features_a": sorted(top3_a), "top3_features_b": sorted(top3_b),
    }


def main(dataset_key, path, feature_names_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"[1/6] Loading data {dataset_key} from {path}")
    X_df, y_raw, spec = load_dataset(dataset_key, path, feature_names_csv=feature_names_csv)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    feature_names = X_df.columns.tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    perturb_mask = np.array([1.0 if f in PERTURBABLE_FEATURES else 0.0 for f in feature_names])

    print("[2/6] Training the standard model and the adversarially trained model")
    X_res, y_res, _ = baseline_smote(X_train, y_train, n_classes)
    model_standard = train_model(X_res, y_res, X_train.shape[1], n_classes, adversarial=False, perturb_mask=perturb_mask)
    model_adversarial = train_model(X_res, y_res, X_train.shape[1], n_classes, adversarial=True, perturb_mask=perturb_mask)

    acc_standard = (make_predict_proba(model_standard)(X_test).argmax(axis=1) == y_test).mean()
    acc_adversarial = (make_predict_proba(model_adversarial)(X_test).argmax(axis=1) == y_test).mean()
    print(f"      clean accuracy: standard={acc_standard:.4f}, adversarial-trained={acc_adversarial:.4f}")

    print("[3/6] Preparing the same background/explain samples for every condition")
    background = shap.sample(X_train, N_BACKGROUND, random_state=42)
    explain_clean = X_test[:N_EXPLAIN].copy()

    print("[4/6] Building adversarial examples from the same explain set (epsilon=0.3, standard-model gradients)")
    loss_fn = nn.CrossEntropyLoss()
    explain_perturbed = fgsm_attack(model_standard, loss_fn, explain_clean,
                                     y_test[:N_EXPLAIN], EPSILON, perturb_mask)

    print("[5/6] Computing SHAP for all three conditions (may take several minutes per condition)")
    results = {}
    conditions = [
        ("standard_clean", model_standard, explain_clean),
        ("standard_perturbed", model_standard, explain_perturbed),
        ("adversarial_clean", model_adversarial, explain_clean),
    ]
    importance_vectors = {}
    for name, model, sample in conditions:
        print(f"      computing SHAP: {name}")
        explainer = shap.KernelExplainer(make_predict_proba(model), background)
        raw_shap = explainer.shap_values(sample, nsamples=100)
        shap_values = raw_shap if isinstance(raw_shap, list) else \
            [raw_shap[:, :, c] for c in range(n_classes)]
        importance_vectors[name] = global_importance_vector(shap_values, n_classes)

        plt.figure()
        shap.summary_plot(shap_values, sample, feature_names=feature_names,
                           class_names=class_names, plot_type="bar", show=False)
        plt.title(f"Global Feature Importance \u2014 {name} ({dataset_key})")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"shap_explanation_robustness_{name}.png"), dpi=1200)
        plt.savefig(os.path.join(output_dir, f"shap_explanation_robustness_{name}.pdf"), dpi=1200)
        plt.close()

    print("[6/6] Comparing drift between conditions")
    comparisons = [
        compare_importance(importance_vectors["standard_clean"], importance_vectors["standard_perturbed"],
                            feature_names, "standard_clean", "standard_perturbed"),
        compare_importance(importance_vectors["standard_clean"], importance_vectors["adversarial_clean"],
                            feature_names, "standard_clean", "adversarial_clean"),
        compare_importance(importance_vectors["standard_perturbed"], importance_vectors["adversarial_clean"],
                            feature_names, "standard_perturbed", "adversarial_clean"),
    ]
    for c in comparisons:
        print(f"  {c['comparison']}: Spearman rho={c['spearman_rho']:.3f} (p={c['spearman_p']:.4f}), "
              f"cosine={c['cosine_similarity']:.3f}, top1 same={c['top1_same']}, "
              f"top3 overlap={c['top3_overlap_count']}/3")

    # single grouped-bar plot comparing the importance ranking across the three conditions
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(feature_names))
    width = 0.27
    order = np.argsort(importance_vectors["standard_clean"])[::-1]
    sorted_features = [feature_names[i] for i in order]
    for i, name in enumerate(["standard_clean", "standard_perturbed", "adversarial_clean"]):
        vals = importance_vectors[name][order]
        ax.bar(x + i * width, vals, width, label=name)
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted_features, rotation=45, ha="right")
    ax.set_ylabel("Total mean(|SHAP value|) across classes")
    ax.set_title(f"Explanation-Robustness Comparison ({dataset_key})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_explanation_robustness_comparison.png"), dpi=1200)
    plt.savefig(os.path.join(output_dir, "fig_explanation_robustness_comparison.pdf"), dpi=1200)
    plt.close()

    output = {
        "dataset_key": dataset_key,
        "epsilon": EPSILON,
        "clean_accuracy": {"standard": float(acc_standard), "adversarial_trained": float(acc_adversarial)},
        "feature_names": feature_names,
        "importance_vectors": {k: v.tolist() for k, v in importance_vectors.items()},
        "comparisons": comparisons,
    }
    out_path = os.path.join(output_dir, "explanation_robustness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_path}")
    print(f"Saved figures to {output_dir}/fig_explanation_robustness_comparison.png and shap_explanation_robustness_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ertam_kaya")
    parser.add_argument("--path", required=True)
    parser.add_argument("--feature_names_csv", default=None)
    parser.add_argument("--output", default="explanation_robustness_results")
    args = parser.parse_args()
    main(args.dataset, args.path, args.feature_names_csv, args.output)
