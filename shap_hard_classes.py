"""
shap_hard_classes.py

Extends the RQ3 SHAP analysis from the Analysis class
to the classes that are genuinely hardest on real data: Worms (F1=0.072) and Backdoors
(F1=0.080), in the main results. This addresses the observation that all three
severely underperforming classes merit comparative SHAP profiling rather than treating
Analysis as representative.

Protocol (identical to this study):
- SHAP KernelExplainer over the model's softmax probabilities
- Background set: 50 samples randomly drawn from the training set
- Up to 100 test-set samples of each target class are explained (Worms has only ~44
  test samples at a 75/25 split, in which case all available samples are used)
- Global level: mean(|SHAP|) per feature for the target-class output
- Local level: force plot for one instance per target class (a correctly classified
  instance if one exists; otherwise a misclassified one, which is flagged in the JSON
  output and is itself diagnostically interesting, as in Fig. 7 of this study)

The explained model is trained with the baseline (SMOTE) technique, consistent with
the main experiment. Because SMOTE plus 150-epoch training on the full 2.5M-row UNSW-NB15
training split is expensive, --max_train_samples allows a stratified subsample for the
model-training step (minority classes are always kept in full). Use 0 for the full
training split.

Usage:
    python shap_hard_classes.py --dataset unsw_nb15 --path /path/to/unsw.csv \
        --targets Worms Backdoors --max_train_samples 400000
"""

import argparse
import json
import os
import random

import numpy as np
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
import shap
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_loader import load_dataset
from adversarial_robustness import train_model
from imbalance_methods import baseline_smote

try:  # read experiment defaults from the user's config (never overwritten by us)
    from dataset_config import DATASET_PATHS, EPOCHS
except ImportError:
    DATASET_PATHS, EPOCHS = {}, 100

GLOBAL_SEED = 42
BACKGROUND_SIZE = 50
MAX_EXPLAINED = 100


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_subsample(X, y, max_samples, seed=GLOBAL_SEED):
    """Cap the number of training rows while always keeping minority classes in full.

    Classes are filled smallest-first so rare classes (e.g., Worms) are never reduced;
    only the largest classes are downsampled to fit the budget.
    """
    if max_samples <= 0 or len(y) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    order = np.argsort(counts)  # smallest classes first
    keep_idx = []
    remaining_budget = max_samples
    remaining_classes = len(classes)
    for pos in order:
        c = classes[pos]
        idx_c = np.where(y == c)[0]
        quota = remaining_budget // remaining_classes
        if len(idx_c) <= quota:
            keep_idx.append(idx_c)
            remaining_budget -= len(idx_c)
        else:
            keep_idx.append(rng.choice(idx_c, size=quota, replace=False))
            remaining_budget -= quota
        remaining_classes -= 1
    keep_idx = np.concatenate(keep_idx)
    rng.shuffle(keep_idx)
    return X[keep_idx], y[keep_idx]


def get_class_shap_matrix(shap_values, class_idx):
    """Normalize KernelExplainer output across shap versions.

    Older shap returns a list of per-class matrices; newer versions return a single
    array of shape (n_samples, n_features, n_classes).
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[class_idx])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, class_idx]
    return arr


def get_expected_value(explainer, class_idx):
    ev = explainer.expected_value
    if np.ndim(ev) == 0:
        return float(ev)
    return float(np.asarray(ev)[class_idx])


def main(dataset_key, path, feature_names_csv, targets, output_dir, max_train_samples):
    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(GLOBAL_SEED)

    print(f"[1/4] Loading dataset {dataset_key} from {path}")
    X_df, y_raw, spec = load_dataset(dataset_key, path,
                                     feature_names_csv=feature_names_csv) \
        if feature_names_csv else load_dataset(dataset_key, path)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    feature_names = X_df.columns.tolist()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)
    for t in targets:
        if t not in class_names:
            raise SystemExit(f"Target class '{t}' not found. Available: {class_names}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_full = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_train, y_train_sub = stratified_subsample(X_train_full, y_train,
                                                max_train_samples)
    print(f"[2/4] SMOTE-resampling {len(y_train_sub):,} rows "
          f"(of {len(y_train):,} available), then training for {EPOCHS} epochs")
    # Baseline (SMOTE) technique, consistent with the main experiment protocol.
    # Handle both possible signatures/returns of baseline_smote defensively.
    try:
        res = baseline_smote(X_train, y_train_sub, n_classes)
    except TypeError:
        res = baseline_smote(X_train, y_train_sub)
    X_res, y_res = res[0], res[1]
    model, _ = train_model(X_res, y_res, X_res.shape[1], n_classes,
                           epochs=EPOCHS)
    model.eval()
    device = next(model.parameters()).device

    def predict_proba(x):
        with torch.no_grad():
            logits = model(torch.as_tensor(np.asarray(x, dtype=np.float32)).to(device))
            return torch.softmax(logits, dim=1).cpu().numpy()

    with torch.no_grad():
        test_pred = predict_proba(X_test).argmax(axis=1)

    rng = np.random.default_rng(GLOBAL_SEED)
    background = X_train[rng.choice(len(X_train), size=BACKGROUND_SIZE,
                                    replace=False)]
    explainer = shap.KernelExplainer(predict_proba, background)

    summary_json = {"dataset_key": dataset_key,
                    "protocol": {"background_size": BACKGROUND_SIZE,
                                 "max_explained_per_class": MAX_EXPLAINED,
                                 "train_rows_used": int(len(y_train_sub)),
                                 "split": "75/25 stratified, random_state=42",
                                 "global_seed": GLOBAL_SEED},
                    "targets": {}}

    print(f"[3/4] Explaining target classes: {targets}")
    for target in targets:
        cls_idx = class_names.index(target)
        idx_target = np.where(y_test == cls_idx)[0]
        n_explain = min(MAX_EXPLAINED, len(idx_target))
        if n_explain == 0:
            print(f"  {target}: no test samples, skipping")
            continue
        chosen = rng.choice(idx_target, size=n_explain, replace=False)
        X_explain = X_test[chosen]
        print(f"  {target}: explaining {n_explain} of {len(idx_target)} test samples "
              f"(KernelExplainer, this is the slow step)")
        shap_values = explainer.shap_values(X_explain)
        sv_target = get_class_shap_matrix(shap_values, cls_idx)  # (n, n_features)

        # ---- Global: mean(|SHAP|) per feature for the target-class output ----
        mean_abs = np.abs(sv_target).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        plt.figure(figsize=(7, 4.5))
        plt.barh([feature_names[i] for i in order[::-1]],
                 mean_abs[order[::-1]], color="tab:blue")
        plt.xlabel(f"mean(|SHAP value|) for the {target} output")
        plt.title(f"Global feature importance for the {target} class (Dataset C)")
        plt.tight_layout()
        summary_path = os.path.join(output_dir, f"shap_summary_C_{target.lower()}.png")
        plt.savefig(summary_path, dpi=1200)
        plt.savefig(summary_path.replace(".png", ".pdf"))
        plt.close()

        # ---- Local: force plot for one instance ----
        pred_for_chosen = test_pred[chosen]
        correct_pos = np.where(pred_for_chosen == cls_idx)[0]
        if len(correct_pos) > 0:
            local_i = int(correct_pos[0])
            local_correct = True
        else:
            local_i = 0
            local_correct = False
        predicted_class = class_names[int(pred_for_chosen[local_i])]
        shap.force_plot(get_expected_value(explainer, cls_idx),
                        sv_target[local_i], X_explain[local_i],
                        feature_names=feature_names, matplotlib=True, show=False)
        local_path = os.path.join(output_dir, f"shap_local_C_{target.lower()}.png")
        plt.savefig(local_path, dpi=1200, bbox_inches="tight")
        plt.savefig(local_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()

        n_correct = int((pred_for_chosen == cls_idx).sum())
        summary_json["targets"][target] = {
            "test_samples_available": int(len(idx_target)),
            "samples_explained": int(n_explain),
            "explained_samples_correctly_classified": n_correct,
            "top_features_by_mean_abs_shap": [
                {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
                for i in order],
            "local_example": {"correctly_classified": local_correct,
                              "predicted_class": predicted_class,
                              "true_class": target},
            "figures": {"summary": os.path.basename(summary_path),
                        "local": os.path.basename(local_path)},
        }
        print(f"    -> {n_correct}/{n_explain} explained samples correctly "
              f"classified; local example is "
              f"{'correct' if local_correct else 'MISCLASSIFIED as ' + predicted_class}")

    print("[4/4] Saving JSON summary")
    out_path = os.path.join(output_dir, "shap_hard_classes_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="unsw_nb15")
    parser.add_argument("--path", default=None,
                        help="Defaults to DATASET_PATHS in dataset_config.py")
    parser.add_argument("--feature_names_csv", default=None,
                        help="Defaults to DATASET_PATHS in dataset_config.py")
    parser.add_argument("--targets", nargs="+", default=["Worms", "Backdoors"])
    parser.add_argument("--output", default="shap_hard_classes_results")
    parser.add_argument("--max_train_samples", type=int, default=400000,
                        help="Stratified cap on training rows for the explained model "
                             "(minority classes always kept in full); 0 = use all rows")
    args = parser.parse_args()
    cfg = DATASET_PATHS.get(args.dataset, {})
    path = args.path or cfg.get("path")
    fnames = args.feature_names_csv or cfg.get("feature_names_csv")
    if not path:
        raise SystemExit("No --path given and no entry in dataset_config.DATASET_PATHS")
    main(args.dataset, path, fnames, args.targets,
         args.output, args.max_train_samples)
