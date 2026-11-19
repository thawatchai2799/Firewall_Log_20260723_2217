"""
pgd_attack.py

Extends the RQ4 adversarial-robustness evaluation with
Projected Gradient Descent (PGD), a stronger multi-step white-box attack. This addresses
the limitation noted in the main experiment that only single-step FGSM was tested.

Design decisions:
1. Same domain constraint as the FGSM experiment: port-type features are locked via
   perturb_mask; only quantitative traffic-shape features (duration, bytes, packets)
   may be perturbed.
2. FGSM and PGD are evaluated on the SAME trained models within this single run, so the
   FGSM-vs-PGD comparison is free of cross-run seeding variance (the caveat raised for
   the main experiment). Global seeds are also fixed explicitly.
3. Same protocol as adversarial_robustness.py: 75/25 stratified split (random_state=42),
   150 training epochs, epsilon in {0.1, 0.3, 0.5, 1.0} standardized units, and
   adversarial training performed at epsilon_train=0.3.

PGD follows Madry et al. (2018): a random start uniformly sampled inside the epsilon
L-infinity ball, followed by n_steps signed-gradient steps of size alpha, projecting back
onto the epsilon ball after every step. The perturbation mask is applied to the random
start, to every step, and to the projection bounds, so locked features are never modified.

Usage:
    python pgd_attack.py --dataset ertam_kaya --path /path/to/log2.csv
    # optional: --steps 10 --output pgd_results
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
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_loader import load_dataset
from adversarial_robustness import PERTURBABLE_FEATURES, train_model, fgsm_attack

try:  # read dataset paths from the user's config (never overwritten by us)
    from dataset_config import DATASET_PATHS
except ImportError:
    DATASET_PATHS = {}

EPSILON_LIST = [0.1, 0.3, 0.5, 1.0]
# NOTE: training uses epochs=150 to match the main experiment (FGSM) protocol exactly,
# not the EPOCHS=100 used by the RQ1 cross-validation experiments.
GLOBAL_SEED = 42


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pgd_attack(model, loss_fn, X, y, epsilon, perturb_mask, n_steps=10, alpha=None,
               random_start=True, seed=GLOBAL_SEED):
    """Multi-step L-infinity PGD with masked features and projection after every step.

    alpha defaults to 2.5 * epsilon / n_steps (Madry et al.), which allows the attack to
    reach the boundary of the epsilon ball from any random start within n_steps.
    """
    device = next(model.parameters()).device
    if alpha is None:
        alpha = 2.5 * epsilon / n_steps

    X = np.asarray(X, dtype=np.float32)
    mask = np.asarray(perturb_mask, dtype=np.float32)
    rng = np.random.default_rng(seed)

    if random_start:
        delta = rng.uniform(-epsilon, epsilon, size=X.shape).astype(np.float32) * mask
    else:
        delta = np.zeros_like(X, dtype=np.float32)
    X_adv = X + delta

    y_t = torch.as_tensor(y, dtype=torch.long).to(device)
    lower = X - epsilon * mask
    upper = X + epsilon * mask

    for _ in range(n_steps):
        X_t = torch.as_tensor(X_adv, dtype=torch.float32).to(device).requires_grad_(True)
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        model.zero_grad()
        loss.backward()
        grad_sign = X_t.grad.data.sign().cpu().numpy()
        X_adv = X_adv + alpha * grad_sign * mask
        X_adv = np.clip(X_adv, lower, upper)  # project back onto the epsilon ball
    return X_adv


def evaluate_attack(model, X, y, epsilon_list, perturb_mask, loss_fn, attack_name,
                    n_steps=10):
    """Evaluate one attack (fgsm or pgd) at every epsilon on one model.

    Returns the same structure as evaluate_robust_accuracy in adversarial_robustness.py
    (clean_accuracy + per-epsilon robust_accuracy / attack_success_rate) so downstream
    reporting is directly comparable with the main experiment tables.
    """
    device = next(model.parameters()).device
    results = {}
    with torch.no_grad():
        clean_pred = model(torch.as_tensor(X, dtype=torch.float32).to(device)) \
            .argmax(dim=1).cpu().numpy()
    results["clean_accuracy"] = float((clean_pred == y).mean())
    correct_mask = clean_pred == y

    for eps in epsilon_list:
        if attack_name == "fgsm":
            X_adv = fgsm_attack(model, loss_fn, X, y, eps, perturb_mask)
        elif attack_name == "pgd":
            X_adv = pgd_attack(model, loss_fn, X, y, eps, perturb_mask, n_steps=n_steps)
        else:
            raise ValueError(f"unknown attack: {attack_name}")

        with torch.no_grad():
            adv_pred = model(torch.as_tensor(X_adv, dtype=torch.float32).to(device)) \
                .argmax(dim=1).cpu().numpy()
        robust_acc = float((adv_pred == y).mean())
        if correct_mask.sum() > 0:
            success = float((adv_pred[correct_mask] != y[correct_mask]).mean())
        else:
            success = float("nan")
        results[f"epsilon_{eps}"] = {"robust_accuracy": robust_acc,
                                     "attack_success_rate": success}
        print(f"    {attack_name.upper():4s} eps={eps}: robust_acc={robust_acc:.4f}  "
              f"attack_success={success:.4f}")
    return results


def plot_results(all_results, epsilon_list, n_steps, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {
        ("standard", "fgsm"): dict(color="tab:blue", linestyle="--", marker="o",
                                   label="Standard / FGSM"),
        ("standard", "pgd"): dict(color="tab:blue", linestyle="-", marker="s",
                                  label=f"Standard / PGD-{n_steps}"),
        ("adversarial", "fgsm"): dict(color="tab:orange", linestyle="--", marker="o",
                                      label="Adv. trained / FGSM"),
        ("adversarial", "pgd"): dict(color="tab:orange", linestyle="-", marker="s",
                                     label=f"Adv. trained / PGD-{n_steps}"),
    }
    for (training, attack), style in styles.items():
        res = all_results[training][attack]
        xs = [0.0] + epsilon_list
        accs = [res["clean_accuracy"]] + [res[f"epsilon_{e}"]["robust_accuracy"]
                                          for e in epsilon_list]
        axes[0].plot(xs, accs, **style)
        succ = [res[f"epsilon_{e}"]["attack_success_rate"] for e in epsilon_list]
        axes[1].plot(epsilon_list, succ, **style)

    axes[0].set_xlabel("Perturbation budget (epsilon)")
    axes[0].set_ylabel("Robust accuracy")
    axes[0].set_title("(a) Robust accuracy: FGSM vs. PGD")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Perturbation budget (epsilon)")
    axes[1].set_ylabel("Attack success rate")
    axes[1].set_title("(b) Attack success rate: FGSM vs. PGD")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=1200)
    plt.savefig(fig_path.replace(".png", ".pdf"))
    plt.close()


def main(dataset_key, path, feature_names_csv, output_dir, n_steps):
    os.makedirs(output_dir, exist_ok=True)
    set_all_seeds(GLOBAL_SEED)

    print(f"[1/4] Loading dataset {dataset_key} from {path}")
    X_df, y_raw, spec = load_dataset(dataset_key, path,
                                     feature_names_csv=feature_names_csv) \
        if feature_names_csv else load_dataset(dataset_key, path)
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    feature_names = X_df.columns.tolist()
    perturb_mask = np.array([1.0 if f in PERTURBABLE_FEATURES else 0.0
                             for f in feature_names], dtype=np.float32)
    print("Feature mask (1=perturbable under domain constraint, 0=locked):")
    for f, m in zip(feature_names, perturb_mask):
        print(f"  {f:20s} {'perturbable' if m else 'LOCKED'}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    print("\n[2/4] Training standard model (no adversarial training)")
    model_std, loss_fn_std = train_model(X_train, y_train, X_train.shape[1],
                                         n_classes, epochs=150)
    print("\n[3/4] Training adversarially trained model (epsilon_train=0.3)")
    model_adv, loss_fn_adv = train_model(X_train, y_train, X_train.shape[1],
                                         n_classes, epochs=150, adversarial=True,
                                         epsilon_train=0.3, perturb_mask=perturb_mask)

    print("\n[4/4] Evaluating FGSM and PGD on both models (same models, same run)")
    all_results = {}
    for training, model, loss_fn in [("standard", model_std, loss_fn_std),
                                     ("adversarial", model_adv, loss_fn_adv)]:
        print(f"  == {training} model ==")
        all_results[training] = {}
        for attack in ["fgsm", "pgd"]:
            all_results[training][attack] = evaluate_attack(
                model, X_test, y_test, EPSILON_LIST, perturb_mask, loss_fn,
                attack, n_steps=n_steps)

    fig_path = os.path.join(output_dir, "fig_pgd_vs_fgsm.png")
    plot_results(all_results, EPSILON_LIST, n_steps, fig_path)

    output = {
        "dataset_key": dataset_key,
        "protocol": {
            "split": "75/25 stratified, random_state=42",
            "epochs": 150,
            "epsilon_list": EPSILON_LIST,
            "adversarial_training_epsilon": 0.3,
            "pgd_steps": n_steps,
            "pgd_alpha": f"2.5*epsilon/{n_steps}",
            "pgd_random_start": True,
            "global_seed": GLOBAL_SEED,
            "note": ("FGSM and PGD evaluated on the same trained models within this "
                     "run; the FGSM-vs-PGD comparison is therefore free of cross-run "
                     "seeding variance."),
        },
        "perturbable_features": PERTURBABLE_FEATURES,
        "results": all_results,
    }
    out_path = os.path.join(output_dir, "pgd_attack_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\nSaved results to {out_path}")
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="ertam_kaya")
    parser.add_argument("--path", default=None,
                        help="Defaults to DATASET_PATHS in dataset_config.py")
    parser.add_argument("--feature_names_csv", default=None,
                        help="Defaults to DATASET_PATHS in dataset_config.py")
    parser.add_argument("--output", default="pgd_results")
    parser.add_argument("--steps", type=int, default=10,
                        help="Number of PGD steps (default 10)")
    args = parser.parse_args()
    cfg = DATASET_PATHS.get(args.dataset, {})
    path = args.path or cfg.get("path")
    fnames = args.feature_names_csv or cfg.get("feature_names_csv")
    if not path:
        raise SystemExit("No --path given and no entry in dataset_config.DATASET_PATHS")
    main(args.dataset, path, fnames, args.output, args.steps)
