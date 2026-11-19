"""
adversarial_robustness.py - RQ4: test model robustness to feature-space evasion attacks
(FGSM-style perturbation), comparing performance before and after adversarial training.

Domain constraint (in the main experiment): the attacker may not modify
src_port/dst_port, because changing a port usually changes the meaning of the traffic
rather than being a small perturbation. Only quantitative features an attacker could
realistically adjust (bytes, duration, packet timing) are perturbable, consistent with
the network-traffic evasion literature (traffic shape is adjustable; protocol semantics are not).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from data_loader import load_dataset
from model import FirewallMLP
from imbalance_methods import baseline_smote

# Features that are "modifiable" under the domain constraint (ports are locked).
PERTURBABLE_FEATURES = ["duration", "bytes_total", "bytes_sent", "bytes_received",
                         "packets_total", "packets_sent", "packets_received"]


def fgsm_attack(model, loss_fn, X, y, epsilon, perturb_mask):
    device = next(model.parameters()).device  # use the same device as the model (cuda or cpu)
    X_t = torch.as_tensor(X, dtype=torch.float32).clone().to(device).requires_grad_(True)
    y_t = torch.as_tensor(y, dtype=torch.long).to(device)

    logits = model(X_t)
    loss = loss_fn(logits, y_t)
    model.zero_grad()
    loss.backward()

    grad_sign = X_t.grad.data.sign().cpu().numpy()  # always move back to CPU before numpy
    perturbation = epsilon * grad_sign * perturb_mask  # mask disables locked features (e.g. ports)
    X_adv = X + perturbation
    return X_adv


def evaluate_robust_accuracy(model, X, y, epsilon_list, perturb_mask, loss_fn):
    device = next(model.parameters()).device
    results = {}
    clean_pred = model(torch.as_tensor(X, dtype=torch.float32).to(device)).argmax(dim=1).cpu().detach().numpy()
    clean_acc = float((clean_pred == y).mean())
    results["clean_accuracy"] = clean_acc

    correctly_classified_mask = clean_pred == y

    for eps in epsilon_list:
        X_adv = fgsm_attack(model, loss_fn, X, y, eps, perturb_mask)
        adv_pred = model(torch.as_tensor(X_adv, dtype=torch.float32).to(device)).argmax(dim=1).cpu().detach().numpy()
        robust_acc = float((adv_pred == y).mean())

        # attack success rate: of samples originally classified correctly, the fraction now flipped
        if correctly_classified_mask.sum() > 0:
            success_rate = float((adv_pred[correctly_classified_mask] != y[correctly_classified_mask]).mean())
        else:
            success_rate = float("nan")

        results[f"epsilon_{eps}"] = {"robust_accuracy": robust_acc, "attack_success_rate": success_rate}
    return results


def train_model(X_train, y_train, n_features, n_classes, epochs, adversarial=False,
                 epsilon_train=0.3, perturb_mask=None, hidden_size=200):
    model = FirewallMLP(n_features=n_features, n_classes=n_classes, hidden_size=hidden_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_t = torch.as_tensor(y_train, dtype=torch.long)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)

        if adversarial:
            # simple adversarial training: mix adversarial examples into every epoch
            X_adv = fgsm_attack(model, loss_fn, X_train, y_train, epsilon_train, perturb_mask)
            X_adv_t = torch.as_tensor(X_adv, dtype=torch.float32)
            logits_adv = model(X_adv_t)
            loss_adv = loss_fn(logits_adv, y_t)
            loss = 0.5 * loss + 0.5 * loss_adv

        loss.backward()
        optimizer.step()

    model.eval()
    return model, loss_fn


def main():
    X_df, y_raw, spec = load_dataset("ertam_kaya", "/tmp/dataset_A_ertam_kaya_style.csv")
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)
    feature_names = X_df.columns.tolist()
    perturb_mask = np.array([1.0 if f in PERTURBABLE_FEATURES else 0.0 for f in feature_names])
    print("Feature mask (1=perturbable under domain constraint, 0=locked):")
    for f, m in zip(feature_names, perturb_mask):
        print(f"  {f:20s} {'perturbable' if m else 'LOCKED (port)'}")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    n_classes = len(label_encoder.classes_)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_df.values, y, test_size=0.25, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    epsilon_list = [0.1, 0.3, 0.5, 1.0]

    print("\n=== [1] Train standard model (no adversarial training) ===")
    model_standard, loss_fn = train_model(X_train, y_train, X_train.shape[1], n_classes, epochs=150)
    results_standard = evaluate_robust_accuracy(model_standard, X_test, y_test, epsilon_list, perturb_mask, loss_fn)
    print(results_standard)

    print("\n=== [2] Train model with adversarial training (epsilon_train=0.3) ===")
    model_adv, loss_fn2 = train_model(X_train, y_train, X_train.shape[1], n_classes, epochs=150,
                                       adversarial=True, epsilon_train=0.3, perturb_mask=perturb_mask)
    results_adv = evaluate_robust_accuracy(model_adv, X_test, y_test, epsilon_list, perturb_mask, loss_fn2)
    print(results_adv)

    import json
    with open("/tmp/adversarial_results.json", "w") as f:
        json.dump({"standard_training": results_standard, "adversarial_training": results_adv,
                   "epsilon_list": epsilon_list, "perturbable_features": PERTURBABLE_FEATURES}, f, indent=2)
    print("\nSaved /tmp/adversarial_results.json")


if __name__ == "__main__":
    main()
