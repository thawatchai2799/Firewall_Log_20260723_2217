"""
run_full_experiment.py

Main script for running the full experiment (RQ1-RQ4 on all three datasets), designed to be
left running for days.

Key features:
  - Always resumable, even after a power loss / crash / sudden terminal close
    (every work unit writes its result to disk as soon as it finishes; nothing is only held in memory).
  - Reports progress (% done, elapsed time, estimated time remaining) every 60 seconds,
    to both the screen and progress.log.
  - A unit that errors is recorded and then "skipped so the next unit runs"; it does not halt the whole experiment.
  - Handles large datasets (mini-batch training + automatic GPU use if available).

How to use:
  1. Edit DATASET_PATHS below to point at your real data files
     (the defaults point at existing mock data, useful for checking the script runs correctly first).
  2. Run:  python run_full_experiment.py
  3. Leave it running - if power is lost / the machine restarts, just run the same command again.
     The script skips already-completed units automatically; it does not start over.
  4. Check progress at any time without waiting, with:  python run_full_experiment.py --status
  5. When complete, all combined results are at results_final/final_summary.json,
     along with progress.log, which keeps the full run log for later inspection.
"""

import argparse
import json
import os
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_recall_fscore_support

from data_loader import load_dataset
from model import FirewallMLP
from imbalance_methods import TECHNIQUES
from checkpoint_utils import (
    ProgressTracker, is_unit_done, save_unit_result, load_all_done_units,
    ensure_checkpoint_dir, CHECKPOINT_DIR,
)

# =============================================================================
# All defaults live separately in dataset_config.py (that file is *never* modified again,
# no matter how many new bug-fixed versions of run_full_experiment.py are shipped).
# Edit all paths/defaults in dataset_config.py instead; there is no need to edit this file.
# =============================================================================
from dataset_config import DATASET_PATHS, TECHNIQUE_NAMES, N_FOLDS, EPOCHS, HIDDEN_SIZE, \
    BATCH_SIZE, LEARNING_RATE, HEARTBEAT_SECONDS, SHAP_DATASETS, ADVERSARIAL_DATASET, \
    OUTPUT_DIR, PROGRESS_LOG_PATH
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Cache loaded/preprocessed data (in memory during a run, to avoid reloading for every unit)
# =============================================================================
_dataset_cache = {}


def get_prepared_dataset(dataset_key: str):
    """Load + clean + label-encode a dataset (done once per run, cached in RAM)."""
    if dataset_key in _dataset_cache:
        return _dataset_cache[dataset_key]

    cfg = DATASET_PATHS[dataset_key]
    X_df, y_raw, spec = load_dataset(dataset_key, cfg["path"], feature_names_csv=cfg.get("feature_names_csv"))
    X_df = X_df.fillna(X_df.median(numeric_only=True)).fillna(0)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    class_names = label_encoder.classes_.tolist()
    n_classes = len(class_names)

    X = X_df.values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))  # deterministic given a fixed random_state -> resume reproduces exactly

    prepared = {
        "X": X, "y": y, "class_names": class_names, "n_classes": n_classes,
        "n_features": X.shape[1], "folds": folds, "feature_names": X_df.columns.tolist(),
    }
    _dataset_cache[dataset_key] = prepared
    return prepared


# =============================================================================
# Model training (mini-batch, GPU-capable, handles large datasets)
# =============================================================================
def train_and_eval_fold(X_train, y_train, X_val, y_val, n_features, n_classes, loss_fn,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
                         hidden_size=HIDDEN_SIZE, progress_cb=None):
    model = FirewallMLP(n_features=n_features, n_classes=n_classes, hidden_size=hidden_size).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if hasattr(loss_fn, "to"):
        loss_fn = loss_fn.to(DEVICE)

    X_train_t = torch.as_tensor(X_train, dtype=torch.float32)
    y_train_t = torch.as_tensor(y_train, dtype=torch.long)
    n_samples = X_train_t.shape[0]

    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            xb = X_train_t[idx].to(DEVICE)
            yb = y_train_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        if progress_cb is not None:
            progress_cb(epoch + 1, epochs)

    model.eval()
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
    return val_pred


# =============================================================================
# Define all work units (deterministic - identical on every run, no matter how many times it resumes)
# =============================================================================
def build_unit_list():
    units = []
    for dataset_key in DATASET_PATHS:
        for tech_name in TECHNIQUE_NAMES:
            for fold_idx in range(N_FOLDS):
                units.append({
                    "id": f"cv/{dataset_key}/{tech_name}/fold{fold_idx}",
                    "type": "cv_fold",
                    "dataset_key": dataset_key, "technique": tech_name, "fold_idx": fold_idx,
                })
    for dataset_key, hardest_class in SHAP_DATASETS.items():
        units.append({
            "id": f"shap/{dataset_key}",
            "type": "shap", "dataset_key": dataset_key, "hardest_class": hardest_class,
        })
    for mode in ["standard", "adversarial"]:
        units.append({
            "id": f"adversarial/{ADVERSARIAL_DATASET}/{mode}",
            "type": "adversarial", "dataset_key": ADVERSARIAL_DATASET, "mode": mode,
        })
    return units


# =============================================================================
# Functions that process each type of unit
# =============================================================================
def run_cv_fold_unit(unit, tracker):
    dataset_key, tech_name, fold_idx = unit["dataset_key"], unit["technique"], unit["fold_idx"]
    data = get_prepared_dataset(dataset_key)
    train_idx, val_idx = data["folds"][fold_idx]
    X_train_raw, X_val_raw = data["X"][train_idx], data["X"][val_idx]
    y_train, y_val = data["y"][train_idx], data["y"][val_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    tech_fn = TECHNIQUES[tech_name]
    X_res, y_res, loss_fn = tech_fn(X_train, y_train, data["n_classes"])

    def epoch_cb(done_epoch, total_epoch):
        tracker.heartbeat()

    val_pred = train_and_eval_fold(
        X_res, y_res, X_val, y_val,
        n_features=data["n_features"], n_classes=data["n_classes"], loss_fn=loss_fn,
        progress_cb=epoch_cb,
    )

    macro_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
    _, _, per_class_f1, _ = precision_recall_fscore_support(
        y_val, val_pred, labels=range(data["n_classes"]), zero_division=0)

    return {
        "dataset_key": dataset_key, "technique": tech_name, "fold_idx": fold_idx,
        "macro_f1": float(macro_f1), "per_class_f1": per_class_f1.tolist(),
        "class_names": data["class_names"],
    }


def run_shap_unit(unit, tracker):
    import shap
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
    import pandas as pd
    from imbalance_methods import baseline_smote

    dataset_key, hardest_class = unit["dataset_key"], unit["hardest_class"]
    data = get_prepared_dataset(dataset_key)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.25, stratify=data["y"], random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    X_res, y_res, loss_fn = baseline_smote(X_train, y_train, data["n_classes"])
    tracker.heartbeat()

    model = FirewallMLP(n_features=data["n_features"], n_classes=data["n_classes"], hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    X_res_t = torch.as_tensor(X_res, dtype=torch.float32)  # keep on CPU first, then send to GPU batch by batch
    y_res_t = torch.as_tensor(y_res, dtype=torch.long)
    n_samples_shap = X_res_t.shape[0]
    model.train()
    for epoch in range(150):
        perm = torch.randperm(n_samples_shap)
        for start in range(0, n_samples_shap, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = X_res_t[idx].to(DEVICE)
            yb = y_res_t[idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            tracker.heartbeat()
    model.eval()

    def predict_proba(x_numpy):
        x_t_full = torch.as_tensor(x_numpy, dtype=torch.float32)
        outputs = []
        with torch.no_grad():
            for start in range(0, x_t_full.shape[0], BATCH_SIZE):
                xb = x_t_full[start:start + BATCH_SIZE].to(DEVICE)
                logits = model(xb)
                outputs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(outputs, axis=0)

    background = shap.sample(X_train, 50, random_state=42)
    n_explain = 100
    explain_sample = X_test[:n_explain].copy()
    y_explain = y_test[:n_explain].copy()
    target_class_idx = data["class_names"].index(hardest_class)
    hardest_idx_full = np.where(y_test == target_class_idx)[0]
    if len(hardest_idx_full) > 0 and target_class_idx not in y_explain:
        explain_sample[-1] = X_test[hardest_idx_full[0]]
        y_explain[-1] = target_class_idx

    explainer = shap.KernelExplainer(predict_proba, background)
    raw_shap_values = explainer.shap_values(explain_sample, nsamples=100)
    tracker.heartbeat()
    shap_values = raw_shap_values if isinstance(raw_shap_values, list) else \
        [raw_shap_values[:, :, c] for c in range(data["n_classes"])]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, explain_sample, feature_names=data["feature_names"],
                       class_names=data["class_names"], plot_type="bar", show=False)
    plt.title(f"Global Feature Importance - {dataset_key} (SHAP)")
    plt.tight_layout()
    summary_path = os.path.join(OUTPUT_DIR, f"shap_summary_{dataset_key}.png")
    plt.savefig(summary_path, dpi=1200)
    plt.savefig(summary_path.replace(".png", ".pdf"))
    plt.close()

    result = {"dataset_key": dataset_key, "hardest_class": hardest_class,
              "summary_plot_path": summary_path}

    idx_candidates = np.where(y_explain == target_class_idx)[0]
    if len(idx_candidates) > 0:
        idx = idx_candidates[0]
        predicted_class = int(predict_proba(explain_sample[idx:idx + 1]).argmax(axis=1)[0])
        plt.figure()
        shap.force_plot(
            explainer.expected_value[target_class_idx], shap_values[target_class_idx][idx],
            features=pd.Series(explain_sample[idx], index=data["feature_names"]).round(2),
            matplotlib=True, show=False,
        )
        plt.title(f"Local Explanation - {dataset_key}: true={hardest_class}, predicted={data['class_names'][predicted_class]}")
        plt.tight_layout()
        local_path = os.path.join(OUTPUT_DIR, f"shap_local_{dataset_key}.png")
        plt.savefig(local_path, dpi=1200, bbox_inches="tight")
        plt.savefig(local_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
        result["local_plot_path"] = local_path
        result["local_example_predicted_class"] = data["class_names"][predicted_class]

    return result


def run_adversarial_unit(unit, tracker):
    from adversarial_robustness import fgsm_attack, evaluate_robust_accuracy, PERTURBABLE_FEATURES

    dataset_key, mode = unit["dataset_key"], unit["mode"]
    data = get_prepared_dataset(dataset_key)
    perturb_mask = np.array([1.0 if f in PERTURBABLE_FEATURES else 0.0 for f in data["feature_names"]])

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        data["X"], data["y"], test_size=0.25, stratify=data["y"], random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    model = FirewallMLP(n_features=data["n_features"], n_classes=data["n_classes"], hidden_size=HIDDEN_SIZE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    X_t = torch.as_tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.as_tensor(y_train, dtype=torch.long).to(DEVICE)

    adversarial = (mode == "adversarial")
    for epoch in range(150):
        model.train()
        optimizer.zero_grad()
        logits = model(X_t)
        loss = loss_fn(logits, y_t)
        if adversarial:
            X_adv = fgsm_attack(model, loss_fn, X_train, y_train, 0.3, perturb_mask)
            X_adv_t = torch.as_tensor(X_adv, dtype=torch.float32).to(DEVICE)
            loss = 0.5 * loss + 0.5 * loss_fn(model(X_adv_t), y_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            tracker.heartbeat()
    model.eval()

    epsilon_list = [0.1, 0.3, 0.5, 1.0]
    results = evaluate_robust_accuracy(model, X_test, y_test, epsilon_list, perturb_mask, loss_fn)
    return {"dataset_key": dataset_key, "mode": mode, "results": results}


UNIT_RUNNERS = {"cv_fold": run_cv_fold_unit, "shap": run_shap_unit, "adversarial": run_adversarial_unit}


# =============================================================================
# Main loop
# =============================================================================
def run_experiment():
    ensure_checkpoint_dir(CHECKPOINT_DIR)
    tracker = ProgressTracker(log_path=PROGRESS_LOG_PATH, interval_seconds=HEARTBEAT_SECONDS)

    units = build_unit_list()
    tracker.set_total(len(units))

    already_done = sum(1 for u in units if is_unit_done(u["id"]))
    if already_done > 0:
        tracker._write_line(f"[RESUME] found {already_done}/{len(units)} units already completed "
                             f"- skipping to the remaining ones, not starting over")

    stop_event = threading.Event()

    def heartbeat_loop():
        while not stop_event.is_set():
            tracker.heartbeat()
            stop_event.wait(5)

    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()

    failed_units = []
    done_count = already_done

    try:
        for unit in units:
            if is_unit_done(unit["id"]):
                continue

            tracker.update(done_count, len(units), current_unit=unit["id"])
            tracker.heartbeat(force=True)

            try:
                runner = UNIT_RUNNERS[unit["type"]]
                result = runner(unit, tracker)
                save_unit_result(unit["id"], result)
                done_count += 1
                tracker.update(done_count, len(units))
            except Exception as e:
                err_msg = f"[ERROR] unit {unit['id']} failed: {e}\n{traceback.format_exc()}"
                tracker._write_line(err_msg)
                failed_units.append({"unit_id": unit["id"], "error": str(e)})
                continue
    except KeyboardInterrupt:
        tracker._write_line("[INTERRUPTED] user pressed Ctrl+C - all progress so far has been saved. "
                             "Run the same command again to continue immediately.")
        stop_event.set()
        return

    stop_event.set()
    tracker.heartbeat(force=True)
    tracker.finish()

    if failed_units:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(os.path.join(OUTPUT_DIR, "failed_units.json"), "w", encoding="utf-8") as f:
            json.dump(failed_units, f, ensure_ascii=False, indent=2)
        tracker._write_line(f"[WARN] {len(failed_units)} unit(s) errored - see details in "
                             f"{OUTPUT_DIR}/failed_units.json (re-run the script to retry these units)")

    summarize_results()


def print_status():
    units = build_unit_list()
    done = load_all_done_units()
    n_done = sum(1 for u in units if u["id"] in done)
    print(f"Progress: {n_done}/{len(units)} units ({n_done/len(units)*100:.1f}%)")
    by_type = {}
    for u in units:
        t = u["type"]
        by_type.setdefault(t, [0, 0])
        by_type[t][1] += 1
        if u["id"] in done:
            by_type[t][0] += 1
    for t, (d, total) in by_type.items():
        print(f"  - {t}: {d}/{total}")
    if os.path.exists(PROGRESS_LOG_PATH):
        print(f"\nSee the detailed log at {PROGRESS_LOG_PATH} (last 10 lines):")
        with open(PROGRESS_LOG_PATH, encoding="utf-8") as f:
            lines = f.readlines()
        print("".join(lines[-10:]))


def summarize_results():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done = load_all_done_units()

    cv_results = {}
    shap_results = {}
    adversarial_results = {}

    for unit_id, result in done.items():
        parts = unit_id.split("/")
        if parts[0] == "cv":
            _, dataset_key, tech, fold_str = parts
            cv_results.setdefault(dataset_key, {}).setdefault(tech, {})[fold_str] = result
        elif parts[0] == "shap":
            shap_results[parts[1]] = result
        elif parts[0] == "adversarial":
            adversarial_results.setdefault(parts[1], {})[parts[2]] = result

    summary = {"cv_summary": {}, "shap": shap_results, "adversarial": adversarial_results}

    for dataset_key, techs in cv_results.items():
        summary["cv_summary"][dataset_key] = {}
        macro_f1_by_tech = {}
        for tech, folds in techs.items():
            macro_f1s = [r["macro_f1"] for r in folds.values()]
            per_class = np.mean([r["per_class_f1"] for r in folds.values()], axis=0).tolist() if folds else []
            class_names = next(iter(folds.values()))["class_names"] if folds else []
            summary["cv_summary"][dataset_key][tech] = {
                "n_folds_completed": len(folds),
                "mean_macro_f1": float(np.mean(macro_f1s)) if macro_f1s else None,
                "std_macro_f1": float(np.std(macro_f1s)) if macro_f1s else None,
                "mean_per_class_f1": per_class, "class_names": class_names,
            }
            macro_f1_by_tech[tech] = macro_f1s

        pairwise = {}
        techs_list = list(macro_f1_by_tech.keys())
        for i, ta in enumerate(techs_list):
            for tb in techs_list[i + 1:]:
                a, b = macro_f1_by_tech[ta], macro_f1_by_tech[tb]
                if len(a) == len(b) and len(a) >= 2:
                    try:
                        stat, pval = wilcoxon(a, b)
                        pairwise[f"{ta}_vs_{tb}"] = {"statistic": float(stat), "p_value": float(pval)}
                    except ValueError:
                        pass
        summary["cv_summary"][dataset_key]["_pairwise_wilcoxon"] = pairwise

    out_path = os.path.join(OUTPUT_DIR, "final_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[SUMMARY] saved the latest summary to {out_path} (updated after every run, or call --summarize yourself)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", action="store_true", help="show current progress and exit (run no further work)")
    parser.add_argument("--summarize", action="store_true", help="combine completed units into the summary file and exit (run no further work)")
    args = parser.parse_args()

    if args.status:
        print_status()
    elif args.summarize:
        summarize_results()
    else:
        run_experiment()
