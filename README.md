# Explainable, Cost-Sensitive Deep Learning for Extreme Class-Imbalance Firewall Log Classification

A reproducible experimental framework for classifying firewall and network-intrusion logs under
**extreme class imbalance**, where the security-critical classes often account for well under 1% of
records. The framework compares imbalance-handling techniques and model families on three real
datasets, explains predictions with SHAP, and evaluates adversarial robustness.

## What this code does

The experiments are organized around four questions:

1. **Imbalance handling** — Compares four techniques (baseline SMOTE, cost-sensitive learning,
   focal loss, hybrid SMOTE-Tomek resampling) on an identical multilayer perceptron (MLP), with
   Wilcoxon signed-rank tests and Cohen's d effect sizes.
2. **Cross-dataset generalization** — Checks whether the technique ranking holds across datasets
   with different structures and class distributions, and identifies the hardest class in each.
3. **Explainability** — Uses SHAP (global and local) to check that the model relies on
   domain-plausible features, and to profile why the hardest classes fail.
4. **Adversarial robustness** — Evaluates FGSM and multi-step PGD attacks under a domain-aware
   perturbation constraint, and measures the effect of adversarial training.

The framework additionally compares three **model families** — gradient-boosted trees (XGBoost,
LightGBM), the MLP, and a deep tabular network (TabNet) — to separate effects of the imbalance
technique from effects of the model family.

## Repository layout

```
.
├── README.md                              This file
├── README.html                            HTML version of this file
├── requirements.txt                       Python dependencies
├── dataset_config.py                      EDIT THIS: dataset paths and experiment defaults
│
│   Core pipeline modules
├── data_loader.py                         Loads and harmonizes the three datasets
├── feature_schema.py                      Canonical feature schema shared across datasets
├── imbalance_methods.py                   The four imbalance-handling techniques
├── model.py                               The MLP architecture (FirewallMLP)
├── pipeline.py                            Stratified k-fold training / evaluation loop
├── adversarial_robustness.py              FGSM attack, adversarial training, robustness eval
│
│   Experiment scripts
├── run_full_experiment.py                 Main experiment (RQ1-RQ4): trains MLP, runs SHAP + FGSM
├── tree_baselines.py                      XGBoost and LightGBM baselines (3 modes each)
├── tabnet_baseline.py                     TabNet (deep tabular) baseline (3 modes)
├── tune_xgb_datasetA.py                   Nested-CV hyperparameter tuning for XGBoost
├── make_tree_vs_mlp_table.py              Merges MLP + tree + TabNet results into one table
├── lgbm_investigation.py                  Probes why LightGBM collapses on large datasets
├── pgd_attack.py                          Stronger multi-step PGD attack (vs. single-step FGSM)
├── explanation_robustness_experiment.py   Measures SHAP drift under attack / adversarial training
├── focal_loss_gamma_sweep.py              Sweeps the focal-loss focusing parameter gamma
├── shap_hard_classes.py                   SHAP profiling of the hardest minority classes
│
├── results/                               Precomputed result files (JSON + ranked tables)
└── figures/                               Publication figures (.png at 300 DPI and .pdf)
```

## Installation

Requires Python 3.10 or newer.

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Key dependencies: PyTorch, scikit-learn, imbalanced-learn, SHAP, XGBoost, LightGBM,
pytorch-tabnet, matplotlib.

## Datasets

The datasets are **not** included in this repository (they are large and separately licensed).
Download them and point `dataset_config.py` at their locations:

| Key          | Dataset                              | Rows       | Source |
|--------------|--------------------------------------|------------|--------|
| `ertam_kaya` | Internet Firewall Data Set           | 65,532     | UCI Machine Learning Repository |
| `cicids2017` | CIC-IDS2017 (MachineLearningCVE)     | 2,830,743  | Canadian Institute for Cybersecurity |
| `unsw_nb15`  | UNSW-NB15                            | 2,540,047  | UNSW Canberra |

Edit the paths at the top of `dataset_config.py`:

```python
DATASET_PATHS = {
    "ertam_kaya": {"path": "path/to/log2.csv", "feature_names_csv": None},
    "cicids2017": {"path": "path/to/MachineLearningCVE/", "feature_names_csv": None},
    "unsw_nb15":  {"path": "path/to/UNSW_NB15/",
                   "feature_names_csv": "path/to/UNSW_NB15/NUSW-NB15_features.csv"},
}
```

All experiments read paths from this one file, so it is the only place you need to edit.

## How to run

Run scripts from the repository root (so the flat module imports resolve). Each script writes a
JSON result file and, where applicable, figures.

**Main experiment (imbalance techniques + SHAP + FGSM):**
```bash
python run_full_experiment.py
```

**Model-family baselines:**
```bash
python tree_baselines.py    --dataset ertam_kaya --path path/to/log2.csv --folds 5 --output results
python tabnet_baseline.py   --dataset ertam_kaya --path path/to/log2.csv --folds 5 --output results
python make_tree_vs_mlp_table.py --dataset ertam_kaya --results results
```

**Follow-up analyses:**
```bash
python tune_xgb_datasetA.py       --dataset ertam_kaya --path path/to/log2.csv
python pgd_attack.py              --dataset ertam_kaya --path path/to/log2.csv --steps 10
python explanation_robustness_experiment.py --dataset ertam_kaya --path path/to/log2.csv
python focal_loss_gamma_sweep.py  --dataset ertam_kaya --path path/to/log2.csv
python shap_hard_classes.py       --dataset unsw_nb15  --path path/to/unsw.csv --max_train_samples 400000
python lgbm_investigation.py      --dataset cicids2017 --path path/to/MachineLearningCVE --folds 3
```

## Reproducibility

- All experiments use `StratifiedKFold(shuffle=True, random_state=42)` and fixed global seeds.
- Each fold applies its own `StandardScaler` fitted only on the training split (no leakage).
- SMOTE is applied inside each training fold, never to the validation split.
- Result files under `results/` were produced with the configurations above and can be regenerated.

## Precomputed results

The `results/` directory contains the exact outputs used to build the figures and tables:

| File | Contents |
|------|----------|
| `ertam_kaya_tree_baseline_results.json`   | XGBoost / LightGBM, 10-fold, Dataset A |
| `ertam_kaya_tabnet_baseline_results.json` | TabNet, 5-fold, Dataset A |
| `ertam_kaya_xgb_tuned_results.json`       | Nested-CV tuned vs. fixed XGBoost |
| `pgd_attack_results.json`                 | FGSM + PGD robustness, both models |
| `explanation_robustness_results.json`     | SHAP importance under three conditions |
| `shap_hard_classes_results.json`          | SHAP profiling of Worms / Backdoors |
| `focal_loss_gamma_sweep_results.json`     | Focal-loss gamma sweep |
| `*_tree_vs_mlp_table.txt`                 | Ranked three-family comparison tables |

## Summary of findings

- **Model family matters more than the imbalance technique.** On Dataset A, class-weighted XGBoost
  reaches macro-F1 0.938 (10-fold) versus 0.736 for the best MLP configuration — a gap several times
  larger than the spread between imbalance techniques.
- **Difficulty tracks feature overlap, not rarity.** The rarest classes are often among the easiest;
  feature-overlapped attack classes stay hard regardless of frequency.
- **A three-way family ordering.** Gradient-boosted trees rank above the MLP, which ranks above the
  deep tabular network (TabNet), consistent with the broader tabular-learning literature.
- **Adversarial robustness has a real cost.** Adversarial training cuts attack success from 97.5% to
  15.5% but at a measurable clean-accuracy cost, and it shifts the model's reliance toward features
  an attacker cannot manipulate.

## License

Released under the MIT License. See `LICENSE` for details.
