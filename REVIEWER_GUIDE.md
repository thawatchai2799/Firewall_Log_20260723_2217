# Reviewer Guide

Companion repository for the manuscript

> **Explainable, Cost-Sensitive Deep Learning Framework for Extreme Class-Imbalance Firewall Log
> Classification: Multi-Dataset Validation with Adversarial Robustness Testing**
> T. Chomsiri and P. Noiumkar, Mahasarakham University, Thailand.

**You do not need to run anything to check the numbers in the paper.** Every value reported in
the manuscript is included pre-computed under `results/`. Re-running is only necessary to
reproduce those numbers from the raw data or to extend the experiments.

Section, figure, and table numbers below follow the IEEE-format manuscript. If you are reading
the Springer-format version, the mapping is: Section I = 1, II = 2, ... VIII = 8, and
subsection VI-C3 = 6.3.3, and so on. Figure and table numbers are identical in both.

---

## 1. Repository layout

```
├── run_full_experiment.py       Main resumable orchestrator (RQ1-RQ4)
├── dataset_config.py            Dataset paths and all defaults - EDIT THIS FIRST
├── data_loader.py               Loads and cleans all three datasets into a common schema
├── feature_schema.py            Canonical feature mapping table for each dataset
├── model.py                     MLP architecture, focal loss, class-weight utilities
├── imbalance_methods.py         The four compared imbalance-handling techniques
├── adversarial_robustness.py    FGSM attack, adversarial training, robustness evaluation
├── checkpoint_utils.py          Crash-safe checkpointing and progress logging
├── pipeline.py                  Standalone single-dataset script (used during development)
├── focal_loss_gamma_sweep.py    Follow-up: focusing-parameter sweep      -> Section VI-A
├── pgd_attack.py                Follow-up: PGD-10 vs FGSM                -> Section VI-D, Fig. 9
├── shap_hard_classes.py         Follow-up: SHAP for Worms and Backdoors  -> Section VI-C3, Fig. 8
├── run_all.bat / run_all.sh     Runs all four stages in order
├── requirements.txt             Python dependencies
├── REVIEWER_GUIDE.md / .html    This document
└── results/                     Pre-computed outputs referenced by the manuscript
```

## 2. Where each reported number comes from

| Manuscript item | Source file in `results/` |
|---|---|
| Tables 5-8, Figs. 1-3 | `final_summary.json` |
| Fig. 4 (SHAP global, Dataset A) | `shap_summary_ertam_kaya.png` |
| Fig. 5 (SHAP local, deny) | `shap_local_ertam_kaya.png` |
| Fig. 6 (SHAP global, Dataset C) | `shap_summary_unsw_nb15.png` |
| Fig. 7 (SHAP local, Analysis) | `shap_local_unsw_nb15.png` |
| Fig. 8 (SHAP, Worms + Backdoors) | `fig_shap_hard_classes.png`, built from `shap_summary_C_worms.png` and `shap_summary_C_backdoors.png` |
| Section VI-C3 numbers | `shap_hard_classes_results.json` |
| Fig. 9 (FGSM vs PGD-10) | `fig_pgd_vs_fgsm.png` |
| Section VI-D PGD numbers | `pgd_attack_results.json` |
| Fig. 10, Table 9 (explanation drift) | `explanation_robustness_results.json` |
| Section VI-A gamma sweep | `focal_loss_gamma_sweep_results.json`, `fig_focal_loss_gamma_sweep.png` |

## 3. Datasets

None of the three datasets is redistributed here. All are public:

- **Dataset A** - Ertam & Kaya Internet Firewall Data Set (65,532 rows):
  https://www.kaggle.com/datasets/tunguz/internet-firewall-data-set
- **Dataset B** - CIC-IDS2017 (2,830,743 rows):
  https://www.unb.ca/cic/datasets/ids-2017.html
- **Dataset C** - UNSW-NB15 (2,540,047 rows):
  https://research.unsw.edu.au/projects/unsw-nb15-dataset

After downloading, point `DATASET_PATHS` in `dataset_config.py` at your local copies. Note that
the UNSW-NB15 feature-name file is officially spelled `NUSW-NB15_features.csv`.

## 4. Reproducing the results

```bash
pip install -r requirements.txt
# edit dataset_config.py so DATASET_PATHS matches your machine
run_all.bat          # Windows;  ./run_all.sh on Linux/macOS
```

Individual stages can be run alone: `run_all.bat main | focal | pgd | shap`.

Stage 1 (`run_full_experiment.py`) is the long one and is **resumable**: it writes a checkpoint
after each unit of work (one fold x one technique x one dataset), so an interrupted run can be
restarted with the same command and will skip everything already completed.

### Protocol notes that matter when comparing runs

- **Folds.** All cross-validation uses `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
  The gamma sweep reuses exactly these folds so its numbers are directly comparable with Table 5.
- **Epoch counts differ by design.** The RQ1-RQ3 experiments train for `EPOCHS=100`
  (`dataset_config.py`). The adversarial experiments in Section VI-D and `pgd_attack.py` train for
  **150** epochs, to match the original FGSM protocol reported in Table 8. This is intentional and
  is set by `ADVERSARIAL_EPOCHS`.
- **FGSM and PGD share their models.** `pgd_attack.py` trains the standard and adversarially
  trained models once per run and evaluates both attacks on those same two models, so the
  FGSM-versus-PGD comparison carries no cross-run training variance. Global seed is 42.
- **The Section VI-C3 SHAP model is not an RQ1 fold model.** It is trained separately on a
  stratified 400,000-row subsample (minority classes retained in full). It is used for explanatory
  analysis only; no per-class score in the manuscript comes from it. See Section VII-C.
- **Section VI-E was run before seeding was unified**, which is why its clean-accuracy cost is
  23.3 points against 12.5 in Section VI-D. This is stated as a limitation in Section VII-C. The
  seeded PGD re-run reproduced the Table 8 FGSM figures within 1-2 points, which localises the
  variance to the Section VI-E retraining rather than the protocol.

## 5. The three follow-up experiments

### 5.1 Focal loss gamma sweep -> Section VI-A

Sweeps gamma over {0.5, 1.0, 2.0, 5.0} on Dataset A using the same folds as the main experiment.
Reported in text only; the figure is provided here but is not part of the manuscript.

```bash
python focal_loss_gamma_sweep.py --dataset ertam_kaya
```

Result: the best value (gamma=1.0, macro-F1 0.7082) does not significantly beat the default
gamma=2.0 (0.6966; Wilcoxon p=0.3125, Cohen's d=1.19) and no gamma improves the reset-both class
beyond F1=0.019. At gamma=0.5 the model collapses to the majority class in all five folds
(macro-F1 0.1824, std 4e-06). This is what supports the claim that focal loss's weakness here is
inherent rather than a tuning artifact.

### 5.2 PGD-10 -> Sections IV-F and VI-D, Fig. 9

Madry-style PGD: 10 steps, random start inside the epsilon ball, step size 2.5*epsilon/10,
projection after every step, and the same perturbation mask as FGSM (port fields locked).

```bash
python pgd_attack.py
```

Result: against the standard model PGD adds essentially nothing (robust accuracy within 0.4
points of FGSM at every epsilon) because the single-step attack already compromises it. Against
the adversarially trained model PGD is modestly stronger (attack success 15.6-20.3% versus
10.7-16.1%) but robust accuracy stays in the same flat regime. The `protocol` block of
`pgd_attack_results.json` records every setting used.

### 5.3 SHAP for the hardest Dataset C classes -> Section VI-C3, Fig. 8

Protocol identical to Section IV-E: baseline-SMOTE model, KernelExplainer, 50-sample background,
up to 100 explained test samples per class (all 44 available for Worms).

```bash
python shap_hard_classes.py --max_train_samples 400000
```

Result: two different failure modes. Backdoors depends almost entirely on `protocol`
(mean|SHAP| = 0.0367, about 24x the next feature) - an insufficient shortcut. Worms has no
feature that moves the output at all (every mean|SHAP| on the order of 1e-4). Both classes were
misclassified in every explained sample, predominantly as Exploits.

If the script exits with "Target class not found", it prints the available class names; re-run
with `--targets` set to the exact spellings in your copy of the data.

## 6. Known deviations and limitations

These are stated in Section VII-C of the manuscript and repeated here so that nothing in this
repository is more reassuring than the paper itself:

- The technique ranking is **not statistically certified**. With five folds the minimum attainable
  two-sided Wilcoxon p-value is 0.0625, so no comparison crosses alpha=0.05 despite Cohen's d up
  to 11.5.
- **A single MLP architecture is used throughout.** No comparison against gradient-boosted trees
  (XGBoost, LightGBM) or tabular deep architectures is included, so the ranking is conditional on
  this backbone.
- Adversarial evaluation covers **Dataset A only**, **white-box attacks only** (FGSM and PGD-10),
  and a **single adversarial-training budget** (epsilon_train=0.3). Results are point estimates
  from single runs, without confidence intervals.
- SHAP results are **descriptive, not inferential**: at most 100 explained instances per class
  against a 50-sample background, with no stability analysis across draws.
- Only F1 is reported per class; precision and recall are not broken out.
- Computational cost (training time, memory, SMOTE overhead) is not measured.
- ADASYN, Borderline-SMOTE, and generative resampling are discussed but not evaluated.
- The perturbable/locked feature split is a **threat-model assumption**, not a measured property.

## 7. Contact

Corresponding author: Preecha Noiumkar, preecha.n@msu.ac.th
