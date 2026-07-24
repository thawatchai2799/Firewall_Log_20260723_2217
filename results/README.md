# Explainable, Cost-Sensitive Deep Learning for Extreme Class-Imbalance Firewall Log Classification

Code and experimental results for the manuscript *"Explainable, Cost-Sensitive Deep Learning
Framework for Extreme Class-Imbalance Firewall Log Classification: Multi-Dataset Validation with
Adversarial Robustness Testing"* by T. Chomsiri and P. Noiumkar (Mahasarakham University).

**Reviewers: start with [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)** (or `REVIEWER_GUIDE.html`).
It maps every figure and table in the paper to the file that produced it. All reported numbers
are included pre-computed in `results/`, so nothing has to be re-run to check them.

## What this study does

Four class-imbalance-handling techniques - baseline SMOTE, cost-sensitive learning, focal loss,
and hybrid SMOTE-Tomek resampling - are compared on an identical MLP across three real, public
datasets (Ertam & Kaya Internet Firewall Data Set, CIC-IDS2017, UNSW-NB15), with SHAP
explainability analysis and adversarial robustness evaluation under FGSM and PGD-10.

Headline findings:

- Resampling-based techniques outperform cost-sensitive learning and focal loss on all three
  datasets, with very large effect sizes - though five folds cap the attainable Wilcoxon p-value
  at 0.0625, so the ranking is **not statistically certified**.
- Classification difficulty tracks **feature overlap, not rarity**: Heartbleed at 0.0004% of the
  data reaches F1=76%, while less-rare SQL-injection attacks reach 0.2%.
- Adversarial training cuts attack success from 97.5% to 15.5% but costs **12.5 percentage points
  of clean accuracy** - a real trade-off, and one that survives the stronger PGD-10 attack.
- The security-critical minority classes remain essentially unclassifiable throughout. This
  negative result is reported deliberately; see Section VII-D of the paper.

## Quick start

```bash
pip install -r requirements.txt
# Download the three datasets (links in REVIEWER_GUIDE.md section 3),
# then edit DATASET_PATHS in dataset_config.py to match your machine.
run_all.bat            # Windows
./run_all.sh           # Linux / macOS
```

`dataset_config.py` is the only file you should need to edit. Every script reads its paths and
defaults from it.

Stage 1 is long and **resumable** - re-run the same command after an interruption and it resumes
from the last completed unit.

## Repository contents

| File | Purpose |
|---|---|
| `run_full_experiment.py` | Main orchestrator, RQ1-RQ4 |
| `dataset_config.py` | Paths and defaults (edit this) |
| `data_loader.py`, `feature_schema.py` | Dataset loading and canonical feature mapping |
| `model.py`, `imbalance_methods.py` | MLP, focal loss, the four compared techniques |
| `adversarial_robustness.py` | FGSM attack and adversarial training |
| `checkpoint_utils.py` | Crash-safe checkpointing |
| `explanation_robustness_experiment.py` | Explanation drift, Section VI-E, Fig. 10, Table 9 |
| `focal_loss_gamma_sweep.py` | Follow-up experiment, Section VI-A |
| `pgd_attack.py` | Follow-up experiment, Section VI-D and Fig. 9 |
| `shap_hard_classes.py` | Follow-up experiment, Section VI-C3 and Fig. 8 |
| `results/` | All pre-computed outputs cited by the paper |

## Datasets

Not redistributed here. All three are public - see section 3 of the Reviewer Guide for links and
for the exact spelling of the UNSW-NB15 feature-name file.

## Citation

Please cite the published version once available. Corresponding author: Preecha Noiumkar
(preecha.n@msu.ac.th).
