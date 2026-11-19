# Extreme Class-Imbalance Firewall Log Classification: Measuring and Manipulating Class Overlap and Rarity Across Three Datasets

Code, precomputed results and figure scripts for the paper of the same title. The study asks what
makes a rare class hard to classify in firewall and intrusion logs, class rarity or class overlap
in feature space, and how the answer depends on the model. It uses three public datasets (the
Ertam & Kaya Internet Firewall Data Set, CIC-IDS2017 and UNSW-NB15), ten model–technique
combinations trained on identical folds with fixed seeds, five data-complexity measures for all
29 classes, controlled rarity and overlap sweeps, two protocol checks (per-class time-block folds
and native feature sets), exact TreeSHAP attribution profiles, and a calibration analysis.

Every number in the paper can be traced to a file in `results/`, and every figure to a script in
`figures/` that reads only files in that folder. Nothing has to be re-trained to check the
reported values; the sections below say how to re-run each experiment if you want to.

## Repository layout

```
.
├── README.md
├── LICENSE
├── requirements.txt                  Python dependencies (minimum versions)
├── dataset_config.py                 EDIT THIS: dataset paths and training defaults
│
│   Shared modules (imported by the experiment scripts; run nothing on their own)
├── data_loader.py                    Loads the three datasets and maps them onto the common schema
├── feature_schema.py                 The common flow-feature schema (9 / 8 / 10 features for A / B / C)
├── imbalance_methods.py              Capped SMOTE, cost-sensitive loss, focal loss, hybrid resampling
├── model.py                          The MLP (two hidden layers of 200 units)
├── pipeline.py                       Stratified k-fold training loop used by the original MLP runs
├── tree_baselines.py                 XGBoost and LightGBM in the plain / SMOTE / class-weighted regimes
├── run_full_experiment.py            Original MLP training loop (imported by focal_loss_gamma_sweep.py)
│
│   Experiments reported in the paper (in the order of the Results section)
├── overlap_analysis.py               Section 5.1: N1, N2, N3, R, F1inv for every class            -> results/overlap/
├── rerun_with_predictions.py         Sections 5.5-5.7: seeded re-run of all 10 combinations,
│                                     predictions and probabilities saved                            -> results/main_comparison/
├── paired_tests_from_rerun.py        Sections 4.3, 5.5, 5.9, 5.10: paired bootstrap / permutation tests
│                                     of XGBoost against the MLP under each protocol                -> results/main_comparison/paired_tests.csv
├── controlled_experiment.py          Sections 5.3-5.4: rarity sweep and overlap sweep (Dataset A)   -> results/controlled/
├── run_extensions.py                 Sections 5.9-5.11: time-block folds, native feature sets,
│   ├── evaluation_extensions.py        exact TreeSHAP profiles (one command, resumable)              -> results/protocol_checks/, results/shap/
│   └── shap_overlap_profiles.py
├── overlap_native_features.py        Section 5.10: overlap measures recomputed on native features   -> results/overlap_native/
├── calibration_from_npz.py           Section 5.12: ECE, Brier, NLL, reliability diagrams            -> results/calibration/
│
│   Auxiliary analyses (reported from their original runs; see Section 6.3 of the paper)
├── tune_xgb_datasetA.py              Nested randomised search for XGBoost on Dataset A              -> results/auxiliary/
├── focal_loss_gamma_sweep.py         Focal-loss gamma sweep on Dataset A                            -> results/auxiliary/
├── lgbm_investigation.py             Section 5.8: twelve-configuration LightGBM probe on Dataset B  -> results/auxiliary/
├── collect_results.py                Fold tables and paired tests of the original (unseeded) run    -> results/original_run/
│
│   Scripts for analyses that are no longer in the paper (kept for the record; see below)
├── adversarial_robustness.py, pgd_attack.py, explanation_robustness_experiment.py,
├── tabnet_baseline.py, shap_hard_classes.py, make_tree_vs_mlp_table.py
│
├── results/                          Precomputed results behind every table (see "Results" below)
└── figures/                          Figure scripts, their input files and the rendered figures
```

All scripts import the shared modules by their bare names, so run every script from the
repository root.

## Installation

Python 3.10 or newer. A CUDA GPU is used by the MLP when available; XGBoost and LightGBM run on
the CPU or GPU according to their own builds.

```bash
python -m venv venv
source venv/bin/activate            # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` gives the minimum versions. The exact versions used for the reported runs are
listed in `requirements-frozen.txt` when present (generated with `pip freeze` on the experiment
machine).

## Datasets

The datasets are not distributed with this repository. Download them and point
`dataset_config.py` at their locations; every script reads the paths from that one file.

| Key          | Dataset                                   | Records    | Classes | Source |
|--------------|-------------------------------------------|------------|---------|--------|
| `ertam_kaya` | Internet Firewall Data Set (`log2.csv`)   | 65,532     | 4       | UCI Machine Learning Repository / Kaggle |
| `cicids2017` | CIC-IDS2017, MachineLearningCVE CSV files | 2,830,743  | 15      | Canadian Institute for Cybersecurity |
| `unsw_nb15`  | UNSW-NB15, the four `UNSW-NB15_*.csv` files and `NUSW-NB15_features.csv` | 2,540,047 | 10 | UNSW Canberra Cyber |

```python
DATASET_PATHS = {
    "ertam_kaya": {"path": "path/to/log2.csv", "feature_names_csv": None},
    "cicids2017": {"path": "path/to/MachineLearningCVE/", "feature_names_csv": None},
    "unsw_nb15":  {"path": "path/to/UNSW_NB15/",
                   "feature_names_csv": "path/to/UNSW_NB15/NUSW-NB15_features.csv"},
}
```

Two label corrections are applied on loading and are described in the paper: the UNSW-NB15
labels "Backdoor" and "Backdoors" are merged into one class, and the mis-encoded dash in the
CIC-IDS2017 "Web Attack" labels is treated as cosmetic.

## Reproducing the results

Each command below writes into its own output folder and never overwrites an earlier run; the
long-running scripts print progress every 30 seconds with an estimated finishing time and can be
resumed after an interruption.

**Main comparison (Tables 6-9, Table 15; about 14 GPU-hours on the hardware below):**
```bash
python rerun_with_predictions.py --output rerun_results
python rerun_with_predictions.py --output rerun_results --summarize      # writes rerun_metrics.csv, rerun_per_class.csv, rerun_confusion_matrices.json
python paired_tests_from_rerun.py --output results/main_comparison/paired_tests.csv
```
Dataset A is run with seeds 42, 43 and 44 and Datasets B and C with seed 42 (`--seeds` controls
this); every unit saves its predicted labels and class probabilities as a `.npz` file.

**Overlap measures (Table 3, Fig. 1; about 20 minutes):**
```bash
python overlap_analysis.py --all --output overlap_results
```

**Controlled experiments (Table 5, Figs. 2-3; about 1.5 hours):**
```bash
python controlled_experiment.py --dataset ertam_kaya --output controlled_results
```

**Protocol checks and SHAP profiles (Tables 10-13, Figs. 4-5; about 8 hours):**
```bash
python run_extensions.py --rerun_dir rerun_results --output extension_results
python run_extensions.py --rerun_dir rerun_results --output extension_results --resume latest   # after an interruption
python overlap_native_features.py --output overlap_native_results
```
`run_extensions.py` runs three stages (`shap`, `time_blocks`, `full_features`) for the three
headline models (class-weighted XGBoost, SMOTE XGBoost, SMOTE MLP) with seed 42 on the same folds
as the main comparison; `--stages` selects a subset. `overlap_native_features.py` recomputes the
five overlap measures on each dataset's native feature set using the same stratified draws as
`overlap_analysis.py`.

**Calibration (Table 14, Fig. 6; a few minutes):**
```bash
python calibration_from_npz.py --input rerun_results --output calibration_results
```

**Auxiliary analyses:**
```bash
python tune_xgb_datasetA.py      --path path/to/log2.csv --output results
python focal_loss_gamma_sweep.py --dataset ertam_kaya --path path/to/log2.csv
python lgbm_investigation.py     --dataset cicids2017 --path path/to/MachineLearningCVE --folds 3 --output results
```

**Self-tests.** `run_extensions.py`, `overlap_native_features.py`, `shap_overlap_profiles.py` and
`calibration_from_npz.py` accept `--selftest`, which runs the script end to end on a small
synthetic dataset and checks its own output. `rerun_with_predictions.py --synthetic` does the same
for the main loop. These finish in a few minutes and need no real data.

## Results

`results/` holds the files from which every table in the paper was produced.

| Folder | Contents | Paper |
|---|---|---|
| `results/main_comparison/` | `rerun_metrics.csv` (one row per unit: 250 units = 10 combinations × 5 folds × 1-3 seeds), `rerun_per_class.csv` (per-class precision, recall, F1, support, FP, FPR for every unit), `rerun_confusion_matrices.json`, `paired_tests.csv` (output of `paired_tests_from_rerun.py`) | Tables 6-9, 15; Sections 5.5, 5.9, 5.10 |
| `results/original_run/` | fold scores and paired tests of the original, unseeded run that preceded the re-run (`collect_results.py` output); kept because Section 5.13 compares the two | Section 5.13 |
| `results/overlap/` | `*_overlap_measures.csv` and `*_overlap_per_draw.json` per dataset; `table_overlap_measures_v2.csv` (29 classes with per-class F1 attached) | Table 3, Table 4, Fig. 1 |
| `results/controlled/` | `table_overlap_controlled.csv` (the 72 overlap-sweep runs reported: the `drop` rows of run 1 and the `deny` rows of run 2), `controlled_run1_all_classes_*` (first run over all four classes; its `deny` sweep used `reset-both` as the blend partner, which the paper explains is not a valid partner), `controlled_run2_deny_*` (re-run of `deny` with `allow` as partner, after the minimum partner-pool rule was added), `table_rarity_sweep_from_log.csv` (rarity sweep; transcribed from the run's console log, as noted in its `source` column) | Table 5, Figs. 2-3 |
| `results/protocol_checks/time_blocks/` | `metrics.csv`, `per_class.csv`, `confusion_matrices.json`, `*_split_info.json` (the ordering key used for each dataset) | Table 10, Fig. 4 |
| `results/protocol_checks/full_features/` | `metrics.csv`, `per_class.csv`, `confusion_matrices.json`, `*_feature_info.json` (which columns were used, dropped or imputed) | Tables 10-11, Fig. 4 |
| `results/overlap_native/` | overlap measures on the common and the native feature set, same draws | Table 12 |
| `results/shap/` | per-class importance profiles, class summaries (attribution margin, closest rival, top feature), bootstrap stability, pair similarities, `*_shap_model_check.json` (additivity error and the match with the main run's fold-0 score), `shap_merged_29.csv` (joined with overlap and F1) | Table 13, Fig. 5 |
| `results/calibration/` | `calibration_summary.csv`, `calibration_per_unit.csv`, `calibration_per_class.csv`, `calibration_reliability_bins.csv` | Table 14, Fig. 6 |
| `results/auxiliary/` | XGBoost tuning check, focal-loss gamma sweep, LightGBM configuration probe | Sections 4.1, 5.8 |

The per-unit prediction and probability files (`.npz`, several gigabytes in total) are not stored
in the repository. They are regenerated by `rerun_with_predictions.py`, and every summary file
above was computed from them by the scripts listed in the layout.

## Figures

`figures/` contains one script per figure. Each script reads only the CSV files that sit beside
it (copies of the corresponding files in `results/`) and writes a `.png` and a `.pdf` at 600 DPI:

| Script | Figure |
|---|---|
| `fig_difficulty_vs_rarity_overlap.py` | Fig. 1: per-class F1 against class size and against composite overlap |
| `fig_rarity_sweep.py` | Fig. 2: controlled rarity sweep |
| `fig_overlap_controlled.py` | Fig. 3: controlled overlap sweep |
| `fig_protocol_robustness.py` | Fig. 4: macro-F1 under random folds, time-block folds and native features |
| `fig_shap_margin.py` | Fig. 5: SHAP attribution margin against overlap and class size |
| `fig_reliability.py` | Fig. 6: reliability diagrams |

```bash
cd figures && python fig_protocol_robustness.py
```

## Hardware and run times

All reported runs were made on one laptop: NVIDIA GeForce RTX 3050 (4 GB), Intel Core i5-11300H,
16 GB RAM, Windows 11. The 250-unit main comparison took 14.3 GPU-hours, the controlled
experiments about 1.5 hours, the protocol checks and SHAP analysis about 8 hours, the overlap
measures about 20 minutes and the calibration analysis a few minutes.

XGBoost results are deterministic and reproduce exactly between runs. LightGBM on the two large
datasets varies by up to 0.12 macro-F1 between runs because its multithreaded training is not
deterministic by default; all such variation lies within the collapsed regime reported in the
paper. The MLP is seeded (Python, NumPy, PyTorch on CPU and GPU, cuDNN deterministic mode).

## Scripts for analyses no longer in the paper

`adversarial_robustness.py`, `pgd_attack.py`, `explanation_robustness_experiment.py`,
`shap_hard_classes.py`, `tabnet_baseline.py` and `make_tree_vs_mlp_table.py` belong to analyses
(adversarial robustness, SHAP drift, a TabNet baseline) that were removed from the paper and are
not reported. They are kept so that the history of the code is complete, and because
`rerun_with_predictions.py` optionally imports `tabnet_baseline.py`. They are not maintained.

## License

MIT, see `LICENSE`.
