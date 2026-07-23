# Before you push: what is in this package and what you must add

This package is **not a complete repository on its own.** It contains everything that was
produced or corrected in the assistant's environment, but the core pipeline files live only on
your machine and could not be included from here. Copy them in before pushing.

Delete this file after you have gone through it &mdash; it is a working checklist, not part of the
published repository.

---

## 1. Files included in this package

| File | Status |
|---|---|
| `dataset_config.py` | **Corrected** &mdash; see &sect;3 below |
| `pgd_attack.py` | New (Section VI-D, Fig. 9) |
| `shap_hard_classes.py` | New (Section VI-C3, Fig. 8) |
| `run_all.bat`, `run_all.sh` | New &mdash; runs all four stages in order |
| `requirements.txt` | Rewritten |
| `README.md` | New |
| `REVIEWER_GUIDE.md`, `REVIEWER_GUIDE.html` | New, consistent with manuscript v11 |
| `results/` (10 files) | Real experimental outputs from your runs |

## 2. Files you must copy in from your machine

These nine live only on your machine. The Reviewer Guide and `run_all.bat` both reference them,
so the repository is broken until they are added:

```
run_full_experiment.py
data_loader.py
feature_schema.py
model.py
imbalance_methods.py
adversarial_robustness.py
checkpoint_utils.py
pipeline.py
focal_loss_gamma_sweep.py
```

**Do not copy your local `dataset_config.py`** &mdash; use the corrected one in this package
(reason in &sect;3), then re-enter your own paths in `DATASET_PATHS` if they differ.

### Two checks before copying

1. **No Thai text in any file.** The repository is English-only. Quick check from the repo folder:

   ```bat
   findstr /R /C:"[^\x00-\x7F]" *.py
   ```

   ```bash
   grep -P '[\x{0E00}-\x{0E7F}]' *.py
   ```

   Both should return nothing. Several pipeline files were translated during development, but the
   copies on your machine are the ones that will be published, so verify them rather than assuming.

2. **Two bug fixes must be present in `adversarial_robustness.py`.** Both were made during
   development and both are easy to lose if an older copy gets pushed:
   - `fgsm_attack()` and `evaluate_robust_accuracy()` must be **device-aware** &mdash; they should
     start with `device = next(model.parameters()).device` and call `.cpu()` before `.numpy()`.
     Without this the code crashes on a GPU machine.
   - `train_model()` must use **mini-batch** training (`BATCH_SIZE` from `dataset_config.py`).
     Without this, the million-row datasets exhaust memory.

   Grep for `next(model.parameters()).device` and for `BATCH_SIZE` to confirm.

## 3. Why `dataset_config.py` was changed

Your uploaded copy had **nine lines of Thai comments**, which breaks the English-only rule for the
repository. It also predates the three follow-up experiments, so it had no settings for them.

The corrected file:

- translates every comment to English, **changing no values**
  (`N_FOLDS=5`, `EPOCHS=100`, `HIDDEN_SIZE=200`, `BATCH_SIZE=512`, `LEARNING_RATE=1e-3`,
  `HEARTBEAT_SECONDS=60`, `OUTPUT_DIR`, `PROGRESS_LOG_PATH`, `ADVERSARIAL_DATASET`, and all three
  dataset paths are byte-identical to yours);
- adds a block of settings for the three follow-up experiments;
- documents the deliberate epoch mismatch: RQ1&ndash;RQ3 use 100 epochs, adversarial experiments use
  150 to match the Table 8 protocol. Anyone reading the config would otherwise assume this was a bug;
- notes that the UNSW-NB15 feature file is genuinely spelled `NUSW-NB15_features.csv`, so nobody
  "corrects" it later;
- clarifies that `SHAP_DATASETS` covers Sections VI-C1 and VI-C2 only, and that the Worms and
  Backdoors analysis in Section VI-C3 comes from `shap_hard_classes.py`.

## 4. Result files still missing from `results/`

The ten files in `results/` are the ones that were uploaded during this work. The Reviewer Guide
also references these, which are on your machine from the main experiment run:

```
final_summary.json                  <- Tables 5-8 and Figs. 1-3 derive from this
explanation_robustness_results.json <- Fig. 10 and Table 9
shap_summary_ertam_kaya.png         <- Fig. 4
shap_local_ertam_kaya.png           <- Fig. 5
shap_summary_unsw_nb15.png          <- Fig. 6
shap_local_unsw_nb15.png            <- Fig. 7
fig_rq1_dataset_A_real.png          <- Fig. 1
fig_rq2_cross_dataset_real.png      <- Fig. 2
fig_rq2_hardest_class_real.png      <- Fig. 3
```

If any filename on your machine differs, fix the mapping table in **&sect;2 of REVIEWER_GUIDE.md and
in REVIEWER_GUIDE.html** rather than renaming the files &mdash; a reviewer following a broken
mapping is worse than an unfamiliar filename.

## 5. Final sequence

```bat
git init
git add .
git commit -m "Pipeline, follow-up experiments, and results for the firewall log classification study"
git branch -M main
git remote add origin https://github.com/thawatchai2799/Firewall_Log_20260723_2217.git
git push -u origin main
```

Then open the repository page in a private browser window and confirm:

- the file list is complete and `README.md` renders;
- `REVIEWER_GUIDE.md` renders and its tables are intact;
- `results/` contains every file referenced in &sect;2 of the guide;
- this checklist file has been deleted.

The repository URL is printed in the manuscript's data-availability statement, so a reviewer will
open it. As of the last check it was still empty.
