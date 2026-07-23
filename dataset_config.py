"""
dataset_config.py

This configuration file is deliberately kept separate from the main experiment code
(run_full_experiment.py and the follow-up experiment scripts).

**This file is never overwritten when a new version of any other script is released.**
Edit all machine-specific paths and default settings here, in this one file, so that
updating run_full_experiment.py, data_loader.py, pgd_attack.py, etc. can never clobber
your local configuration.

All scripts in this repository read their dataset paths from DATASET_PATHS below, so the
--path and --feature_names_csv command-line arguments are optional.
"""

# =============================================================================
# Real dataset paths - EDIT THESE TO MATCH YOUR MACHINE
# =============================================================================
DATASET_PATHS = {
    "ertam_kaya": {
        "path": "../data/Internet_Firewall_DataSet/log2.csv",
        "feature_names_csv": None,
    },
    "cicids2017": {
        "path": "../data/CIC_IDS2017/MachineLearningCVE/",
        "feature_names_csv": None,
    },
    "unsw_nb15": {
        "path": "../data/UNSW_NB15/",
        # Note: the official file name is spelled "NUSW-NB15_features.csv" (not UNSW).
        "feature_names_csv": "../data/UNSW_NB15/NUSW-NB15_features.csv",
    },
}

# =============================================================================
# Main experiment defaults (normally no need to change)
# =============================================================================
TECHNIQUE_NAMES = ["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"]
N_FOLDS = 5
EPOCHS = 100
HIDDEN_SIZE = 200
BATCH_SIZE = 512          # mini-batch - essential for datasets of a million rows or more
LEARNING_RATE = 1e-3
HEARTBEAT_SECONDS = 60    # progress report interval, in seconds

# (dataset_key -> target class name) for the SHAP analysis in the main experiment.
# Reported in Section VI-C1 (deny) and Section VI-C2 (Analysis) of the manuscript.
# The two hardest Dataset C classes (Worms, Backdoors) reported in Section VI-C3 are
# produced by the separate script shap_hard_classes.py, not by this entry.
SHAP_DATASETS = {
    "ertam_kaya": "deny",
    "unsw_nb15": "Analysis",
}

ADVERSARIAL_DATASET = "ertam_kaya"

OUTPUT_DIR = "results_final"
PROGRESS_LOG_PATH = "progress.log"

# =============================================================================
# Follow-up experiment defaults (added for manuscript v08 and later)
# =============================================================================

# focal_loss_gamma_sweep.py - Section VI-A, closing paragraph
FOCAL_GAMMA_VALUES = [0.5, 1.0, 2.0, 5.0]   # 2.0 is the default used in the main results
FOCAL_SWEEP_DATASET = "ertam_kaya"
FOCAL_SWEEP_OUTPUT = "focal_loss_sweep_results"

# pgd_attack.py - Sections IV-F and VI-D, Fig. 9
# NOTE: the adversarial experiments train for 150 epochs to match the original
# Section VI-D FGSM protocol exactly, NOT the EPOCHS=100 used by the RQ1
# cross-validation experiments above. This difference is intentional.
ADVERSARIAL_EPOCHS = 150
ADVERSARIAL_EPSILON_LIST = [0.1, 0.3, 0.5, 1.0]
ADVERSARIAL_TRAIN_EPSILON = 0.3
PGD_STEPS = 10                              # PGD-10, following Madry et al.
PGD_OUTPUT = "pgd_results"
GLOBAL_SEED = 42

# shap_hard_classes.py - Section VI-C3, Fig. 8
SHAP_HARD_DATASET = "unsw_nb15"
SHAP_HARD_TARGETS = ["Worms", "Backdoors"]
SHAP_BACKGROUND_SIZE = 50
SHAP_MAX_EXPLAINED = 100
# Stratified cap on training rows for the explained model; minority classes are always
# kept in full and only the largest classes are downsampled. Set to 0 to use all rows.
SHAP_MAX_TRAIN_SAMPLES = 400000
SHAP_HARD_OUTPUT = "shap_hard_classes_results"
