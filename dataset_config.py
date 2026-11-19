"""
dataset_config.py

A configuration file kept deliberately separate from the main code (run_full_experiment.py).

**This file is never modified by the developer, no matter how many new versions of
run_full_experiment.py are shipped to fix bugs.** Edit all paths and defaults in this one file;
it is safe from being overwritten when the other code files (run_full_experiment.py,
data_loader.py, etc.) are updated.
"""

# =============================================================================
# Real dataset paths - edit these to match your machine
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
        "feature_names_csv": "../data/UNSW_NB15/NUSW-NB15_features.csv",
    },
}

# =============================================================================
# Other experiment defaults (normally no need to change unless tuning)
# =============================================================================
TECHNIQUE_NAMES = ["baseline_smote", "cost_sensitive", "focal_loss", "hybrid_resample"]
N_FOLDS = 5
EPOCHS = 100
HIDDEN_SIZE = 200
BATCH_SIZE = 512          # mini-batch - important for datasets of a million-plus rows
LEARNING_RATE = 1e-3
HEARTBEAT_SECONDS = 60    # how often (in seconds) to report progress

SHAP_DATASETS = {         # (dataset_key -> hardest_class_name) for SHAP analysis
    "ertam_kaya": "deny",
    "unsw_nb15": "Analysis",
}
ADVERSARIAL_DATASET = "ertam_kaya"

OUTPUT_DIR = "results_final"
PROGRESS_LOG_PATH = "progress.log"
