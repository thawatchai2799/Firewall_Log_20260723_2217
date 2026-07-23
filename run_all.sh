#!/usr/bin/env bash
# ===========================================================================
# run_all.sh - Linux/macOS runner for the full experiment pipeline
#
# Usage:
#     ./run_all.sh          Run every stage in order (1 -> 4)
#     ./run_all.sh main     Stage 1 only: main RQ1-RQ4 experiment
#     ./run_all.sh focal    Stage 2 only: focal loss gamma sweep
#     ./run_all.sh pgd      Stage 3 only: PGD-10 vs FGSM
#     ./run_all.sh shap     Stage 4 only: SHAP for Worms and Backdoors
#
# All dataset paths come from DATASET_PATHS in dataset_config.py.
# Stage 1 is resumable: re-running it skips units that already have checkpoints.
# ===========================================================================
set -euo pipefail

check_env() {
    echo "[check] Python and required packages"
    python3 -c "import torch, sklearn, imblearn, shap, scipy, matplotlib, pandas, numpy" \
        || { echo "ERROR: missing packages. Run: pip install -r requirements.txt"; exit 1; }
    python3 -c "import dataset_config" \
        || { echo "ERROR: dataset_config.py not found or has a syntax error."; exit 1; }
    echo "[check] OK"
}

stage_main()  { echo; echo "=== Stage 1/4: main experiment (RQ1-RQ4) ==="; python3 run_full_experiment.py; }
stage_focal() { echo; echo "=== Stage 2/4: focal loss gamma sweep ===";    python3 focal_loss_gamma_sweep.py --dataset ertam_kaya; }
stage_pgd()   { echo; echo "=== Stage 3/4: PGD-10 vs FGSM ===";            python3 pgd_attack.py; }
stage_shap()  { echo; echo "=== Stage 4/4: SHAP Worms/Backdoors ===";      python3 shap_hard_classes.py --max_train_samples 400000; }

case "${1:-all}" in
    all)   check_env; stage_main; stage_focal; stage_pgd; stage_shap
           echo; echo "All four stages completed. Collect:"
           echo "  results_final/final_summary.json      (Tables 5-8, Figs 1-3)"
           echo "  focal_loss_sweep_results/             (Section VI-A)"
           echo "  pgd_results/                          (Section VI-D, Fig. 9)"
           echo "  shap_hard_classes_results/            (Section VI-C3, Fig. 8)" ;;
    main)  stage_main ;;
    focal) stage_focal ;;
    pgd)   stage_pgd ;;
    shap)  stage_shap ;;
    *)     echo "Unknown stage '$1'. Valid: main, focal, pgd, shap"; exit 1 ;;
esac
