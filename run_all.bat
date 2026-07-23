@echo off
REM ===========================================================================
REM run_all.bat - Windows runner for the full experiment pipeline
REM
REM Usage:
REM     run_all.bat            Run every stage in order (1 -> 4)
REM     run_all.bat main       Stage 1 only: main RQ1-RQ4 experiment
REM     run_all.bat focal      Stage 2 only: focal loss gamma sweep
REM     run_all.bat pgd        Stage 3 only: PGD-10 vs FGSM
REM     run_all.bat shap       Stage 4 only: SHAP for Worms and Backdoors
REM
REM All dataset paths come from DATASET_PATHS in dataset_config.py, so no path
REM arguments are needed here. Edit dataset_config.py before the first run.
REM
REM Stage 1 is resumable: if it is interrupted, run it again and it will skip
REM every unit that already has a checkpoint file.
REM ===========================================================================

setlocal

if "%~1"=="" goto ALL
if /I "%~1"=="main"  goto MAIN
if /I "%~1"=="focal" goto FOCAL
if /I "%~1"=="pgd"   goto PGD
if /I "%~1"=="shap"  goto SHAP
echo Unknown stage "%~1". Valid values: main, focal, pgd, shap
exit /b 1

:ALL
call :CHECKENV || exit /b 1
call :MAIN  || exit /b 1
call :FOCAL || exit /b 1
call :PGD   || exit /b 1
call :SHAP  || exit /b 1
echo.
echo ===========================================================================
echo All four stages completed.
echo Collect these files for the manuscript and the repository:
echo   results_final\final_summary.json          (Tables 5-8, Figs 1-3)
echo   focal_loss_sweep_results\                 (Section VI-A)
echo   pgd_results\                              (Section VI-D, Fig. 9)
echo   shap_hard_classes_results\                (Section VI-C3, Fig. 8)
echo ===========================================================================
goto :EOF

REM ---------------------------------------------------------------------------
:CHECKENV
echo [check] Python and required packages
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: python not found on PATH.
    exit /b 1
)
python -c "import torch, sklearn, imblearn, shap, scipy, matplotlib, pandas, numpy" 2>nul
if errorlevel 1 (
    echo ERROR: missing packages. Run:  pip install -r requirements.txt
    exit /b 1
)
python -c "import dataset_config" 2>nul
if errorlevel 1 (
    echo ERROR: dataset_config.py not found or has a syntax error.
    exit /b 1
)
echo [check] OK
exit /b 0

REM ---------------------------------------------------------------------------
:MAIN
echo.
echo === Stage 1/4: main experiment (RQ1-RQ4) - this is the long one ===
python run_full_experiment.py
if errorlevel 1 (
    echo ERROR in stage 1. Stage 1 is resumable: fix the problem and re-run
    echo "run_all.bat main" to continue from the last completed unit.
    exit /b 1
)
exit /b 0

REM ---------------------------------------------------------------------------
:FOCAL
echo.
echo === Stage 2/4: focal loss gamma sweep (Dataset A) ===
python focal_loss_gamma_sweep.py --dataset ertam_kaya
if errorlevel 1 (
    echo ERROR in stage 2.
    exit /b 1
)
exit /b 0

REM ---------------------------------------------------------------------------
:PGD
echo.
echo === Stage 3/4: PGD-10 vs FGSM (Dataset A) ===
python pgd_attack.py
if errorlevel 1 (
    echo ERROR in stage 3.
    exit /b 1
)
exit /b 0

REM ---------------------------------------------------------------------------
:SHAP
echo.
echo === Stage 4/4: SHAP for Worms and Backdoors (Dataset C) ===
echo Note: KernelExplainer is slow. Expect roughly 1-2 hours.
python shap_hard_classes.py --max_train_samples 400000
if errorlevel 1 (
    echo ERROR in stage 4. If it reports "Target class not found", re-run with
    echo the exact class-name spellings it printed, for example:
    echo     python shap_hard_classes.py --targets Worms Backdoor
    exit /b 1
)
exit /b 0
