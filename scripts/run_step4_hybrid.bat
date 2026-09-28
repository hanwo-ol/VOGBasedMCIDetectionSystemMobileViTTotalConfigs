@echo off
REM ==============================================================================
REM Step 4: Hybrid CWT Baseline (Task 0, 1 Baseline Subtraction, Task 2-7 Bypass)
REM Uses dedicated data_store_full_4err_hybrid.pkl cache.
REM Evaluates with True Stacking meta-learner.
REM ==============================================================================
echo [INFO] Starting Hybrid CWT Baseline Pipeline...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --cwt-baseline-mode hybrid ^
    --eval-batch-size 32 ^
    --vote-mode auto_true_stacking
