@echo off
REM ==============================================================================
REM Step 3a: True Stacking (Logistic Regression Meta-Learner with Intercept)
REM P(s) = sigma(beta_0 + sum(beta_t * p_bar_{s, t}))
REM ==============================================================================
echo [INFO] Starting True Stacking CV...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode auto_true_stacking
