@echo off
REM ==============================================================================
REM Step 3b: NNLS Voting (Non-Negative Least Squares Sparse Weight Estimation)
REM Solves min_w ||X w - y||_2^2 s.t. w >= 0, then normalizes sum(w) = 1.
REM Harmful tasks receive exact weight 0.0.
REM ==============================================================================
echo [INFO] Starting NNLS Sparse Voting CV...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode auto_nnls
