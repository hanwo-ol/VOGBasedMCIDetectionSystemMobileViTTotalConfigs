@echo off
REM ==============================================================================
REM Step 5b: Differentiable Task Weighter (Softmax Dense Weights)
REM Joint window CE + Subject BCE with Softmax normalized task weights.
REM Dense probability distribution across all 8 tasks.
REM ==============================================================================
echo [INFO] Starting Differentiable Softmax Voting Pipeline...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode auto_diff_softmax
