@echo off
REM ==============================================================================
REM Step 5a: Differentiable Task Weighter (Sparsemax Projection)
REM Joint window CE + Subject BCE with simplex-projected learnable task weights.
REM Exactly prunes uninformative tasks to 0.0 weight autonomously.
REM ==============================================================================
echo [INFO] Starting Differentiable Sparsemax Voting Pipeline...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode auto_diff_sparsemax
