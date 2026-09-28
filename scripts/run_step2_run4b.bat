@echo off
REM ==============================================================================
REM Step 2: Run 4b Benchmark (Refined Heuristic Voting: Task 1 & 2 Excluded)
REM Task Weights: 0.0(H-A), 0.0(H-B), 0.0(H-B-anti), 1.5(H-R), 0.5(V-A), 1.5(V-B), 1.0(V-B-anti), 3.5(V-R)
REM ==============================================================================
echo [INFO] Starting Run 4b Benchmark...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode manual ^
    --vote-weights "0.0 0.0 0.0 1.5 0.5 1.5 1.0 3.5"
