@echo off
REM ==============================================================================
REM Step 5c: Hierarchical Gated Attention Multiple Instance Learning (Attention MIL)
REM Pools windows per task, then learns subject-level gated attention across tasks.
REM ==============================================================================
echo [INFO] Starting Attention MIL Voting Pipeline...
call conda activate mci 2>nul
python src/four_error_using/detection_caller/detection_caller.py ^
    --signal-mode four_error ^
    --artifact-threshold 30.0 ^
    --stratified ^
    --patience 15 ^
    --n-splits 15 ^
    --no-cwt-baseline ^
    --eval-batch-size 32 ^
    --vote-mode auto_attention_mil
