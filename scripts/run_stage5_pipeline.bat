@echo off
REM ==============================================================================
REM Stage 5 Automated End-to-End Sequential Pipeline
REM Runs Stage 5a, 5a_hybrid, 5b, and 5c in sequence.
REM Updates experiment_tracker.csv automatically after each stage.
REM ==============================================================================
echo ==============================================================================
echo [INFO] Launching Stage 5 Sequential Experiment Pipeline
echo   1. Stage 5a        : auto_diff_sparsemax (Bypass)
echo   2. Stage 5a_hybrid : auto_diff_sparsemax (Hybrid)
echo   3. Stage 5b        : auto_diff_softmax   (Bypass)
echo   4. Stage 5c        : auto_attention_mil  (Bypass)
echo ==============================================================================
call conda activate mci 2>nul
python scripts/run_experiments.py --stage all_5
pause
