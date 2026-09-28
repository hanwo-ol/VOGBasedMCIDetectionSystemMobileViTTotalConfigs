#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cross-platform runner for MCI post-Run 6 experiment stages."""

import argparse
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CALLER = _ROOT / "src" / "four_error_using" / "detection_caller" / "detection_caller.py"


def _resolve_python() -> str:
    try:
        import torch  # noqa: F401
        return sys.executable
    except ImportError:
        pass
    candidates = [
        Path(sys.executable).parent / "envs" / "mci" / "python.exe",
        Path.home() / "anaconda3" / "envs" / "mci" / "python.exe",
        Path("C:/Users/user.LAPTOP-M31S3OSQ/anaconda3/envs/mci/python.exe"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


_PYTHON_EXE = _resolve_python()

STAGES = {
    "2": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "manual",
        "--vote-weights", "0.0 0.0 0.0 1.5 0.5 1.5 1.0 3.5",
    ],
    "3a": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_true_stacking",
    ],
    "3b": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_nnls",
    ],
    "4": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--cwt-baseline-mode", "hybrid",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_true_stacking",
    ],
    "5a": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_diff_sparsemax",
    ],
    "5b": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_diff_softmax",
    ],
    "5c": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--no-cwt-baseline",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_attention_mil",
    ],
    "5a_hybrid": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--cwt-baseline-mode", "hybrid",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_diff_sparsemax",
    ],
    "5b_hybrid": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--cwt-baseline-mode", "hybrid",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_diff_softmax",
    ],
    "5c_hybrid": [
        _PYTHON_EXE, str(_CALLER),
        "--signal-mode", "four_error",
        "--artifact-threshold", "30.0",
        "--stratified",
        "--patience", "15",
        "--n-splits", "15",
        "--cwt-baseline-mode", "hybrid",
        "--eval-batch-size", "32",
        "--vote-mode", "auto_attention_mil",
    ],
}


def main():
    parser = argparse.ArgumentParser(description="Run MCI pipeline stages.")
    parser.add_argument(
        "--stage",
        nargs="+",
        choices=list(STAGES.keys()) + ["all_5"],
        required=True,
        help="Stage(s) to run in sequence. Example: --stage 5a 5a_hybrid 5b 5c, or --stage all_5",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command(s) without executing",
    )
    args = parser.parse_args()

    stages_to_run = []
    for s in args.stage:
        if s == "all_5":
            stages_to_run.extend(["5a", "5a_hybrid", "5b", "5c"])
        else:
            stages_to_run.append(s)

    for stage_name in stages_to_run:
        cmd = STAGES[stage_name]
        print(f"\n{'=' * 80}")
        print(f"[INFO] Stage {stage_name} command:\n{' '.join(cmd)}")
        print(f"{'=' * 80}\n")
        if args.dry_run:
            continue

        ret = subprocess.call(cmd, cwd=str(_ROOT))
        if ret != 0:
            print(f"[ERROR] Stage {stage_name} failed with exit code {ret}. Aborting sequence.")
            sys.exit(ret)

        # Automatically update experiment tracker after each successful stage
        tracker_cmd = [_PYTHON_EXE, str(_ROOT / "tools" / "track_runs.py")]
        subprocess.call(tracker_cmd, cwd=str(_ROOT))


if __name__ == "__main__":
    main()
