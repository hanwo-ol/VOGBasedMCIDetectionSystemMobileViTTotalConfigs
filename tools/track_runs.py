#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Experiment Performance Tracker
------------------------------
Non-invasive tracker that extracts:
1. All CLI hyperparameters used in each run
2. Pre-voting (window-level) performance: AUROC, Acc, Sens, Spec
3. Post-voting (subject-level) performance: AUROC, Acc, Sens, Spec
4. Voting Gain (Delta AUROC / Delta Acc)

Outputs a consolidated CSV: outputs/reports/experiment_tracker.csv
Original source code remains 100% untouched.
"""

import argparse
import csv
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _safe_float(val: Optional[str]) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def parse_log_file(log_path: Path) -> Optional[dict]:
    """Parse a single run log file for hyperparameters and metrics."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"[Warning] Failed to read {log_path}: {e}")
        return None

    # Check if run completed
    has_completed = "MC-CV Results" in text
    if not has_completed:
        fold_results = re.findall(r"-> Fold (\d+) Result", text)
        if not fold_results:
            return None

    # 1. Run ID & Timestamp
    rid_m = re.search(r"Run ID:\s*([^\s|]+)", text)
    run_id = rid_m.group(1).strip() if rid_m else log_path.stem.replace("run_", "")

    time_m = re.search(r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", text, re.MULTILINE)
    timestamp = time_m.group(1) if time_m else ""

    # 2. CLI Hyperparameters
    sig_m = re.search(r"signal_mode=([a-z_]+)", text)
    signal_mode = sig_m.group(1) if sig_m else "legacy"

    art_m = re.search(r"artifact=(?:thr=)?([0-9.]+|OFF[^\s|]*)", text)
    artifact_thr = art_m.group(1) if art_m else "45.0"

    bs_m = re.search(r"batch_size=(\d+)", text)
    batch_size = int(bs_m.group(1)) if bs_m else 32

    eval_bs_m = re.search(r"eval_batch=(\d+)", text)
    eval_batch_size = int(eval_bs_m.group(1)) if eval_bs_m else (batch_size * 8)

    drop_m = re.search(r"dropout=([0-9.]+)", text)
    dropout = float(drop_m.group(1)) if drop_m else 0.5

    pat_m = re.search(r"patience=(\d+)|es_patience=(\d+)", text)
    patience = int(pat_m.group(1) or pat_m.group(2)) if pat_m else 30

    aug_m = re.search(r"augment=(\S+)", text)
    augment = aug_m.group(1) if aug_m else "False"

    strat_m = re.search(r"stratified=(\{[^}]+\}|True|False)", text)
    stratified = strat_m.group(1) if strat_m else "False"

    vote_m = re.search(r"weighted_vote=(\{[^}]+\}|off)", text)
    vote_weights = vote_m.group(1) if vote_m else "off"

    vote_mode_m = re.search(r"vote_mode=([a-z_]+)", text)
    if vote_mode_m:
        vote_mode = vote_mode_m.group(1)
    elif "_vdiffsparse" in run_id or "_vdiff_sparsemax" in run_id:
        vote_mode = "auto_diff_sparsemax"
    elif "_vdiffsoft" in run_id or "_vdiff_softmax" in run_id:
        vote_mode = "auto_diff_softmax"
    elif "_vattnmil" in run_id or "_vmil" in run_id or "_vattn_mil" in run_id:
        vote_mode = "auto_attention_mil"
    elif "_vauroc" in run_id:
        vote_mode = "auto_auroc"
    elif "_vstack" in run_id:
        vote_mode = "auto_stacking"
    elif "_vtruestack" in run_id:
        vote_mode = "auto_true_stacking"
    elif "_vnnls" in run_id:
        vote_mode = "auto_nnls"
    else:
        vote_mode = "manual"

    splits_m = re.search(r"(\d+)\s+folds", text)
    n_splits = int(splits_m.group(1)) if splits_m else 30

    backbone_m = re.search(r"backbone=([^\s|]+)", text)
    backbone = backbone_m.group(1) if backbone_m else "mobilevit-small"

    region_m = re.search(r"region=([^\s|]+)", text)
    region = region_m.group(1) if region_m else "event"

    cwt_bl_m = re.search(r"cwt_baseline=([a-zA-Z0-9_]+)", text)
    if cwt_bl_m:
        raw_bl = cwt_bl_m.group(1)
        if raw_bl == "True":
            cwt_baseline = "subtraction"
        elif raw_bl == "False":
            cwt_baseline = "bypass"
        else:
            cwt_baseline = raw_bl
    elif "_nocwtbl" in run_id:
        cwt_baseline = "bypass"
    elif "_hybrid" in run_id:
        cwt_baseline = "hybrid"
    else:
        cwt_baseline = "subtraction"

    # 3. Post-voting (Subject-level) Metrics
    subj_acc = re.search(r"Accuracy\s+:\s*([0-9.]+)\s*±\s*([0-9.]+)", text)
    subj_sens = re.search(r"Sensitivity\s+:\s*([0-9.]+)\s*±\s*([0-9.]+)", text)
    subj_spec = re.search(r"Specificity\s+:\s*([0-9.]+)\s*±\s*([0-9.]+)", text)
    subj_auroc = re.search(r"AUROC\s+:\s*([0-9.]+)\s*±\s*([0-9.]+)", text)

    subj_acc_mean = _safe_float(subj_acc.group(1)) if subj_acc else None
    subj_acc_std = _safe_float(subj_acc.group(2)) if subj_acc else None
    subj_sens_mean = _safe_float(subj_sens.group(1)) if subj_sens else None
    subj_sens_std = _safe_float(subj_sens.group(2)) if subj_sens else None
    subj_spec_mean = _safe_float(subj_spec.group(1)) if subj_spec else None
    subj_spec_std = _safe_float(subj_spec.group(2)) if subj_spec else None
    subj_auroc_mean = _safe_float(subj_auroc.group(1)) if subj_auroc else None
    subj_auroc_std = _safe_float(subj_auroc.group(2)) if subj_auroc else None

    # 4. Pre-voting (Window-level) Metrics from log (Validation window AUROCs across folds)
    fold_val_aurocs = [float(x) for x in re.findall(r"Done\. Best AUROC:\s*([0-9.]+)", text)]
    win_auroc_mean = float(np.mean(fold_val_aurocs)) if fold_val_aurocs else None
    win_auroc_std = float(np.std(fold_val_aurocs)) if fold_val_aurocs else None

    return {
        "timestamp": timestamp,
        "run_id": run_id,
        "signal_mode": signal_mode,
        "artifact_threshold": artifact_thr,
        "vote_mode": vote_mode,
        "vote_weights": vote_weights,
        "stratified": stratified,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "dropout": dropout,
        "patience": patience,
        "backbone": backbone,
        "region": region,
        "augment": augment,
        "n_splits": n_splits,
        "cwt_baseline": cwt_baseline,
        "status": "COMPLETED" if has_completed else "INCOMPLETE",
        # Pre-voting (Window-level)
        "win_auroc_mean": win_auroc_mean,
        "win_auroc_std": win_auroc_std,
        "win_acc_mean": None,
        "win_acc_std": None,
        "win_sens_mean": None,
        "win_sens_std": None,
        "win_spec_mean": None,
        "win_spec_std": None,
        # Post-voting (Subject-level)
        "subj_auroc_mean": subj_auroc_mean,
        "subj_auroc_std": subj_auroc_std,
        "subj_acc_mean": subj_acc_mean,
        "subj_acc_std": subj_acc_std,
        "subj_sens_mean": subj_sens_mean,
        "subj_sens_std": subj_sens_std,
        "subj_spec_mean": subj_spec_mean,
        "subj_spec_std": subj_spec_std,
        # Voting Gain
        "delta_auroc": (subj_auroc_mean - win_auroc_mean) if (subj_auroc_mean is not None and win_auroc_mean is not None) else None,
        "log_path": str(log_path),
    }


def parse_window_probs_csv(csv_path: Path) -> Optional[dict]:
    """Parse test-time window_probs.csv to compute exact test window-level metrics."""
    if not csv_path.exists() or not SKLEARN_AVAILABLE:
        return None

    try:
        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append({
                    "rep": int(row["rep"]),
                    "sid": row["subject"],
                    "task": int(row["task"]),
                    "prob": float(row["prob"]),
                    "label": int(row["label"]),
                })
        if not records:
            return None

        # Group by rep (fold)
        by_rep = {}
        for r in records:
            by_rep.setdefault(r["rep"], []).append(r)

        fold_aurocs, fold_accs, fold_senss, fold_specs = [], [], [], []
        for rep, r_list in by_rep.items():
            y_true = np.array([x["label"] for x in r_list])
            y_prob = np.array([x["prob"] for x in r_list])
            y_pred = (y_prob > 0.5).astype(int)

            if len(np.unique(y_true)) > 1:
                fold_aurocs.append(roc_auc_score(y_true, y_prob))
            fold_accs.append(accuracy_score(y_true, y_pred))

            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            fold_senss.append(sens)
            fold_specs.append(spec)

        return {
            "win_auroc_mean": float(np.mean(fold_aurocs)) if fold_aurocs else None,
            "win_auroc_std": float(np.std(fold_aurocs)) if fold_aurocs else None,
            "win_acc_mean": float(np.mean(fold_accs)) if fold_accs else None,
            "win_acc_std": float(np.std(fold_accs)) if fold_accs else None,
            "win_sens_mean": float(np.mean(fold_senss)) if fold_senss else None,
            "win_sens_std": float(np.std(fold_senss)) if fold_senss else None,
            "win_spec_mean": float(np.mean(fold_specs)) if fold_specs else None,
            "win_spec_std": float(np.std(fold_specs)) if fold_specs else None,
        }
    except Exception as e:
        print(f"[Warning] Failed to parse {csv_path}: {e}")
        return None


def collect_all_runs(project_root: Path) -> List[dict]:
    """Scan outputs/logs and outputs/checkpoints to extract all run records."""
    logs_dir = project_root / "outputs" / "logs"
    checkpoints_dir = project_root / "outputs" / "checkpoints"

    log_files = list(logs_dir.rglob("*.log"))
    records = []

    for log_file in sorted(log_files):
        data = parse_log_file(log_file)
        if not data:
            continue

        # Look for corresponding window_probs.csv in checkpoints
        rid = data["run_id"]
        csv_candidates = [
            checkpoints_dir / f"run_{rid}" / "window_probs.csv",
            checkpoints_dir / rid / "window_probs.csv",
        ]
        win_csv = next((p for p in csv_candidates if p.exists()), None)
        if win_csv:
            win_metrics = parse_window_probs_csv(win_csv)
            if win_metrics:
                for k, v in win_metrics.items():
                    if v is not None:
                        data[k] = v
                if data["subj_auroc_mean"] is not None and data["win_auroc_mean"] is not None:
                    data["delta_auroc"] = data["subj_auroc_mean"] - data["win_auroc_mean"]

        records.append(data)

    return records


def save_to_csv(records: List[dict], output_csv: Path) -> None:
    """Save records to CSV, updating existing entries cleanly."""
    fieldnames = [
        "timestamp",
        "run_id",
        "status",
        "signal_mode",
        "artifact_threshold",
        "vote_mode",
        "vote_weights",
        "stratified",
        "batch_size",
        "eval_batch_size",
        "dropout",
        "patience",
        "backbone",
        "region",
        "augment",
        "n_splits",
        "cwt_baseline",
        # Pre-voting (Window-level)
        "win_auroc_mean",
        "win_auroc_std",
        "win_acc_mean",
        "win_acc_std",
        "win_sens_mean",
        "win_sens_std",
        "win_spec_mean",
        "win_spec_std",
        # Post-voting (Subject-level)
        "subj_auroc_mean",
        "subj_auroc_std",
        "subj_acc_mean",
        "subj_acc_std",
        "subj_sens_mean",
        "subj_sens_std",
        "subj_spec_mean",
        "subj_spec_std",
        # Delta
        "delta_auroc",
        "log_path",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = {}
            for k in fieldnames:
                v = r.get(k)
                if isinstance(v, float):
                    row[k] = f"{v:.4f}"
                else:
                    row[k] = v if v is not None else ""
            writer.writerow(row)

    print(f"\n[Success] Saved {len(records)} run record(s) to: {output_csv.resolve()}")


def print_summary_table(records: List[dict]) -> None:
    """Print a clean CLI table of the tracked runs."""
    if not records:
        print("[Info] No completed runs found.")
        return

    import sys
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n" + "=" * 148)
    print(f"{'Run ID':<35} | {'Mode':<10} | {'Vote':<18} | {'Strat':<5} | {'BL-Sub':<7} | {'Win AUROC (Pre)':<18} | {'Subj AUROC (Post)':<18} | {'dAUROC':<8}")
    print("-" * 148)
    for r in records:
        rid = r["run_id"]
        if len(rid) > 35:
            rid = rid[:32] + "..."
        mode = r["signal_mode"]
        vote_m = r.get("vote_mode", "manual")
        strat = "Y" if "True" in str(r["stratified"]) or "0" in str(r["stratified"]) else "N"
        bl_val = r.get("cwt_baseline", "subtraction")
        if bl_val == "hybrid":
            bl_sub = "Hybrid"
        elif bl_val in (False, "bypass", "False"):
            bl_sub = "Bypass"
        else:
            bl_sub = "Sub"
        win_a = f"{r['win_auroc_mean']:.3f}+/-{r['win_auroc_std']:.3f}" if r["win_auroc_mean"] is not None else "N/A"
        subj_a = f"{r['subj_auroc_mean']:.3f}+/-{r['subj_auroc_std']:.3f}" if r["subj_auroc_mean"] is not None else "N/A"
        delta = f"{r['delta_auroc']:+.3f}" if r["delta_auroc"] is not None else "N/A"
        print(f"{rid:<35} | {mode:<10} | {vote_m:<18} | {strat:<5} | {bl_sub:<7} | {win_a:<18} | {subj_a:<18} | {delta:<8}")
    print("=" * 148 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Track experiment parameters & pre/post voting performance.")
    parser.add_argument(
        "--csv-path",
        default="outputs/reports/experiment_tracker.csv",
        help="Destination CSV file path (default: outputs/reports/experiment_tracker.csv)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / args.csv_path

    records = collect_all_runs(project_root)
    print_summary_table(records)
    save_to_csv(records, csv_path)


if __name__ == "__main__":
    main()
