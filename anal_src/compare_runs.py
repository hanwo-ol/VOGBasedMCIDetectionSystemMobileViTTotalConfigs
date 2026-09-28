"""Reusable Comparative Analysis Tool for Saccade MCI Detection Runs.

Compares any two runs (Run A vs Run B) using:
1. 15-Fold Paired Statistical Significance Testing (Paired t-test, Wilcoxon signed-rank, Cohen's d)
2. 8-Task Granular Dissection (Window AUROC & Task-isolated Subject AUROC)
3. 37-Subject Error Transitions (Hard flips FP<->TN, FN<->TP, soft probability margin shifts)
4. High-Quality Scientific Visualizations
5. Deterministic, Hallucination-Free Markdown Report Generation
"""

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score

TASK_NAMES = {
    0: "Task 0: Horizontal Saccade A",
    1: "Task 1: Horizontal Saccade B",
    2: "Task 2: Horizontal Saccade B (anti)",
    3: "Task 3: Horizontal Saccade R",
    4: "Task 4: Vertical Saccade A",
    5: "Task 5: Vertical Saccade B",
    6: "Task 6: Vertical Saccade B (anti)",
    7: "Task 7: Vertical Saccade R",
}

TASK_SHORT_NAMES = {
    0: "H-A",
    1: "H-B",
    2: "H-Anti",
    3: "H-R",
    4: "V-A",
    5: "V-B",
    6: "V-Anti",
    7: "V-R",
}

DEFAULT_TASK_WEIGHTS = {0: 0.0, 1: 0.5, 2: 0.5, 3: 1.0, 4: 0.0, 5: 1.5, 6: 1.5, 7: 3.0}


def resolve_run_probs(run_identifier: str, project_root: Path) -> Tuple[Path, str]:
    """Resolve a run identifier (path, run_id, or folder name) to its window_probs.csv path."""
    p = Path(run_identifier)
    if p.is_file() and p.name == "window_probs.csv":
        run_name = p.parent.name.replace("run_", "")
        return p, run_name
    if p.is_dir() and (p / "window_probs.csv").is_file():
        run_name = p.name.replace("run_", "")
        return p / "window_probs.csv", run_name

    # Check under outputs/checkpoints
    chk_dir = project_root / "outputs" / "checkpoints"
    candidates = [
        chk_dir / run_identifier / "window_probs.csv",
        chk_dir / f"run_{run_identifier}" / "window_probs.csv",
    ]
    for cand in candidates:
        if cand.is_file():
            run_name = cand.parent.name.replace("run_", "")
            return cand, run_name

    # Partial match in checkpoints directory
    if chk_dir.exists():
        matches = [d for d in chk_dir.iterdir() if d.is_dir() and run_identifier in d.name]
        if matches:
            cand = matches[0] / "window_probs.csv"
            if cand.is_file():
                run_name = matches[0].name.replace("run_", "")
                return cand, run_name

    raise FileNotFoundError(f"Could not locate window_probs.csv for run identifier: {run_identifier}")


def parse_weights(weights_str: Optional[str]) -> Dict[int, float]:
    """Parse string representation of task weights dictionary."""
    if not weights_str:
        return DEFAULT_TASK_WEIGHTS
    try:
        # Replace python-like syntax if needed
        clean_str = weights_str.replace("'", '"')
        parsed = json.loads(clean_str)
        return {int(k): float(v) for k, v in parsed.items()}
    except Exception:
        # Fallback eval for {0: 0.0, ...}
        import ast
        parsed = ast.literal_eval(weights_str)
        return {int(k): float(v) for k, v in parsed.items()}


def compute_fold_metrics(df: pd.DataFrame, task_weights: Dict[int, float]) -> Dict[str, np.ndarray]:
    """Compute fold-level window and subject metrics across all folds."""
    folds = sorted(df["rep"].unique())
    win_aurocs, win_accs, win_sens, win_specs = [], [], [], []
    subj_aurocs, subj_accs, subj_sens, subj_specs = [], [], [], []

    for rep in folds:
        df_rep = df[df["rep"] == rep]
        
        # Window-level metrics
        w_labels = df_rep["label"].values
        w_probs = df_rep["prob"].values
        win_aurocs.append(roc_auc_score(w_labels, w_probs))
        w_preds = (w_probs >= 0.5).astype(int)
        win_accs.append(accuracy_score(w_labels, w_preds))
        win_sens.append(recall_score(w_labels, w_preds, pos_label=1, zero_division=0))
        w_cm = confusion_matrix(w_labels, w_preds, labels=[0, 1])
        w_tn, w_fp, w_fn, w_tp = w_cm.ravel()
        win_specs.append(w_tn / (w_tn + w_fp) if (w_tn + w_fp) > 0 else 0.0)

        # Subject-level metrics via task-weighted voting
        subj_preds, subj_labels = [], []
        for sid, grp in df_rep.groupby("subject"):
            w_sum = sum(task_weights.get(t, 0.0) for t in grp["task"])
            if w_sum > 0:
                p_sub = sum(task_weights.get(t, 0.0) * p for t, p in zip(grp["task"], grp["prob"])) / w_sum
            else:
                p_sub = float(grp["prob"].mean())
            subj_preds.append(p_sub)
            subj_labels.append(int(grp["label"].iloc[0]))

        subj_preds = np.array(subj_preds)
        subj_labels = np.array(subj_labels)
        
        subj_aurocs.append(roc_auc_score(subj_labels, subj_preds))
        s_preds = (subj_preds >= 0.5).astype(int)
        subj_accs.append(accuracy_score(subj_labels, s_preds))
        subj_sens.append(recall_score(subj_labels, s_preds, pos_label=1, zero_division=0))
        s_cm = confusion_matrix(subj_labels, s_preds, labels=[0, 1])
        s_tn, s_fp, s_fn, s_tp = s_cm.ravel()
        subj_specs.append(s_tn / (s_tn + s_fp) if (s_tn + s_fp) > 0 else 0.0)

    return {
        "folds": np.array(folds),
        "win_auroc": np.array(win_aurocs),
        "win_acc": np.array(win_accs),
        "win_sens": np.array(win_sens),
        "win_spec": np.array(win_specs),
        "subj_auroc": np.array(subj_aurocs),
        "subj_acc": np.array(subj_accs),
        "subj_sens": np.array(subj_sens),
        "subj_spec": np.array(subj_specs),
    }


def compute_task_dissection(
    df_a: pd.DataFrame, df_b: pd.DataFrame, task_weights: Dict[int, float]
) -> List[Dict]:
    """Compute window AUROC and isolated subject AUROC per task for both runs."""
    tasks = sorted(set(df_a["task"].unique()).union(set(df_b["task"].unique())))
    folds = sorted(df_a["rep"].unique())
    dissection_results = []

    for t in tasks:
        w_auc_a, w_auc_b = [], []
        s_auc_a, s_auc_b = [], []

        for rep in folds:
            # Window level
            sub_a = df_a[(df_a["rep"] == rep) & (df_a["task"] == t)]
            sub_b = df_b[(df_b["rep"] == rep) & (df_b["task"] == t)]

            if len(sub_a) > 0 and len(np.unique(sub_a["label"])) > 1:
                w_auc_a.append(roc_auc_score(sub_a["label"], sub_a["prob"]))
            if len(sub_b) > 0 and len(np.unique(sub_b["label"])) > 1:
                w_auc_b.append(roc_auc_score(sub_b["label"], sub_b["prob"]))

            # Isolated Subject level (unweighted mean of task t windows only)
            s_preds_a, s_labels_a = [], []
            for _, grp in sub_a.groupby("subject"):
                s_preds_a.append(grp["prob"].mean())
                s_labels_a.append(grp["label"].iloc[0])
            if len(s_labels_a) > 0 and len(np.unique(s_labels_a)) > 1:
                s_auc_a.append(roc_auc_score(s_labels_a, s_preds_a))

            s_preds_b, s_labels_b = [], []
            for _, grp in sub_b.groupby("subject"):
                s_preds_b.append(grp["prob"].mean())
                s_labels_b.append(grp["label"].iloc[0])
            if len(s_labels_b) > 0 and len(np.unique(s_labels_b)) > 1:
                s_auc_b.append(roc_auc_score(s_labels_b, s_preds_b))

        w_auc_a = np.array(w_auc_a)
        w_auc_b = np.array(w_auc_b)
        s_auc_a = np.array(s_auc_a)
        s_auc_b = np.array(s_auc_b)

        # Paired tests on window AUROC
        w_diff = w_auc_b - w_auc_a
        w_p = stats.ttest_rel(w_auc_b, w_auc_a).pvalue if len(w_diff) > 1 and np.std(w_diff) > 1e-9 else 1.0

        # Paired tests on subject AUROC
        s_diff = s_auc_b - s_auc_a
        s_p = stats.ttest_rel(s_auc_b, s_auc_a).pvalue if len(s_diff) > 1 and np.std(s_diff) > 1e-9 else 1.0

        dissection_results.append({
            "task_id": t,
            "task_name": TASK_NAMES.get(t, f"Task {t}"),
            "short_name": TASK_SHORT_NAMES.get(t, f"T{t}"),
            "weight": task_weights.get(t, 0.0),
            "n_windows_a": int((df_a["task"] == t).sum() / len(folds)),
            "win_auc_a_mean": float(np.mean(w_auc_a)),
            "win_auc_a_std": float(np.std(w_auc_a)),
            "win_auc_b_mean": float(np.mean(w_auc_b)),
            "win_auc_b_std": float(np.std(w_auc_b)),
            "win_delta": float(np.mean(w_diff)),
            "win_p": float(w_p),
            "subj_auc_a_mean": float(np.mean(s_auc_a)),
            "subj_auc_a_std": float(np.std(s_auc_a)),
            "subj_auc_b_mean": float(np.mean(s_auc_b)),
            "subj_auc_b_std": float(np.std(s_auc_b)),
            "subj_delta": float(np.mean(s_diff)),
            "subj_p": float(s_p),
        })

    return dissection_results


def compute_subject_transitions(
    df_a: pd.DataFrame, df_b: pd.DataFrame, task_weights: Dict[int, float]
) -> pd.DataFrame:
    """Compute subject-by-subject probability shifts and error transitions across test folds."""
    subjects = sorted(set(df_a["subject"].unique()).union(set(df_b["subject"].unique())))
    records = []

    for sid in subjects:
        # Subject true label
        sub_a = df_a[df_a["subject"] == sid]
        sub_b = df_b[df_b["subject"] == sid]
        label = int(sub_a["label"].iloc[0]) if len(sub_a) > 0 else int(sub_b["label"].iloc[0])

        # Compute subject prob in each fold where tested
        folds_tested_a = sorted(sub_a["rep"].unique())
        probs_a = []
        for rep in folds_tested_a:
            grp = sub_a[sub_a["rep"] == rep]
            w_sum = sum(task_weights.get(t, 0.0) for t in grp["task"])
            p = sum(task_weights.get(t, 0.0) * p for t, p in zip(grp["task"], grp["prob"])) / w_sum if w_sum > 0 else grp["prob"].mean()
            probs_a.append(p)

        folds_tested_b = sorted(sub_b["rep"].unique())
        probs_b = []
        for rep in folds_tested_b:
            grp = sub_b[sub_b["rep"] == rep]
            w_sum = sum(task_weights.get(t, 0.0) for t in grp["task"])
            p = sum(task_weights.get(t, 0.0) * p for t, p in zip(grp["task"], grp["prob"])) / w_sum if w_sum > 0 else grp["prob"].mean()
            probs_b.append(p)

        mean_p_a = float(np.mean(probs_a)) if probs_a else np.nan
        mean_p_b = float(np.mean(probs_b)) if probs_b else np.nan

        pred_a = int(mean_p_a >= 0.5)
        pred_b = int(mean_p_b >= 0.5)

        # Confusion categories
        cat_a = "TP" if (label == 1 and pred_a == 1) else ("TN" if (label == 0 and pred_a == 0) else ("FP" if (label == 0 and pred_a == 1) else "FN"))
        cat_b = "TP" if (label == 1 and pred_b == 1) else ("TN" if (label == 0 and pred_b == 0) else ("FP" if (label == 0 and pred_b == 1) else "FN"))

        # Transition classification
        correct_a = (pred_a == label)
        correct_b = (pred_b == label)

        if not correct_a and correct_b:
            transition = "Fixed"  # Error -> Correct
        elif correct_a and not correct_b:
            transition = "Broken"  # Correct -> Error
        elif correct_a and correct_b:
            transition = "Maintained Correct"
        else:
            transition = "Maintained Error"

        delta_p = mean_p_b - mean_p_a
        # Effective margin gain: positive means probability moved towards true label
        margin_gain = (1 if label == 1 else -1) * delta_p

        records.append({
            "subject": sid,
            "label": label,
            "class_name": "MCI" if label == 1 else "HC",
            "folds_tested": len(folds_tested_a),
            "prob_a": mean_p_a,
            "prob_b": mean_p_b,
            "delta_p": delta_p,
            "margin_gain": margin_gain,
            "pred_a": pred_a,
            "pred_b": pred_b,
            "cat_a": cat_a,
            "cat_b": cat_b,
            "transition": transition,
        })

    return pd.DataFrame(records)


def generate_visualizations(
    task_dissection: List[Dict],
    subj_df: pd.DataFrame,
    run_a_name: str,
    run_b_name: str,
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate publication-ready comparison figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # Figure 1: 8-Task Granular AUROC Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    tasks = [d["short_name"] for d in task_dissection]
    x = np.arange(len(tasks))
    width = 0.35

    # Panel 1: Window AUROC
    ax1 = axes[0]
    w_a_means = [d["win_auc_a_mean"] for d in task_dissection]
    w_a_stds = [d["win_auc_a_std"] for d in task_dissection]
    w_b_means = [d["win_auc_b_mean"] for d in task_dissection]
    w_b_stds = [d["win_auc_b_std"] for d in task_dissection]

    r1 = ax1.bar(x - width/2, w_a_means, width, yerr=w_a_stds, label=f"Run A ({run_a_name[:12]}...)", color="#4A90E2", capsize=3, alpha=0.9)
    r2 = ax1.bar(x + width/2, w_b_means, width, yerr=w_b_stds, label=f"Run B ({run_b_name[:12]}...)", color="#E94A4A", capsize=3, alpha=0.9)
    ax1.set_title("Window-Level AUROC by Task", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("AUROC", fontsize=11)
    ax1.set_ylim(0.3, 1.0)
    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", frameon=True)

    # Panel 2: Isolated Subject AUROC
    ax2 = axes[1]
    s_a_means = [d["subj_auc_a_mean"] for d in task_dissection]
    s_a_stds = [d["subj_auc_a_std"] for d in task_dissection]
    s_b_means = [d["subj_auc_b_mean"] for d in task_dissection]
    s_b_stds = [d["subj_auc_b_std"] for d in task_dissection]

    r3 = ax2.bar(x - width/2, s_a_means, width, yerr=s_a_stds, label=f"Run A", color="#4A90E2", capsize=3, alpha=0.9)
    r4 = ax2.bar(x + width/2, s_b_means, width, yerr=s_b_stds, label=f"Run B", color="#E94A4A", capsize=3, alpha=0.9)
    ax2.set_title("Task-Isolated Subject AUROC", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0.3, 1.0)
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # Annotate significance on Subject AUROC panel
    for idx, d in enumerate(task_dissection):
        p_val = d["subj_p"]
        delta = d["subj_delta"]
        if p_val < 0.05:
            mark = "**" if p_val < 0.01 else "*"
            y_max = max(s_a_means[idx] + s_a_stds[idx], s_b_means[idx] + s_b_stds[idx]) + 0.03
            ax2.text(idx, min(y_max, 0.96), mark, ha="center", va="bottom", fontsize=12, fontweight="bold", color="black")

    plt.tight_layout()
    fig1_path = fig_dir / "task_wise_auroc_comparison.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -------------------------------------------------------------
    # Figure 2: Subject Probability Shift Dumbbell / Slope Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(15, 7))

    # Sort subjects: first by label (HC then MCI), then by Run A probability
    sorted_df = subj_df.sort_values(by=["label", "prob_a"]).reset_index(drop=True)
    n_subjs = len(sorted_df)

    hc_mask = sorted_df["label"] == 0
    mci_mask = sorted_df["label"] == 1

    y_pos = np.arange(n_subjs)

    # Draw dumbbell lines
    for i, row in sorted_df.iterrows():
        p_a = row["prob_a"]
        p_b = row["prob_b"]
        label = row["label"]
        trans = row["transition"]

        # Color line based on transition
        if trans == "Fixed":
            line_col = "#2ECC71"  # Bright green
            lw = 2.2
        elif trans == "Broken":
            line_col = "#E67E22"  # Bright orange
            lw = 2.2
        else:
            line_col = "#BDC3C7"  # Muted grey
            lw = 1.0

        ax.plot([p_a, p_b], [i, i], color=line_col, linewidth=lw, zorder=2)

    # Plot points for Run A and Run B
    # HC subjects
    ax.scatter(sorted_df.loc[hc_mask, "prob_a"], y_pos[hc_mask], color="#3498DB", marker="o", s=45, label="Run A (HC)", zorder=3, alpha=0.85)
    ax.scatter(sorted_df.loc[hc_mask, "prob_b"], y_pos[hc_mask], color="#2980B9", marker="s", s=55, label="Run B (HC)", zorder=4)

    # MCI subjects
    ax.scatter(sorted_df.loc[mci_mask, "prob_a"], y_pos[mci_mask], color="#E74C3C", marker="o", s=45, label="Run A (MCI)", zorder=3, alpha=0.85)
    ax.scatter(sorted_df.loc[mci_mask, "prob_b"], y_pos[mci_mask], color="#C0392B", marker="s", s=55, label="Run B (MCI)", zorder=4)

    # Highlight transitions
    fixed_idx = sorted_df.index[sorted_df["transition"] == "Fixed"].tolist()
    broken_idx = sorted_df.index[sorted_df["transition"] == "Broken"].tolist()

    if fixed_idx:
        ax.scatter(sorted_df.loc[fixed_idx, "prob_b"], y_pos[fixed_idx], facecolors="none", edgecolors="#2ECC71", s=130, linewidths=2.0, label="Fixed (Error->Correct)", zorder=5)
    if broken_idx:
        ax.scatter(sorted_df.loc[broken_idx, "prob_b"], y_pos[broken_idx], facecolors="none", edgecolors="#E67E22", s=130, linewidths=2.0, label="Broken (Correct->Error)", zorder=5)

    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.2, alpha=0.8, label="Decision Threshold (P=0.5)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['subject'][-10:]} ({row['class_name']})" for _, row in sorted_df.iterrows()], fontsize=8)
    ax.set_xlabel("Predicted Probability P(MCI)", fontsize=11, fontweight="bold")
    ax.set_title("Subject Probability Shifts and Diagnostic Transitions (Run A -> Run B)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", linestyle=":", alpha=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=9, bbox_to_anchor=(1.0, 0.05))

    plt.tight_layout()
    fig2_path = fig_dir / "subject_probability_shift.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fig1_path, fig2_path


def generate_markdown_report(
    run_a_name: str,
    run_b_name: str,
    run_a_path: Path,
    run_b_path: Path,
    m_a: Dict[str, np.ndarray],
    m_b: Dict[str, np.ndarray],
    task_dissection: List[Dict],
    subj_df: pd.DataFrame,
    fig1_path: Path,
    fig2_path: Path,
    task_weights: Dict[int, float],
) -> str:
    """Generate comprehensive dry, academic, objective markdown report."""
    n_folds = len(m_a["folds"])

    # Statistical significance on overall metrics
    metrics_to_test = [
        ("Window AUROC", "win_auroc"),
        ("Window Accuracy", "win_acc"),
        ("Window Sensitivity", "win_sens"),
        ("Window Specificity", "win_spec"),
        ("Subject AUROC", "subj_auroc"),
        ("Subject Accuracy", "subj_acc"),
        ("Subject Sensitivity", "subj_sens"),
        ("Subject Specificity", "subj_spec"),
    ]

    stat_rows = []
    for label, key in metrics_to_test:
        vals_a = m_a[key]
        vals_b = m_b[key]
        diff = vals_b - vals_a
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        cohen_d = mean_diff / std_diff if std_diff > 1e-9 else 0.0

        t_res = stats.ttest_rel(vals_b, vals_a)
        t_stat, t_pval = float(t_res.statistic), float(t_res.pvalue)

        try:
            w_res = stats.wilcoxon(vals_b, vals_a)
            w_stat, w_pval = float(w_res.statistic), float(w_res.pvalue)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        stat_rows.append({
            "metric": label,
            "mean_a": float(np.mean(vals_a)),
            "std_a": float(np.std(vals_a)),
            "mean_b": float(np.mean(vals_b)),
            "std_b": float(np.std(vals_b)),
            "delta": mean_diff,
            "cohen_d": cohen_d,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "w_pval": w_pval,
        })

    # Subject transition counts
    n_fixed = (subj_df["transition"] == "Fixed").sum()
    n_broken = (subj_df["transition"] == "Broken").sum()
    n_m_corr = (subj_df["transition"] == "Maintained Correct").sum()
    n_m_err = (subj_df["transition"] == "Maintained Error").sum()
    n_total_subjs = len(subj_df)

    fixed_subjs = subj_df[subj_df["transition"] == "Fixed"]
    broken_subjs = subj_df[subj_df["transition"] == "Broken"]

    # Mean margin shifts
    hc_shift = subj_df[subj_df["label"] == 0]["delta_p"].mean()
    mci_shift = subj_df[subj_df["label"] == 1]["delta_p"].mean()
    net_margin_gain = subj_df["margin_gain"].mean()

    # Build markdown
    md = []
    md.append(f"# Comparative Ablation Analysis: Run A vs Run B\n")
    md.append(f"- **Run A (Reference)**: `{run_a_name}`\n")
    md.append(f"  - Source: `{run_a_path}`\n")
    md.append(f"- **Run B (Experimental)**: `{run_b_name}`\n")
    md.append(f"  - Source: `{run_b_path}`\n")
    md.append(f"- **Evaluation Protocol**: {n_folds}-Fold Stratified Monte-Carlo Cross Validation (37 Subjects Total)\n")
    md.append(f"- **Subject Aggregation Weights**: `{json.dumps(task_weights)}`\n\n")

    md.append("## 1. Overall Performance and 15-Fold Paired Statistical Significance\n\n")
    md.append("| Metric | Run A (Mean ± SD) | Run B (Mean ± SD) | Δ (B - A) | Cohen's d | Paired t-test p | Wilcoxon p |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for r in stat_rows:
        sig_marker = " *" if r["t_pval"] < 0.05 else ""
        w_p_str = f"{r['w_pval']:.4f}" if not np.isnan(r["w_pval"]) else "N/A"
        md.append(
            f"| **{r['metric']}** | {r['mean_a']:.4f} ± {r['std_a']:.4f} | {r['mean_b']:.4f} ± {r['std_b']:.4f} | "
            f"**{r['delta']:+.4f}** | {r['cohen_d']:.3f} | {r['t_pval']:.4f}{sig_marker} | {w_p_str} |\n"
        )
    md.append("\n> *Note: Significance marker (*) denotes two-tailed p < 0.05.*\n\n")

    md.append("## 2. Granular 8-Task Dissection\n\n")
    md.append("Analysis of individual task performance across window-level classification and isolated subject-level soft voting.\n\n")
    md.append("| Task ID | Task Name | Weight | Win AUROC (Run A) | Win AUROC (Run B) | Win Δ | Subj AUROC (Run A) | Subj AUROC (Run B) | Subj Δ | Subj p-val |\n")
    md.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for t in task_dissection:
        sig = " *" if t["subj_p"] < 0.05 else ""
        md.append(
            f"| {t['task_id']} | {t['task_name']} | {t['weight']} | "
            f"{t['win_auc_a_mean']:.4f} ± {t['win_auc_a_std']:.4f} | {t['win_auc_b_mean']:.4f} ± {t['win_auc_b_std']:.4f} | "
            f"{t['win_delta']:+.4f} | {t['subj_auc_a_mean']:.4f} ± {t['subj_auc_a_std']:.4f} | {t['subj_auc_b_mean']:.4f} ± {t['subj_auc_b_std']:.4f} | "
            f"**{t['subj_delta']:+.4f}** | {t['subj_p']:.4f}{sig} |\n"
        )
    md.append("\n")

    # Top contributors
    sorted_by_gain = sorted(task_dissection, key=lambda x: x["subj_delta"], reverse=True)
    top_pos = [t for t in sorted_by_gain if t["subj_delta"] > 0]
    top_neg = [t for t in sorted_by_gain if t["subj_delta"] < 0]
    md.append("### Key Task Dissection Findings:\n")
    md.append(f"- **Top Positive Contributor Tasks**: " + ", ".join([f"{t['task_name']} ({t['subj_delta']:+.4f})" for t in top_pos]) + "\n")
    if top_neg:
        md.append(f"- **Negative/Degraded Tasks**: " + ", ".join([f"{t['task_name']} ({t['subj_delta']:+.4f})" for t in top_neg]) + "\n")
    else:
        md.append("- **Negative/Degraded Tasks**: None observed.\n")
    md.append("\n")

    md.append("## 3. 37-Subject Error Transitions and Diagnostic Shift\n\n")
    md.append("### 3.1 Transition Summary\n")
    md.append(f"- **Total Evaluated Subjects**: {n_total_subjs} (HC: {(subj_df['label'] == 0).sum()}, MCI: {(subj_df['label'] == 1).sum()})\n")
    md.append(f"- **Fixed (Error → Correct)**: **{n_fixed} subjects**\n")
    md.append(f"- **Broken (Correct → Error)**: **{n_broken} subjects**\n")
    md.append(f"- **Maintained Correct**: **{n_m_corr} subjects**\n")
    md.append(f"- **Maintained Error**: **{n_m_err} subjects**\n")
    md.append(f"- **Net Subject Diagnostic Flip**: **{n_fixed - n_broken:+d} subjects**\n")
    md.append(f"- **Mean HC Probability Shift ΔP**: {hc_shift:+.4f} (target: < 0)\n")
    md.append(f"- **Mean MCI Probability Shift ΔP**: {mci_shift:+.4f} (target: > 0)\n")
    md.append(f"- **Mean Directional Margin Gain**: {net_margin_gain:+.4f}\n\n")

    md.append("### 3.2 Flipped Subjects Detail\n\n")
    if len(fixed_subjs) > 0:
        md.append("#### Fixed Subjects (Recovered from Misclassification):\n")
        md.append("| Subject ID | Class | Run A Prob (Cat) | Run B Prob (Cat) | ΔP | Transition |\n")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in fixed_subjs.iterrows():
            md.append(f"| `{r['subject']}` | {r['class_name']} | {r['prob_a']:.4f} ({r['cat_a']}) | {r['prob_b']:.4f} ({r['cat_b']}) | {r['delta_p']:+.4f} | {r['cat_a']} → {r['cat_b']} |\n")
        md.append("\n")

    if len(broken_subjs) > 0:
        md.append("#### Broken Subjects (New Misclassifications Introduced):\n")
        md.append("| Subject ID | Class | Run A Prob (Cat) | Run B Prob (Cat) | ΔP | Transition |\n")
        md.append("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for _, r in broken_subjs.iterrows():
            md.append(f"| `{r['subject']}` | {r['class_name']} | {r['prob_a']:.4f} ({r['cat_a']}) | {r['prob_b']:.4f} ({r['cat_b']}) | {r['delta_p']:+.4f} | {r['cat_a']} → {r['cat_b']} |\n")
        md.append("\n")

    md.append("### 3.3 Full Subject Probability Shift Roster\n\n")
    md.append("| Subject ID | Class | Folds Tested | Run A P(MCI) | Run B P(MCI) | ΔP | Margin Gain | Cat A | Cat B | State |\n")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
    for _, r in subj_df.sort_values(by=["label", "prob_a"]).iterrows():
        md.append(
            f"| `{r['subject']}` | {r['class_name']} | {r['folds_tested']} | "
            f"{r['prob_a']:.4f} | {r['prob_b']:.4f} | {r['delta_p']:+.4f} | {r['margin_gain']:+.4f} | "
            f"{r['cat_a']} | {r['cat_b']} | {r['transition']} |\n"
        )
    md.append("\n")

    md.append("## 4. Visualizations\n\n")
    md.append(f"### Figure 1: 8-Task Granular AUROC Comparison\n")
    md.append(f"![Task-wise AUROC Comparison](figures/{fig1_path.name})\n\n")
    md.append(f"### Figure 2: Subject Probability Shift and Transition Analysis\n")
    md.append(f"![Subject Probability Shift](figures/{fig2_path.name})\n\n")

    return "".join(md)


def main():
    parser = argparse.ArgumentParser(description="Deterministic Run Comparison Analysis Tool")
    parser.add_argument("--run-a", type=str, required=True, help="Baseline/Reference run identifier or path")
    parser.add_argument("--run-b", type=str, required=True, help="Experimental/Comparison run identifier or path")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save report and figures")
    parser.add_argument("--hw-docs-dir", type=str, default="HW_docs", help="HW_docs directory to sync report and figures")
    parser.add_argument("--task-weights", type=str, default=None, help="JSON or dict string of task weights")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Resolve input paths
    path_a, name_a = resolve_run_probs(args.run_a, project_root)
    path_b, name_b = resolve_run_probs(args.run_b, project_root)

    print(f"Loading Run A: {name_a} from {path_a}")
    print(f"Loading Run B: {name_b} from {path_b}")

    df_a = pd.read_csv(path_a)
    df_b = pd.read_csv(path_b)

    task_weights = parse_weights(args.task_weights)

    # 1. Compute fold-level metrics
    print("Computing fold-level metrics...")
    m_a = compute_fold_metrics(df_a, task_weights)
    m_b = compute_fold_metrics(df_b, task_weights)

    # 2. Compute 8-Task granular dissection
    print("Computing 8-task granular dissection...")
    task_dissection = compute_task_dissection(df_a, df_b, task_weights)

    # 3. Compute 37-Subject transitions
    print("Computing 37-subject error transitions...")
    subj_df = compute_subject_transitions(df_a, df_b, task_weights)

    # Resolve output directory
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = project_root / "outputs" / "reports" / f"comparison_{name_a[:15]}_vs_{name_b[:15]}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4. Generate visualizations
    print("Generating visualizations...")
    fig1_path, fig2_path = generate_visualizations(task_dissection, subj_df, name_a, name_b, out_dir)

    # 5. Generate markdown report
    print("Generating markdown report...")
    report_md = generate_markdown_report(
        run_a_name=name_a,
        run_b_name=name_b,
        run_a_path=path_a,
        run_b_path=path_b,
        m_a=m_a,
        m_b=m_b,
        task_dissection=task_dissection,
        subj_df=subj_df,
        fig1_path=fig1_path,
        fig2_path=fig2_path,
        task_weights=task_weights,
    )

    report_path = out_dir / "ablation_comparison_report.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"Report written to: {report_path}")

    # Also sync to HW_docs if requested
    if args.hw_docs_dir:
        hw_docs = project_root / args.hw_docs_dir
        if hw_docs.exists():
            hw_report_path = hw_docs / f"ablation_{name_a[:15]}_vs_{name_b[:15]}.md"
            # Update image links for HW_docs
            hw_images_dir = hw_docs / "images"
            hw_images_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fig1_path, hw_images_dir / fig1_path.name)
            shutil.copy2(fig2_path, hw_images_dir / fig2_path.name)

            hw_md = report_md.replace("figures/", "images/")
            hw_report_path.write_text(hw_md, encoding="utf-8")
            print(f"Synced report to HW_docs: {hw_report_path}")

    print("\n--- Summary of Analysis ---")
    print(f"Subj AUROC: {np.mean(m_a['subj_auroc']):.4f} -> {np.mean(m_b['subj_auroc']):.4f} (Delta: {np.mean(m_b['subj_auroc']) - np.mean(m_a['subj_auroc']):+.4f})")
    print(f"Subj Sens : {np.mean(m_a['subj_sens']):.4f} -> {np.mean(m_b['subj_sens']):.4f} (Delta: {np.mean(m_b['subj_sens']) - np.mean(m_a['subj_sens']):+.4f})")
    print(f"Subj Spec : {np.mean(m_a['subj_spec']):.4f} -> {np.mean(m_b['subj_spec']):.4f} (Delta: {np.mean(m_b['subj_spec']) - np.mean(m_a['subj_spec']):+.4f})")
    print(f"Fixed subjects: {(subj_df['transition'] == 'Fixed').sum()}, Broken subjects: {(subj_df['transition'] == 'Broken').sum()}")


if __name__ == "__main__":
    main()
