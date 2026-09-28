# Task-Contribution Probe — Folds 01–01

> **Source:** `probe_generator.py`
> **Protocol:** Subject-level soft-vote (matches the main MC evaluator). Every group's standalone and LOO Δ AUROC are computed per fold, then summarised across folds. Positive Δ AUROC ⇒ the group contributes positively to detection.
> **Scope:** num_tasks=8, groupings=by-task, by-axis, by-type, by-inhibition, completed_folds=1.

## 1. Full Ensemble Baseline (all tasks soft-voted)

Same rule as the main evaluator. Subject-level prediction = mean of all window probabilities across all of a subject's tasks. This is the headline metric that each LOO ΔAUROC below is computed against.

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.719 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.719** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

## 2. Grouping: `by-task`

This grouping slices the 8 tasks into 8 group(s) and asks, per group: *(a)* how well does this group alone classify subjects? *(b)* how much does the full ensemble lose when this group is removed?

### 2.a Task 0 — Horizontal Saccade A  (tasks [0])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.250 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.250** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.750 | -0.031 |
| **mean** | — | **0.750** | **-0.031** |
| **std**  | — | 0.000 | 0.000 |

### 2.b Task 1 — Horizontal Saccade B  (tasks [1])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.531 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.531** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

### 2.c Task 2 — Horizontal Saccade B (anti)  (tasks [2])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 9 | 0.667 | 1.000 | 0.000 | 0.500 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.500** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

### 2.d Task 3 — Horizontal Saccade R  (tasks [3])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.438 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.438** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

### 2.e Task 4 — Vertical Saccade A  (tasks [4])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.417 | 0.250 | 0.750 | 0.781 |
| **mean** | — | **0.417** | **0.250** | **0.750** | **0.781** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.625 | +0.094 |
| **mean** | — | **0.625** | **+0.094** |
| **std**  | — | 0.000 | 0.000 |

### 2.f Task 5 — Vertical Saccade B  (tasks [5])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.500 | 0.250 | 1.000 | 0.719 |
| **mean** | — | **0.500** | **0.250** | **1.000** | **0.719** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.688 | +0.031 |
| **mean** | — | **0.688** | **+0.031** |
| **std**  | — | 0.000 | 0.000 |

### 2.g Task 6 — Vertical Saccade B (anti)  (tasks [6])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 11 | 0.636 | 1.000 | 0.000 | 0.929 |
| **mean** | — | **0.636** | **1.000** | **0.000** | **0.929** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

### 2.h Task 7 — Vertical Saccade R  (tasks [7])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.917 | 1.000 | 0.750 | 0.906 |
| **mean** | — | **0.917** | **1.000** | **0.750** | **0.906** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.688 | +0.031 |
| **mean** | — | **0.688** | **+0.031** |
| **std**  | — | 0.000 | 0.000 |

## 3. Grouping: `by-axis`

This grouping slices the 8 tasks into 2 group(s) and asks, per group: *(a)* how well does this group alone classify subjects? *(b)* how much does the full ensemble lose when this group is removed?

### 3.a Horizontal axis (tasks 0–3)  (tasks [0, 1, 2, 3])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.250 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.250** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.781 | -0.062 |
| **mean** | — | **0.781** | **-0.062** |
| **std**  | — | 0.000 | 0.000 |

### 3.b Vertical axis (tasks 4–7)  (tasks [4, 5, 6, 7])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.500 | 0.375 | 0.750 | 0.781 |
| **mean** | — | **0.500** | **0.375** | **0.750** | **0.781** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.250 | +0.469 |
| **mean** | — | **0.250** | **+0.469** |
| **std**  | — | 0.000 | 0.000 |

## 4. Grouping: `by-type`

This grouping slices the 8 tasks into 4 group(s) and asks, per group: *(a)* how well does this group alone classify subjects? *(b)* how much does the full ensemble lose when this group is removed?

### 4.a A — slow visually-guided (tasks 0, 4)  (tasks [0, 4])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.656 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.656** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.750 | -0.031 |
| **mean** | — | **0.750** | **-0.031** |
| **std**  | — | 0.000 | 0.000 |

### 4.b B — gap paradigm (tasks 1, 5)  (tasks [1, 5])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 0.625 | 0.750 | 0.781 |
| **mean** | — | **0.667** | **0.625** | **0.750** | **0.781** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.688 | +0.031 |
| **mean** | — | **0.688** | **+0.031** |
| **std**  | — | 0.000 | 0.000 |

### 4.c B-anti — anti-saccade (tasks 2, 6)  (tasks [2, 6])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.812 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.812** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

### 4.d R — repetitive (tasks 3, 7)  (tasks [3, 7])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.562 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.562** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.688 | +0.031 |
| **mean** | — | **0.688** | **+0.031** |
| **std**  | — | 0.000 | 0.000 |

## 5. Grouping: `by-inhibition`

This grouping slices the 8 tasks into 2 group(s) and asks, per group: *(a)* how well does this group alone classify subjects? *(b)* how much does the full ensemble lose when this group is removed?

### 5.a Reflexive saccades (A, B, R)  (tasks [0, 1, 3, 4, 5, 7])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.719 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.719** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.812 | -0.094 |
| **mean** | — | **0.812** | **-0.094** |
| **std**  | — | 0.000 | 0.000 |

### 5.b Cognitive inhibition (B-anti)  (tasks [2, 6])

**Standalone — subject-level soft-vote using only this group's windows:**

| Fold | n_subj | Acc | Sens | Spec | AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 01 | 12 | 0.667 | 1.000 | 0.000 | 0.812 |
| **mean** | — | **0.667** | **1.000** | **0.000** | **0.812** |
| **std**  | — | 0.000 | 0.000 | 0.000 | 0.000 |

**LOO contribution — subject-level soft-vote with this group's windows excluded:**

| Fold | n_subj | AUROC w/o group | Δ AUROC = full − w/o |
| :---: | :---: | :---: | :---: |
| 01 | 12 | 0.719 | +0.000 |
| **mean** | — | **0.719** | **+0.000** |
| **std**  | — | 0.000 | 0.000 |

## 6. Cross-Grouping Summary

Every group from every grouping, ranked by mean Δ AUROC across folds. Use this to compare contributions across slicings at a glance — e.g., is the Anti-saccade *inhibition* group's Δ larger than the strongest *axis* or any *individual task*'s Δ?

| Rank | Grouping | Group | Tasks | Mean Δ AUROC | Std Δ | Standalone AUROC | Suggested Action |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| 1 | `by-axis` | Vertical axis (tasks 4–7) | [4, 5, 6, 7] | +0.469 | 0.000 | 0.781 | **up-weight (≈ 1.5×)** |
| 2 | `by-task` | Task 4 — Vertical Saccade A | [4] | +0.094 | 0.000 | 0.781 | **up-weight (≈ 1.5×)** |
| 3 | `by-task` | Task 5 — Vertical Saccade B | [5] | +0.031 | 0.000 | 0.719 | keep (1.0×) |
| 4 | `by-task` | Task 7 — Vertical Saccade R | [7] | +0.031 | 0.000 | 0.906 | keep (1.0×) |
| 5 | `by-type` | B — gap paradigm (tasks 1, 5) | [1, 5] | +0.031 | 0.000 | 0.781 | keep (1.0×) |
| 6 | `by-type` | R — repetitive (tasks 3, 7) | [3, 7] | +0.031 | 0.000 | 0.562 | keep (1.0×) |
| 7 | `by-task` | Task 1 — Horizontal Saccade B | [1] | +0.000 | 0.000 | 0.531 | neutral (1.0×) |
| 8 | `by-task` | Task 2 — Horizontal Saccade B (anti) | [2] | +0.000 | 0.000 | 0.500 | neutral (1.0×) |
| 9 | `by-task` | Task 3 — Horizontal Saccade R | [3] | +0.000 | 0.000 | 0.438 | neutral (1.0×) |
| 10 | `by-task` | Task 6 — Vertical Saccade B (anti) | [6] | +0.000 | 0.000 | 0.929 | neutral (1.0×) |
| 11 | `by-type` | B-anti — anti-saccade (tasks 2, 6) | [2, 6] | +0.000 | 0.000 | 0.812 | neutral (1.0×) |
| 12 | `by-inhibition` | Cognitive inhibition (B-anti) | [2, 6] | +0.000 | 0.000 | 0.812 | neutral (1.0×) |
| 13 | `by-task` | Task 0 — Horizontal Saccade A | [0] | -0.031 | 0.000 | 0.250 | down-weight (≈ 0.5×) |
| 14 | `by-type` | A — slow visually-guided (tasks 0, 4) | [0, 4] | -0.031 | 0.000 | 0.656 | down-weight (≈ 0.5×) |
| 15 | `by-axis` | Horizontal axis (tasks 0–3) | [0, 1, 2, 3] | -0.062 | 0.000 | 0.250 | **drop / weight 0** |
| 16 | `by-inhibition` | Reflexive saccades (A, B, R) | [0, 1, 3, 4, 5, 7] | -0.094 | 0.000 | 0.719 | **drop / weight 0** |

---
_Notes:_ Δ AUROC is computed per fold on the SAME subject set (only subjects with non-empty `no_group` set are counted in that fold's Δ). Means/stds are NaN-safe across folds — folds with a single-class test set have NaN AUROC and are excluded from the average. Standalone AUROC is the mean across folds of subject-level AUROC computed using only the group's windows. The Suggested Action heuristic is `mean_delta ≥ 0.05 ⇒ up-weight`, `0.01..0.05 ⇒ keep`, `±0.01 ⇒ neutral`, `−0.05..−0.01 ⇒ down-weight`, `< −0.05 ⇒ drop`.