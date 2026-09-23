"""
Comprehensive Group Comparison (HC vs MCI) Scalogram Visualizer.
Loads the complete cohort cache (5,715 epochs across 37 subjects)
and generates:
1. 8 task-specific 4x3 figures (Rows: 4 channels, Cols: HC Mean | MCI Mean | Difference (MCI - HC))
2. 1 master overview 8x4 grid of difference maps (MCI - HC)
3. 1 statistical t-score map per task highlighting significant frequency-time regions.
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path("c:/Users/USER/MCI-VOG")
CACHE_FILE = ROOT / "outputs/cache/data_store_full_4err.pkl"
OUT_DIR = ROOT / "HW_docs/images/group_comparisons"
ARTIFACT_DIR = Path("C:/Users/USER/.gemini/antigravity-ide/brain/271068a1-daaa-402f-a700-b99607220436/group_comparisons")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 9

TASK_MAP = {
    0: "Horizontal Saccade A",
    1: "Horizontal Saccade B",
    2: "Horizontal Saccade B (anti)",
    3: "Horizontal Saccade R",
    4: "Vertical Saccade A",
    5: "Vertical Saccade B",
    6: "Vertical Saccade B (anti)",
    7: "Vertical Saccade R",
}

CHANNEL_NAMES = [
    "Ch 0: LH - TH (Horiz Left Eye)",
    "Ch 1: RH - TH (Horiz Right Eye)",
    "Ch 2: LV - TV (Vert Left Eye)",
    "Ch 3: RV - TV (Vert Right Eye)",
]

print("Loading cached data_store...")
with open(CACHE_FILE, "rb") as f:
    payload = pickle.load(f)

data_store = payload["data_store"]

# Flatten tensors by group and task
# store: group -> task_id -> list of [4, 32, 32] tensors
tensors_by_task = {"HC": {t: [] for t in range(8)}, "MCI": {t: [] for t in range(8)}}

for grp in ["HC", "MCI"]:
    for sid, epochs in data_store[grp].items():
        for item in epochs:
            tensor = item[0] # [4, 32, 32]
            task_id = item[1]
            tensors_by_task[grp][task_id].append(tensor)

for grp in ["HC", "MCI"]:
    print(f"Group {grp} epoch counts by task:")
    for t in range(8):
        print(f"  Task {t} ({TASK_MAP[t]}): {len(tensors_by_task[grp][t])} epochs")

def save_fig(fig, filename):
    p1 = OUT_DIR / filename
    p2 = ARTIFACT_DIR / filename
    fig.savefig(p1, dpi=200, bbox_inches='tight')
    fig.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1.name}")

# =========================================================================
# Part 1: Task-wise 4-Channel Group Comparison (8 Individual Detailed Figures)
# =========================================================================
diff_summary = {} # task_id -> [4, 32, 32] diff map
t_score_summary = {} # task_id -> [4, 32, 32] t-score map

for t in range(8):
    task_name = TASK_MAP[t]
    hc_list = np.array(tensors_by_task["HC"][t])   # [N_hc, 4, 32, 32]
    mci_list = np.array(tensors_by_task["MCI"][t]) # [N_mci, 4, 32, 32]
    
    n_hc = len(hc_list)
    n_mci = len(mci_list)
    
    if n_hc == 0 or n_mci == 0:
        print(f"Skipping task {t} due to zero epochs.")
        continue
        
    hc_mean = np.mean(hc_list, axis=0)   # [4, 32, 32]
    mci_mean = np.mean(mci_list, axis=0) # [4, 32, 32]
    diff = mci_mean - hc_mean            # [4, 32, 32]
    diff_summary[t] = diff
    
    # Compute Welch's t-test per pixel
    # stats.ttest_ind on axis 0
    t_stat, p_val = stats.ttest_ind(mci_list, hc_list, axis=0, equal_var=False)
    t_score_summary[t] = t_stat
    
    fig, axs = plt.subplots(4, 3, figsize=(14, 13))
    fig.suptitle(
        f"Task {t}: {task_name}\n"
        f"Group Scalogram Comparison [HC: N={n_hc} epochs (14 subj) vs MCI: N={n_mci} epochs (23 subj)]",
        fontsize=14, fontweight='bold', y=0.99
    )
    
    # Global scale for HC and MCI mean to compare fairly
    vmin = min(hc_mean.min(), mci_mean.min())
    vmax = max(hc_mean.max(), mci_mean.max())
    
    # Diff scale (symmetric around 0)
    diff_abs_max = max(abs(diff.min()), abs(diff.max()), 0.1)
    
    for ch in range(4):
        ch_name = CHANNEL_NAMES[ch]
        
        # 1. HC Mean
        im0 = axs[ch, 0].imshow(hc_mean[ch], aspect='equal', origin='lower', cmap='plasma', vmin=vmin, vmax=vmax)
        axs[ch, 0].set_title(f"{ch_name}\nHC Mean (N={n_hc})", fontsize=10)
        axs[ch, 0].set_ylabel("Freq (15~60Hz)")
        if ch == 3: axs[ch, 0].set_xlabel("Time Bin (0~31)")
        fig.colorbar(im0, ax=axs[ch, 0], fraction=0.046, pad=0.04)
        
        # 2. MCI Mean
        im1 = axs[ch, 1].imshow(mci_mean[ch], aspect='equal', origin='lower', cmap='plasma', vmin=vmin, vmax=vmax)
        axs[ch, 1].set_title(f"{ch_name}\nMCI Mean (N={n_mci})", fontsize=10)
        if ch == 3: axs[ch, 1].set_xlabel("Time Bin (0~31)")
        fig.colorbar(im1, ax=axs[ch, 1], fraction=0.046, pad=0.04)
        
        # 3. Difference (MCI - HC)
        im2 = axs[ch, 2].imshow(diff[ch], aspect='equal', origin='lower', cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max)
        axs[ch, 2].set_title(f"Difference (MCI - HC)\n[Red: MCI > HC, Blue: MCI < HC]", fontsize=10)
        if ch == 3: axs[ch, 2].set_xlabel("Time Bin (0~31)")
        cb2 = fig.colorbar(im2, ax=axs[ch, 2], fraction=0.046, pad=0.04)
        cb2.set_label("Δ Z-Score")

    plt.tight_layout()
    slug = task_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    save_fig(fig, f"task_{t:02d}_{slug}_group_comparison.png")

# =========================================================================
# Part 2: Master Difference Overview (8 Tasks x 4 Channels = 32 Panels)
# =========================================================================
print("Generating Master 8x4 Difference Grid...")
fig, axs = plt.subplots(8, 4, figsize=(18, 26))
fig.suptitle(
    "Master Scalogram Difference Map Across All 8 Saccade Tasks\n"
    "Value = MCI Mean - HC Mean (Red = MCI Hyper-activity/Noise, Blue = MCI Energy Deficit)",
    fontsize=16, fontweight='bold', y=0.995
)

for t in range(8):
    task_name = TASK_MAP[t]
    diff = diff_summary[t]
    d_max = max(abs(diff.min()), abs(diff.max()), 0.1)
    
    for ch in range(4):
        ax = axs[t, ch]
        im = ax.imshow(diff[ch], aspect='equal', origin='lower', cmap='bwr', vmin=-d_max, vmax=d_max)
        
        if t == 0:
            ax.set_title(f"{CHANNEL_NAMES[ch]}", fontsize=11, fontweight='bold')
        if ch == 0:
            ax.set_ylabel(f"Task {t}\n{task_name}", fontsize=10, fontweight='bold')
        if t == 7:
            ax.set_xlabel("Time (0~31)")
            
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
save_fig(fig, "master_difference_overview_8tasks_4channels.png")

# =========================================================================
# Part 3: Statistical Significance (t-Score Maps with p < 0.01 Contours)
# =========================================================================
print("Generating Statistical t-Score Dashboard...")
fig, axs = plt.subplots(8, 4, figsize=(18, 26))
fig.suptitle(
    "Statistical Welch's t-Score Maps Across All 8 Tasks\n"
    "t-value = (Mean_MCI - Mean_HC) / SE (Regions |t| > 2.58 correspond to p < 0.01)",
    fontsize=16, fontweight='bold', y=0.995
)

for t in range(8):
    task_name = TASK_MAP[t]
    t_map = t_score_summary[t]
    t_max = max(abs(t_map.min()), abs(t_map.max()), 4.0)
    
    for ch in range(4):
        ax = axs[t, ch]
        im = ax.imshow(t_map[ch], aspect='equal', origin='lower', cmap='coolwarm', vmin=-t_max, vmax=t_max)
        
        # Add significance contour (|t| > 2.58 is p < 0.01)
        sig_mask = np.abs(t_map[ch]) > 2.58
        if sig_mask.any():
            ax.contour(sig_mask, levels=[0.5], colors='black', linewidths=1.0)
            
        if t == 0:
            ax.set_title(f"{CHANNEL_NAMES[ch]}", fontsize=11, fontweight='bold')
        if ch == 0:
            ax.set_ylabel(f"Task {t}\n{task_name}", fontsize=10, fontweight='bold')
        if t == 7:
            ax.set_xlabel("Time (0~31)")
            
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if ch == 3: cb.set_label("t-statistic")

plt.tight_layout()
save_fig(fig, "master_statistical_t_score_maps_all_tasks.png")

print("ALL GROUP COMPARISON FIGURES GENERATED SUCCESSFULLY!")
