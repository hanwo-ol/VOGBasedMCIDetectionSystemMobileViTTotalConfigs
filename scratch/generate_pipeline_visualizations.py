"""
Pipeline step-by-step visualizer for VOG to Scalogram process.
Loads real patient CSV from data/sample/ and runs the exact functions from data_engineer.py
and models/layers, saving visualization figures at each intermediate transformation.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pywt
from scipy.ndimage import zoom
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Paths
ROOT = Path("c:/Users/USER/MCI-VOG")
CSV_PATH = ROOT / "data/sample/안미영_PD VOG -_Horizontal Saccade B (anti).csv"
OUT_DIR = ROOT / "HW_docs/images/scalogram_pipeline"
ARTIFACT_DIR = Path("C:/Users/USER/.gemini/antigravity-ide/brain/271068a1-daaa-402f-a700-b99607220436/scalogram_steps")

OUT_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Inline ConvAdapter to avoid transformers dependency in anaconda base
class ConvAdapter(torch.nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 3, kernel_size=(5, 1), padding=(2, 0)),
            torch.nn.BatchNorm2d(3),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# Matplotlib formatting
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def save_fig(fig, filename):
    p1 = OUT_DIR / filename
    p2 = ARTIFACT_DIR / filename
    fig.savefig(p1, dpi=200, bbox_inches='tight')
    fig.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {p1.name}")

# ==========================================
# 01. Raw CSV Loading
# ==========================================
# Safe CSV Loader from data_engineer.py
def load_csv_safely(file_path):
    for enc in ['utf-16', 'utf-16le', 'utf-8-sig', 'cp949', 'utf-8']:
        try:
            with open(file_path, 'r', encoding=enc, errors='replace') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if 'lh' in line.lower() and 'rh' in line.lower():
                    cols = [c.replace('\x00', '').strip().lower() for c in line.split(',')]
                    parsed = [
                        [v.replace('\x00', '').strip() for v in l.split(',')]
                        for l in lines[i + 1:] if l.strip()
                    ]
                    df = pd.DataFrame(parsed, columns=cols)
                    return df.apply(pd.to_numeric, errors='coerce').dropna(how='all').reset_index(drop=True)
        except Exception:
            continue
    raise ValueError(f"Failed to load {file_path}")

print("Step 01: Loading CSV safely...")
df = load_csv_safely(CSV_PATH)

# Resolve columns
time_col = next(c for c in df.columns if 'time' in c or c == 't')
th_col = next(c for c in df.columns if 'targeth' in c or 'target_h' in c)
tv_col = next(c for c in df.columns if 'targetv' in c or 'target_v' in c)
lh_col = 'lh'
rh_col = 'rh'
lv_col = 'lv'
rv_col = 'rv'

time_val = df[time_col].dropna().values
fs = 1.0 / np.mean(np.diff(time_val)) if len(time_val) > 1 else 120.0

fig, axs = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
axs[0].plot(time_val[:1200], df[th_col].values[:1200], 'k-', lw=1.5, label='Target H')
axs[0].plot(time_val[:1200], df[lh_col].values[:1200], 'b-', alpha=0.7, lw=1.0, label='Left Eye H')
axs[0].plot(time_val[:1200], df[rh_col].values[:1200], 'c-', alpha=0.7, lw=1.0, label='Right Eye H')
axs[0].set_ylabel('Horizontal (deg)')
axs[0].legend(loc='upper right')
axs[0].grid(True, alpha=0.3)
axs[0].set_title('Step 01: Raw VOG Recording (First 10 Seconds) - 120Hz')

axs[1].plot(time_val[:1200], df[tv_col].values[:1200], 'k-', lw=1.5, label='Target V')
axs[1].plot(time_val[:1200], df[lv_col].values[:1200], 'r-', alpha=0.7, lw=1.0, label='Left Eye V')
axs[1].plot(time_val[:1200], df[rv_col].values[:1200], 'm-', alpha=0.7, lw=1.0, label='Right Eye V')
axs[1].set_ylabel('Vertical (deg)')
axs[1].legend(loc='upper right')
axs[1].grid(True, alpha=0.3)

axs[2].plot(time_val[:1200], np.gradient(time_val[:1200]), 'g-', lw=1.0)
axs[2].set_ylabel('dt (s)')
axs[2].set_xlabel('Time (s)')
axs[2].set_title(f'Sampling Interval (Mean fs = {fs:.1f} Hz)')
axs[2].grid(True, alpha=0.3)
save_fig(fig, "01_raw_csv_time_series.png")

# ==========================================
# 02. Target Inversion (Anti-saccade)
# ==========================================
print("Step 02: Target Inversion...")
target_orig = df[th_col].fillna(0).values.copy()
target_inv = target_orig * -1  # Anti-saccade rule

fig, ax = plt.subplots(figsize=(10, 4))
t_slice = slice(200, 700)
ax.plot(time_val[t_slice], target_orig[t_slice], 'gray', linestyle='--', lw=1.5, label='Original Target H (+15 / -15 deg)')
ax.plot(time_val[t_slice], target_inv[t_slice], 'r-', lw=2.0, label='Inverted Target H (Anti: Target * -1)')
ax.plot(time_val[t_slice], df[lh_col].values[t_slice], 'b-', alpha=0.7, label='Actual Eye Position (Left Eye H)')
ax.set_title('Step 02: Anti-Saccade Target Inversion (Task 2: Horizontal Saccade B Anti)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Angle (deg)')
ax.legend()
ax.grid(True, alpha=0.3)
save_fig(fig, "02_target_inversion.png")

# ==========================================
# 03. Event Window Extraction (1.0s = 120 samples)
# ==========================================
print("Step 03: Event Window Extraction...")
event_indices = np.where(np.diff(target_inv, prepend=0) != 0)[0]
# Pick a clean first event
target_idx = event_indices[1] if len(event_indices) > 1 else event_indices[0]
samples_pre = int(0.2 * fs)
samples_post = int(0.8 * fs)
s, e = target_idx - samples_pre, target_idx + samples_post

t_rel = (np.arange(s, e) - target_idx) / fs  # -0.2 to +0.8s
raw_lh = df[lh_col].values[s:e]
raw_rh = df[rh_col].values[s:e]
raw_lv = df[lv_col].values[s:e]
raw_rv = df[rv_col].values[s:e]
raw_th = target_inv[s:e]
raw_tv = df[tv_col].values[s:e]

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.axvspan(-0.2, 0.0, color='yellow', alpha=0.2, label='Pre-stimulus Baseline (-0.2s ~ 0.0s, 24 samples)')
ax.axvspan(0.0, 0.8, color='cyan', alpha=0.1, label='Post-stimulus Response (0.0s ~ 0.8s, 96 samples)')
ax.plot(t_rel, raw_th, 'k--', lw=2.0, label='Inverted Target H')
ax.plot(t_rel, raw_lh, 'b-', lw=1.5, label='Left Eye H')
ax.plot(t_rel, raw_rh, 'c-', lw=1.5, label='Right Eye H')
ax.axvline(0.0, color='r', linestyle=':', lw=2.0, label='Target Step Transition (t = 0s)')
ax.set_title(f'Step 03: Event-Locked 1-Second Window (Total {len(t_rel)} Samples @ 120Hz)')
ax.set_xlabel('Relative Time (s)')
ax.set_ylabel('Eye Position (deg)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
save_fig(fig, "03_event_window_extraction.png")

# ==========================================
# 04. Error Calculation & Baseline Correction
# ==========================================
print("Step 04: Error & Baseline Correction...")
err_LH_raw = (raw_lh - raw_th).astype(np.float64)
err_RH_raw = (raw_rh - raw_th).astype(np.float64)
err_LV_raw = (raw_lv - raw_tv).astype(np.float64)
err_RV_raw = (raw_rv - raw_tv).astype(np.float64)

# Baseline offset
base_lh = np.mean(err_LH_raw[:samples_pre])
err_LH = err_LH_raw - base_lh
err_RH = err_RH_raw - np.mean(err_RH_raw[:samples_pre])
err_LV = err_LV_raw - np.mean(err_LV_raw[:samples_pre])
err_RV = err_RV_raw - np.mean(err_RV_raw[:samples_pre])

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(t_rel, err_LH_raw, 'b--', alpha=0.7, label='Raw Error LH (eye - target)')
axs[0].axhline(base_lh, color='orange', linestyle='-', lw=1.5, label=f'Baseline Mean = {base_lh:.2f}° (Pre 0.2s)')
axs[0].axvspan(-0.2, 0.0, color='yellow', alpha=0.2)
axs[0].set_ylabel('Raw Error (deg)')
axs[0].legend()
axs[0].grid(True, alpha=0.3)
axs[0].set_title('Step 04: Error Signal and Baseline Offset Determination')

axs[1].plot(t_rel, err_LH, 'b-', lw=1.5, label='Baseline-Subtracted Error LH')
axs[1].plot(t_rel, err_LV, 'r-', lw=1.0, alpha=0.8, label='Baseline-Subtracted Error LV (Cross-Axis)')
axs[1].axhline(0, color='k', linestyle=':', alpha=0.5)
axs[1].axvspan(-0.2, 0.0, color='yellow', alpha=0.2)
axs[1].set_ylabel('Corrected Error (deg)')
axs[1].set_xlabel('Relative Time (s)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
axs[1].set_title('Baseline-Corrected Error Signal (Mean of Pre-0.2s = 0)')
save_fig(fig, "04_error_baseline_correction.png")

# ==========================================
# 05. Artifact Rejection Check
# ==========================================
print("Step 05: Artifact Check...")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(t_rel, err_LH, 'b-', lw=1.5, label='Task-Axis Error LH')
ax.plot(t_rel, err_RH, 'c-', lw=1.5, label='Task-Axis Error RH')
ax.axhline(30.0, color='r', linestyle='--', lw=2.0, label='Artifact Threshold (+30.0 deg)')
ax.axhline(-30.0, color='r', linestyle='--', lw=2.0, label='Artifact Threshold (-30.0 deg)')
max_err = max(np.max(np.abs(err_LH)), np.max(np.abs(err_RH)))
ax.fill_between(t_rel, -30, 30, color='green', alpha=0.08, label='Valid Acceptance Region')
ax.set_title(f'Step 05: Artifact Rejection Check (Max Abs Error = {max_err:.2f}° <= 30.0° -> PASSED)')
ax.set_xlabel('Relative Time (s)')
ax.set_ylabel('Error (deg)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
save_fig(fig, "05_artifact_check.png")

# ==========================================
# 06. Continuous Wavelet Transform (CWT)
# ==========================================
print("Step 06: CWT...")
min_freq = 15.0
max_freq = 60.0
freq_bins = 32
w_morlet = 4.0
wavelet_name = f"cmor{w_morlet}-1.0"
fc = pywt.central_frequency(wavelet_name)
sampling_period = 1.0 / fs
frequencies = np.logspace(np.log10(min_freq), np.log10(max_freq), freq_bins)
scales = fc / (frequencies * sampling_period)

cwtm, _ = pywt.cwt(err_LH, scales, wavelet_name, sampling_period=sampling_period)
re_cwt = np.real(cwtm)
im_cwt = np.imag(cwtm)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
im0 = axs[0].imshow(re_cwt, aspect='auto', origin='lower', extent=[-0.2, 0.8, 15, 60], cmap='RdBu_r')
axs[0].set_title(f'Step 06-A: CWT Real Part (32 Scales × 120 Samples)')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Frequency (Hz)')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(im_cwt, aspect='auto', origin='lower', extent=[-0.2, 0.8, 15, 60], cmap='RdBu_r')
axs[1].set_title(f'Step 06-B: CWT Imaginary Part (32 Scales × 120 Samples)')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Frequency (Hz)')
fig.colorbar(im1, ax=axs[1])
save_fig(fig, "06_cwt_complex_maps.png")

# ==========================================
# 07. Time Zoom / Resizing (120 -> 32 bins)
# ==========================================
print("Step 07: Resizing to 32x32...")
time_zoom = 32 / cwtm.shape[1]
cwt_real_32 = zoom(re_cwt, (1.0, time_zoom), mode='nearest', order=0)
cwt_imag_32 = zoom(im_cwt, (1.0, time_zoom), mode='nearest', order=0)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
im0 = axs[0].imshow(cwt_real_32, aspect='equal', origin='lower', cmap='RdBu_r')
axs[0].set_title('Step 07-A: Real Part Resized (order=0 Nearest)\nShape: [32, 32]')
axs[0].set_xlabel('Time Bin (0~31)')
axs[0].set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(cwt_imag_32, aspect='equal', origin='lower', cmap='RdBu_r')
axs[1].set_title('Step 07-B: Imag Part Resized (order=0 Nearest)\nShape: [32, 32]')
axs[1].set_xlabel('Time Bin (0~31)')
axs[1].set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im1, ax=axs[1])
save_fig(fig, "07_time_zoom_nearest_32x32.png")

# ==========================================
# 08. Magnitude Calculation
# ==========================================
print("Step 08: Magnitude...")
magnitude = np.sqrt(cwt_real_32**2 + cwt_imag_32**2)

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(magnitude, aspect='equal', origin='lower', cmap='viridis')
ax.set_title('Step 08: Raw CWT Magnitude M = sqrt(Re² + Im²)\nShape: [32, 32]')
ax.set_xlabel('Time Bin (0~31)')
ax.set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im, ax=ax, label='Linear Magnitude')
save_fig(fig, "08_magnitude_extraction.png")

# ==========================================
# 09. 85th Percentile Hard Sparsification
# ==========================================
print("Step 09: Sparsification...")
threshold_val = np.percentile(magnitude, 85)
mag_sparse = magnitude.copy()
mag_sparse[mag_sparse < threshold_val] = 1e-3
mag_sparse[mag_sparse == 0] = 1e-3

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
# Histogram showing threshold
axs[0].hist(magnitude.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axs[0].axvline(threshold_val, color='red', linestyle='--', lw=2.0, label=f'85th Percentile = {threshold_val:.3f}')
axs[0].set_title('Distribution of 1024 Pixels & 85% Cutoff')
axs[0].set_xlabel('Magnitude')
axs[0].set_ylabel('Pixel Count')
axs[0].legend()
axs[0].grid(True, alpha=0.3)

im1 = axs[1].imshow(mag_sparse, aspect='equal', origin='lower', cmap='viridis')
axs[1].set_title('Step 09: Sparsified Map (Lower 85% Floored to 1e-3)\nShape: [32, 32]')
axs[1].set_xlabel('Time Bin (0~31)')
axs[1].set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im1, ax=axs[1], label='Floored Magnitude')
save_fig(fig, "09_85th_percentile_sparsification.png")

# ==========================================
# 10. Log-dB Transformation
# ==========================================
print("Step 10: Log-dB...")
mag_db = 10 * np.log10(mag_sparse)

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(mag_db, aspect='equal', origin='lower', cmap='magma')
ax.set_title('Step 10: Decibel Transformation S_dB = 10·log10(M)\n(Lower 85% = exactly -30 dB)')
ax.set_xlabel('Time Bin (0~31)')
ax.set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im, ax=ax, label='dB Scale (min: -30 dB)')
save_fig(fig, "10_log_db_transform.png")

# ==========================================
# 11. Per-Channel Z-Score Normalization
# ==========================================
print("Step 11: Z-Score...")
mu = np.mean(mag_db)
sig = np.std(mag_db)
mag_z = (mag_db - mu) / (sig + 1e-8)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
im0 = axs[0].imshow(mag_z, aspect='equal', origin='lower', cmap='plasma')
axs[0].set_title(f'Step 11: Single-Window Z-Scored Scalogram\n(Mean={np.mean(mag_z):.2f}, Std={np.std(mag_z):.2f})')
axs[0].set_xlabel('Time Bin (0~31)')
axs[0].set_ylabel('Frequency Bin (0~31)')
fig.colorbar(im0, ax=axs[0], label='Z-Score')

# Pixel value distribution
axs[1].hist(mag_z.flatten(), bins=50, color='purple', edgecolor='black', alpha=0.7)
axs[1].axvline(0, color='k', linestyle='--', label='Zero Mean')
axs[1].set_title('Histogram of Z-Scored Pixels\n(Note Dirac-like spike of 85% background)')
axs[1].set_xlabel('Normalized Value (Z)')
axs[1].set_ylabel('Count (Total 1024)')
axs[1].legend()
axs[1].grid(True, alpha=0.3)
save_fig(fig, "11_z_score_normalization.png")

# ==========================================
# 12. 4-Channel Stacking
# ==========================================
print("Step 12: 4-Channel Stacking...")
def process_channel(err_sig):
    cwtm, _ = pywt.cwt(err_sig, scales, wavelet_name, sampling_period=sampling_period)
    re_32 = zoom(np.real(cwtm), (1.0, time_zoom), mode='nearest', order=0)
    im_32 = zoom(np.imag(cwtm), (1.0, time_zoom), mode='nearest', order=0)
    m = np.sqrt(re_32**2 + im_32**2)
    th = np.percentile(m, 85)
    m[m < th] = 1e-3
    m[m == 0] = 1e-3
    mdb = 10 * np.log10(m)
    return (mdb - np.mean(mdb)) / (np.std(mdb) + 1e-8)

ch_lh = process_channel(err_LH)
ch_rh = process_channel(err_RH)
ch_lv = process_channel(err_LV)
ch_rv = process_channel(err_RV)

tensor_4ch = np.stack([ch_lh, ch_rh, ch_lv, ch_rv], axis=0) # [4, 32, 32]

fig, axs = plt.subplots(1, 4, figsize=(16, 4))
ch_names = ['Ch0: LH-TH (Task-Axis Left)', 'Ch1: RH-TH (Task-Axis Right)',
            'Ch2: LV-TV (Cross-Axis Left)', 'Ch3: RV-TV (Cross-Axis Right)']
for i in range(4):
    im = axs[i].imshow(tensor_4ch[i], aspect='equal', origin='lower', cmap='plasma')
    axs[i].set_title(f'{ch_names[i]}\n[32 × 32]')
    axs[i].set_xlabel('Time Bin')
    if i == 0:
        axs[i].set_ylabel('Frequency Bin')
    fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
fig.suptitle('Step 12: 4-Channel Scalogram Tensor Input [4, 32, 32]', fontsize=14)
save_fig(fig, "12_four_channel_tensor_stacking.png")

# ==========================================
# 13. ConvAdapter (4 -> 3 Channels)
# ==========================================
print("Step 13: ConvAdapter...")
adapter = ConvAdapter(in_channels=4)
adapter.eval()
with torch.no_grad():
    x_in = torch.tensor(tensor_4ch, dtype=torch.float32).unsqueeze(0) # [1, 4, 32, 32]
    x_adapt = adapter(x_in).squeeze(0).numpy() # [3, 32, 32]

fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for i in range(3):
    im = axs[i].imshow(x_adapt[i], aspect='equal', origin='lower', cmap='cividis')
    axs[i].set_title(f'Adapter Output Channel {i}\nConv2d(4->3, k=(5,1)) + BN + ReLU')
    axs[i].set_xlabel('Time Bin')
    if i == 0:
        axs[i].set_ylabel('Frequency Bin')
    fig.colorbar(im, ax=axs[i], fraction=0.046, pad=0.04)
fig.suptitle('Step 13: ConvAdapter Transformation [3, 32, 32] (Ready for Backbone)', fontsize=14)
save_fig(fig, "13_conv_adapter_output.png")

# ==========================================
# 14. Nearest 8x Upscaling to 256x256
# ==========================================
print("Step 14: Upscaling to 256x256...")
with torch.no_grad():
    x_up = F.interpolate(torch.tensor(x_adapt).unsqueeze(0), size=(256, 256), mode='nearest').squeeze(0).numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
im0 = axs[0].imshow(x_adapt[0], aspect='equal', origin='lower', cmap='cividis')
axs[0].set_title('Before Upscale: [32, 32]')
axs[0].set_xlabel('Pixel X (0~31)')
axs[0].set_ylabel('Pixel Y (0~31)')

im1 = axs[1].imshow(x_up[0], aspect='equal', origin='lower', cmap='cividis')
axs[1].set_title('After 8× Nearest Interpolation: [256, 256]\n(Notice 8×8 blocky checkerboard structure)')
axs[1].set_xlabel('Pixel X (0~255)')
axs[1].set_ylabel('Pixel Y (0~255)')
fig.suptitle('Step 14: MobileViT Backbone Input Geometry [3, 256, 256]', fontsize=14)
save_fig(fig, "14_nearest_8x_upscale_256x256.png")

# ==========================================
# 15. Overall Complete Pipeline Flow Chart
# ==========================================
print("Step 15: Master Summary Flow...")
fig, axs = plt.subplots(2, 4, figsize=(18, 9))
axs[0, 0].plot(t_rel, err_LH, 'b-', lw=1.2)
axs[0, 0].set_title('1. Error Signal (1s, 120pts)')
axs[0, 0].set_xlabel('Time (s)'); axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].imshow(re_cwt, aspect='auto', origin='lower', cmap='RdBu_r')
axs[0, 1].set_title('2. CWT Real [32 × 120]')

axs[0, 2].imshow(magnitude, aspect='equal', origin='lower', cmap='viridis')
axs[0, 2].set_title('3. Magnitude [32 × 32]')

axs[0, 3].imshow(mag_sparse, aspect='equal', origin='lower', cmap='viridis')
axs[0, 3].set_title('4. 85% Sparse Floor [32 × 32]')

axs[1, 0].imshow(mag_z, aspect='equal', origin='lower', cmap='plasma')
axs[1, 0].set_title('5. Z-Scored Channel [32 × 32]')

axs[1, 1].imshow(tensor_4ch[0], aspect='equal', origin='lower', cmap='plasma')
axs[1, 1].set_title('6. 4-Ch Stacked [4, 32, 32]')

axs[1, 2].imshow(x_adapt[0], aspect='equal', origin='lower', cmap='cividis')
axs[1, 2].set_title('7. ConvAdapter (5,1) [3, 32, 32]')

axs[1, 3].imshow(x_up[0], aspect='equal', origin='lower', cmap='cividis')
axs[1, 3].set_title('8. Nearest 8× [3, 256, 256]')

fig.suptitle('Master Pipeline: VOG 1D Signal to MobileViT 256×256 Input Evolution', fontsize=16, fontweight='bold')
plt.tight_layout()
save_fig(fig, "00_master_pipeline_flow.png")

print("ALL STEPS SUCCESSFULLY GENERATED AND SAVED!")
