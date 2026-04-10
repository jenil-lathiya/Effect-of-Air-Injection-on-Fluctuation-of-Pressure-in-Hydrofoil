import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.signal import find_peaks

# ====== EDIT ONLY THIS ======
case_dir = Path(r"0.5")

# --- Averaging window ---
t_avg_start = 0.1600
t_avg_end   = 0.3925

# --- Peak detection window ---
t_peak_start = 0.14
t_peak_end   = 0.36
# =============================

coeff_file = Path("last_folder") /"re-run"/ case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
if not coeff_file.exists():
    raise FileNotFoundError(f"File not found: {coeff_file}")

data = np.loadtxt(coeff_file, comments="#")

time = data[:, 0]
Cd   = data[:, 1]   # TOTAL Cd

# =====================================================
# --- TIME AVERAGING ---
# =====================================================
avg_mask = (time >= t_avg_start) & (time <= t_avg_end)
tt_avg = time[avg_mask]
yy_avg = Cd[avg_mask]

if len(tt_avg) < 2:
    raise ValueError("Not enough points in averaging window")

dt = np.diff(tt_avg)
cum_int = np.concatenate([[0.0],
                          np.cumsum(0.5 * (yy_avg[1:] + yy_avg[:-1]) * dt)])
Cd_runavg = cum_int / (tt_avg - t_avg_start + 1e-30)

Cd_mean = np.trapz(yy_avg, tt_avg) / (t_avg_end - t_avg_start)

print(f"Time-averaged Cd over [{t_avg_start}, {t_avg_end}] = {Cd_mean:.6f}")

# =====================================================
# --- PEAK DETECTION ---
# =====================================================
peak_mask = (time >= t_peak_start) & (time <= t_peak_end)
tt_peak = time[peak_mask]
yy_peak = Cd[peak_mask]

if len(tt_peak) < 2:
    raise ValueError("Not enough points in peak window")

# --- Robust spike removal (MAD) ---
median_cd = np.median(yy_peak)
mad = np.median(np.abs(yy_peak - median_cd))
if mad == 0:
    mad = 1e-12

threshold = 6 * mad
non_spike_mask = np.abs(yy_peak - median_cd) < threshold

Cd_clean = yy_peak[non_spike_mask]
time_clean = tt_peak[non_spike_mask]

# --- Light smoothing ---
kernel = np.ones(11) / 11
Cd_smooth = np.convolve(Cd_clean, kernel, mode='same')

# --- Detect peaks ---
peaks, _ = find_peaks(Cd_smooth)

peak_times = time_clean[peaks]
peak_values = Cd_smooth[peaks]

# =====================================================
# --- KEEP ONLY UPPER ENVELOPE PEAKS (~0.19 REGION) ---
# =====================================================
if len(peak_values) > 0:
    global_max = np.max(peak_values)

    # Keep peaks within 95% of maximum
    upper_mask = peak_values >= 0.95 * global_max

    peak_times = peak_times[upper_mask]
    peak_values = peak_values[upper_mask]

# Sort by time
sort_idx = np.argsort(peak_times)
peak_times = peak_times[sort_idx]
peak_values = peak_values[sort_idx]

print("\nUpper Envelope Cd Peaks:")
for t_peak, cd_peak in zip(peak_times, peak_values):
    print(f"Cd_max = {cd_peak:.6f} at t = {t_peak:.6f} s")

# =====================================================
# ------------------ PLOTTING -------------------------
# =====================================================
fig, ax = plt.subplots(figsize=(10, 5))

# Instantaneous Cd
ax.plot(time, Cd, alpha=0.35, label="Instantaneous $C_D(t)$")

# Running average
ax.plot(tt_avg, Cd_runavg, linewidth=2.2,
        label=f"Running avg [{t_avg_start} → {t_avg_end}]")

# Mean line
ax.axhline(Cd_mean, linewidth=2.0,
           label=f"Mean $C_D$ = {Cd_mean:.4f}")

# Mark upper envelope peaks
ax.plot(peak_times, peak_values,
        linestyle='None',
        marker='x',
        markersize=6,
        markeredgewidth=1.5,
        color='red',
        label="Upper envelope maxima")

# Professional annotation boxes
for i, (t_peak, cd_peak) in enumerate(zip(peak_times, peak_values)):

    offset_y = 25 if i % 2 == 0 else -35

    ax.annotate(
        rf"$C_D = {cd_peak:.5f}$" + "\n" +
        rf"$t = {t_peak:.3f}\,\mathrm{{s}}$",
        xy=(t_peak, cd_peak),
        xytext=(0, offset_y),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        va="bottom" if i % 2 == 0 else "top",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="white",
            edgecolor="black",
            linewidth=0.8
        ),
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.8,
            color="black"
        )
    )

# Shade windows
ax.axvspan(t_avg_start, t_avg_end, alpha=0.12)
ax.axvspan(t_peak_start, t_peak_end, alpha=0.18)

ax.set_xlabel("Time [s]")
ax.set_ylabel("$C_D$")
ax.grid(True, alpha=0.4)
ax.legend()

ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()
