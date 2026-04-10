import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.signal import find_peaks

# ===== EDIT THIS =====
case_dir = Path(r"0")
# =====================

coeff_file = Path("last_folder")/"re-run"/case_dir/"postProcessing"/"forceCoeffs"/"0"/"coefficient.dat"

data = np.loadtxt(coeff_file, comments="#")

time = data[:,0]
Cl_raw = data[:,4]

# ------------------------------------------------
# ROBUST SPIKE DETECTION (MAD + GRADIENT)
# ------------------------------------------------

# --- MAD detection ---
median_cl = np.median(Cl_raw)
mad = np.median(np.abs(Cl_raw - median_cl))

if mad == 0:
    mad = 1e-12

mad_threshold = 3 * mad
mad_mask = np.abs(Cl_raw - median_cl) > mad_threshold

# --- Gradient detection ---
dCl = np.gradient(Cl_raw, time)

grad_threshold = 8 * np.median(np.abs(dCl))
grad_mask = np.abs(dCl) > grad_threshold

# --- Combine both ---
spike_mask = mad_mask | grad_mask

# ------------------------------------------------
# EXPAND SPIKE REGION
# ------------------------------------------------

expanded_spike_mask = spike_mask.copy()

buffer = 5   # 🔥 increased (important)

for i in np.where(spike_mask)[0]:
    start = max(0, i-buffer)
    end   = min(len(spike_mask), i+buffer+1)
    expanded_spike_mask[start:end] = True

# ------------------------------------------------
# INTERPOLATE CLEAN SIGNAL
# ------------------------------------------------

Cl = Cl_raw.copy()

Cl[expanded_spike_mask] = np.interp(
    time[expanded_spike_mask],
    time[~expanded_spike_mask],
    Cl_raw[~expanded_spike_mask]
)

print("Spike points removed (robust):", np.sum(expanded_spike_mask))
# ------------------------------------------------
# PEAK DETECTION
# ------------------------------------------------

peaks, _ = find_peaks(Cl)

# Remove peaks inside spike regions
valid_peaks = [p for p in peaks if not expanded_spike_mask[p]]

peak_times = time[valid_peaks]
peak_values = Cl[valid_peaks]

# Remove small oscillation peaks
Cl_mean = np.mean(Cl)

dominant_mask = peak_values > (Cl_mean + 0.4)

peak_times = peak_times[dominant_mask]
peak_values = peak_values[dominant_mask]

print("\nDominant Oscillation Peaks:\n")

for tpk, cpk in zip(peak_times, peak_values):
    print(f"Cl_max = {cpk:.6f} at t = {tpk:.6f} s")

# ------------------------------------------------
# PLOT
# ------------------------------------------------

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(time, Cl, alpha=0.6, linewidth=1.4, label="Instantaneous $C_L(t)$")

ax.plot(
    peak_times,
    peak_values,
    linestyle='None',
    marker='x',
    markersize=7,
    markeredgewidth=1.6,
    color='red',
    label="Dominant oscillation maxima"
)

for tpk, cpk in zip(peak_times, peak_values):

    ax.annotate(
        rf"$C_L = {cpk:.3f}$"+"\n"+rf"$t = {tpk:.3f}s$",
        xy=(tpk, cpk),
        xytext=(0,25),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3",
                  facecolor="white",
                  edgecolor="black"),
        arrowprops=dict(arrowstyle="-",linewidth=0.8)
    )

ax.set_xlabel("Time [s]")
ax.set_ylabel("$C_L$")

ax.set_ylim(0, 2.0)
ax.grid(True, alpha=0.4)
ax.legend()

ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()