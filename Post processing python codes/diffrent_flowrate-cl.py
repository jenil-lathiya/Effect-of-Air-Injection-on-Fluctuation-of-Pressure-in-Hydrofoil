import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from matplotlib.patches import Rectangle
# ==========================
# USER SETTINGS
# ==========================

t_start = 0.3
t_end   = 0.4

cases = ["0","0.1","0.2","0.3","0.4","0.5","0.6"]

ROOT = Path("last_folder")

# ==========================
# ROBUST FILTER FUNCTION
# ==========================

def robust_spike_filter(time, Cl_raw):

    # --- MAD detection (less aggressive) ---
    median_cl = np.median(Cl_raw)
    mad = np.median(np.abs(Cl_raw - median_cl))

    if mad == 0:
        mad = 1e-12

    mad_threshold = 6 * mad   # 🔥 increased (was 3)
    mad_mask = np.abs(Cl_raw - median_cl) > mad_threshold

    # --- Gradient detection (relaxed) ---
    dCl = np.gradient(Cl_raw, time)
    grad_threshold = 12 * np.median(np.abs(dCl))  # 🔥 increased
    grad_mask = np.abs(dCl) > grad_threshold

    # --- Combine ---
    spike_mask = mad_mask | grad_mask

    # --- Expand spike region (smaller buffer) ---
    expanded_mask = spike_mask.copy()
    buffer = 2   # 🔥 reduced (was 5)

    for i in np.where(spike_mask)[0]:
        start = max(0, i - buffer)
        end   = min(len(Cl_raw), i + buffer + 1)
        expanded_mask[start:end] = True

    # --- Interpolation ---
    Cl_clean = Cl_raw.copy()
    good = ~expanded_mask

    if np.sum(good) > 2:
        Cl_clean[expanded_mask] = np.interp(
            time[expanded_mask],
            time[good],
            Cl_raw[good]
        )

    return Cl_clean

# ==========================
# PLOT
# ==========================

plt.figure(figsize=(12,6))

for case in cases:

    coeff_file = ROOT / "re-run" / case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"Skipping {case}")
        continue

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:,0]
    Cl_raw = data[:,4]

    # -------------------------
    # APPLY FILTER
    # -------------------------
    Cl = robust_spike_filter(time, Cl_raw)

    # -------------------------
    # TIME WINDOW
    # -------------------------
    mask = (time >= t_start) & (time <= t_end)

    time_w = time[mask]
    Cl_w   = Cl[mask]

    # -------------------------
    # PLOT
    # -------------------------
    plt.plot(time_w, Cl_w, linewidth=1.5, label=f"q = {case}")


ax = plt.gca()
# Define zoom region
x1, x2 = 0.342, 0.35   # 🔁 change as needed
y1, y2 = 1.3, 1.39      # 🔁 adjust based on your plot

# Add rectangle
rect = Rectangle(
    (x1, y1),           # bottom-left corner
    x2 - x1,            # width
    y2 - y1,            # height
    linewidth=2,
    edgecolor='black',
    facecolor='none',
    linestyle='--'
)

ax.add_patch(rect)
# ==========================
# FORMAT
# ==========================

plt.xlabel("Time [s]")
plt.ylabel("$C_L$")
plt.title("Filtered Lift Coefficient vs Time (All Injection Rates)")

plt.grid(True, alpha=0.3)
plt.legend(ncol=2)

plt.tight_layout()
plt.show()
