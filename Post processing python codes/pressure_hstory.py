import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# =======================================
# FILE PATH
# =======================================
file_path = "last_folder/re-run/0/postProcessing/p_taps_all.txt"

# =======================================
# LOAD DATA
# =======================================
df = pd.read_csv(file_path, delim_whitespace=True)

# =======================================
# SELECT TIME RANGE
# =======================================
t_start = 0.0
t_end   = 0.4

df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]

time = df["time"].values
pressure = df["tap5"].values

# =======================================
# SPIKE REMOVAL (GRADIENT-BASED - BEST)
# =======================================

# Compute gradient
grad = np.abs(np.gradient(pressure, time))

# Threshold for sharp spikes
grad_threshold = 215 * np.median(grad)

spikes = grad > grad_threshold

# Expand slightly (remove 1–2 points around spike)
expanded_spikes = spikes.copy()

for i in np.where(spikes)[0]:
    start = max(0, i-1)
    end   = min(len(spikes), i+2)
    expanded_spikes[start:end] = True

# Replace spikes with NaN
pressure_filtered = pressure.copy()
pressure_filtered[expanded_spikes] = np.nan

# Interpolate
pressure_filtered = (
    pd.Series(pressure_filtered)
    .interpolate()
    .bfill()
    .ffill()
    .values
)
# =======================================
# PLOT
# =======================================
plt.figure(figsize=(10, 5))

# Optional: plot raw signal
# plt.plot(time, pressure, alpha=0.3, label="Original signal")

plt.plot(time, pressure_filtered, linewidth=2, label="Instantaneous Pressure at tap-5")

plt.xlabel("Time (s)")
plt.ylabel("Pressure (Pa)")
plt.title("Pressure Time History at Tap 5 ")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("tap5_spike_removed.png", dpi=300)

plt.show()