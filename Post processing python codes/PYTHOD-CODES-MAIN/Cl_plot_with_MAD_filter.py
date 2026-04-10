import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ====== EDIT ONLY THIS ======
case_dir = Path(r"0")
# =============================

coeff_file = Path("last_folder")/"re-run"/case_dir/"postProcessing"/"forceCoeffs"/"0"/"coefficient.dat"

if not coeff_file.exists():
    raise FileNotFoundError(f"File not found: {coeff_file}")

data = np.loadtxt(coeff_file, comments="#")

time   = data[:,0]
Cl_raw = data[:,4]

# =====================================================
# ROBUST SPIKE REMOVAL (MAD + GRADIENT)
# =====================================================

# --- MAD detection ---
median_cl = np.median(Cl_raw)
mad = np.median(np.abs(Cl_raw - median_cl))

if mad == 0:
    mad = 1e-12

mad_threshold = 5 * mad
mad_mask = np.abs(Cl_raw - median_cl) > mad_threshold

# --- GRADIENT detection (key addition) ---
dCl = np.gradient(Cl_raw, time)

grad_threshold = 10 * np.median(np.abs(dCl))   # tune if needed
grad_mask = np.abs(dCl) > grad_threshold

# --- Combine both ---
spike_mask = mad_mask | grad_mask

# --- Expand region (IMPORTANT) ---
expand = 6
expanded_mask = spike_mask.copy()

indices = np.where(spike_mask)[0]

for i in indices:
    start = max(0, i - expand)
    end   = min(len(Cl_raw), i + expand + 1)
    expanded_mask[start:end] = True

# --- Remove spikes completely ---
clean_mask = ~expanded_mask

time_clean = time[clean_mask]
Cl_clean   = Cl_raw[clean_mask]

print(f"Removed {np.sum(expanded_mask)} points (robust spike removal).")
# =====================================================
# PLOT RAW vs FILTERED SIGNAL
# =====================================================

plt.figure(figsize=(10,5))

#plt.plot(time, Cl_raw, alpha=0.4, label="Raw $C_L(t)$")
plt.plot(time_clean, Cl_clean, linewidth=1.8, label="$C_L(t)$")

plt.xlabel("Time [s]")
plt.ylabel("$C_L$")

plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

plt.show()