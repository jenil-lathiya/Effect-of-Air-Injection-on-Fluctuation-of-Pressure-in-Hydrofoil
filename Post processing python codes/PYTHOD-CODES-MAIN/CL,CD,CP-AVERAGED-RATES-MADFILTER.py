import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# USER INPUT
# =====================================================
t_start = 0.17
t_end   = 0.37

cases = ["0", "0.1", "0.2", "0.3", "0.4", "0.5"]
ROOT = Path("last_folder")

# MAD filter strength (professional default)
MAD_K = 6.0

# =====================================================
# HELPER: MAD FILTER + INTERPOLATION
# =====================================================
def mad_despike_with_interp(t, y, k=MAD_K):
    """
    Detects impulsive outliers using MAD and replaces them with linear interpolation.
    Returns: y_filtered, spike_mask
    """
    m = np.median(y)
    mad = np.median(np.abs(y - m))
    if mad == 0:
        mad = 1e-12

    thr = k * mad
    spike_mask = np.abs(y - m) > thr

    y_f = y.copy()
    if np.any(spike_mask):
        # interpolate spikes using non-spike points
        good = ~spike_mask
        # If almost everything is flagged (should not happen), just return original
        if np.sum(good) >= 2:
            y_f[spike_mask] = np.interp(t[spike_mask], t[good], y[good])

    return y_f, spike_mask

# =====================================================
# STORAGE
# =====================================================
injection_rates = []
Cl_avg_all = []
Cd_avg_all = []
Cm_avg_all = []

# =====================================================
# LOOP OVER CASES
# =====================================================
for case in cases:

    coeff_file = ROOT / "re-run" / case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"Skipping {case} (file not found)")
        continue

    print(f"\nProcessing {case}")

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:, 0]
    Cd_raw = data[:, 1]
    Cl_raw = data[:, 4]
    Cm_raw = data[:, 7]  # CmPitch

    # --- Despike each signal (full time series) ---
    Cd, Cd_spikes = mad_despike_with_interp(time, Cd_raw, k=MAD_K)
    Cl, Cl_spikes = mad_despike_with_interp(time, Cl_raw, k=MAD_K)
    Cm, Cm_spikes = mad_despike_with_interp(time, Cm_raw, k=MAD_K)

    print(f"  Spikes replaced (MAD={MAD_K:g}): "
          f"Cd={Cd_spikes.sum()}, Cl={Cl_spikes.sum()}, Cm={Cm_spikes.sum()}")

    # --- Window mask for averaging ---
    mask = (time >= t_start) & (time <= t_end)

    if np.sum(mask) < 2:
        print(f"Not enough data in window for {case}")
        continue

    t_window = time[mask]
    window_len = (t_end - t_start)

    # Time-averages on filtered signals
    Cd_avg = np.trapz(Cd[mask], t_window) / window_len
    Cl_avg = np.trapz(Cl[mask], t_window) / window_len
    Cm_avg = np.trapz(Cm[mask], t_window) / window_len

    injection_rates.append(float(case.replace("q=", "")))
    Cl_avg_all.append(Cl_avg)
    Cd_avg_all.append(Cd_avg)
    Cm_avg_all.append(Cm_avg)

# =====================================================
# Convert to arrays and sort
# =====================================================
injection_rates = np.array(injection_rates)
Cl_avg_all = np.array(Cl_avg_all)
Cd_avg_all = np.array(Cd_avg_all)
Cm_avg_all = np.array(Cm_avg_all)

sort_idx = np.argsort(injection_rates)
injection_rates = injection_rates[sort_idx]
Cl_avg_all = Cl_avg_all[sort_idx]
Cd_avg_all = Cd_avg_all[sort_idx]
Cm_avg_all = Cm_avg_all[sort_idx]

# =====================================================
# PRINT RESULTS
# =====================================================
print("\nAveraged Values (after MAD despiking + interpolation):")
for q, cl, cd, cm in zip(injection_rates, Cl_avg_all, Cd_avg_all, Cm_avg_all):
    print(f"q = {q:.2f} | Cl = {cl:.5f} | Cd = {cd:.5f} | Cm = {cm:.5f}")

# =====================================================
# PLOT 1: CL
# =====================================================
plt.figure(figsize=(6, 4))
plt.plot(injection_rates, Cl_avg_all, marker='o', linewidth=2)
plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_L$")
plt.title(f"Time-Averaged $C_L$ ({t_start}–{t_end} s) [MAD filtered]")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# =====================================================
# PLOT 2: CD
# =====================================================
plt.figure(figsize=(6, 4))
plt.plot(injection_rates, Cd_avg_all, marker='s', linewidth=2)
plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_D$")
plt.title(f"Time-Averaged $C_D$ ({t_start}–{t_end} s) [MAD filtered]")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()

# =====================================================
# PLOT 3: CM
# =====================================================
plt.figure(figsize=(6, 4))
plt.plot(injection_rates, Cm_avg_all, marker='^', linewidth=2)
plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_M$")
plt.title(f"Time-Averaged $C_M$ ({t_start}–{t_end} s) [MAD filtered]")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()