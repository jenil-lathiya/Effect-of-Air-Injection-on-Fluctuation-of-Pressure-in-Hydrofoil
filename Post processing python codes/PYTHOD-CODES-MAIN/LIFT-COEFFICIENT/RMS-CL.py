import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# USER SETTINGS
# =====================================================
cases = ["0", "0.1", "0.2", "0.3", "0.4", "0.5","0.6","0.7"]

t_start = 0.3
t_end   = 0.394

ROOT = Path("last_folder")

MAD_K = 5.0   # strength of spike detection

# =====================================================
# MAD DESPIKE FUNCTION
# =====================================================
def mad_despike_with_interp(t, y, k=MAD_K):

    median = np.median(y)
    mad = np.median(np.abs(y - median))

    if mad == 0:
        mad = 1e-12

    threshold = k * mad

    spike_mask = np.abs(y - median) > threshold

    y_filtered = y.copy()

    if np.any(spike_mask):

        good = ~spike_mask

        if np.sum(good) >= 2:
            y_filtered[spike_mask] = np.interp(
                t[spike_mask],
                t[good],
                y[good]
            )

    return y_filtered


# =====================================================
# STORAGE
# =====================================================
injection_rates = []
Cl_mean_all = []
Cl_rms_all = []

# =====================================================
# LOOP OVER CASES
# =====================================================
for case_dir in cases:

    coeff_file = ROOT / "re-run" / case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"File not found for case {case_dir}")
        continue

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:, 0]
    Cl_raw = data[:, 4]

    # -------------------------------------------------
    # APPLY MAD FILTER
    # -------------------------------------------------
    Cl = mad_despike_with_interp(time, Cl_raw)

    # -------------------------------------------------
    # STEADY WINDOW
    # -------------------------------------------------
    mask = (time >= t_start) & (time <= t_end)

    if np.sum(mask) < 10:
        print(f"Not enough steady data for {case_dir}")
        continue

    t_window = time[mask]
    Cl_window = Cl[mask]

    T = t_end - t_start

    # -------------------------------------------------
    # MEAN CL
    # -------------------------------------------------
    Cl_mean = np.trapz(Cl_window, t_window) / T

    # -------------------------------------------------
    # FLUCTUATION COMPONENT
    # -------------------------------------------------
    Cl_fluct = Cl_window - Cl_mean

    # -------------------------------------------------
    # RMS OF FLUCTUATIONS
    # -------------------------------------------------
    rms = np.sqrt(
        np.trapz(Cl_fluct**2, t_window) / T
    )

    injection_rates.append(float(case_dir))
    Cl_mean_all.append(Cl_mean)
    Cl_rms_all.append(rms)

# =====================================================
# SORT RESULTS
# =====================================================
injection_rates = np.array(injection_rates)
Cl_mean_all = np.array(Cl_mean_all)
Cl_rms_all = np.array(Cl_rms_all)

sort_idx = np.argsort(injection_rates)

injection_rates = injection_rates[sort_idx]
Cl_mean_all = Cl_mean_all[sort_idx]
Cl_rms_all = Cl_rms_all[sort_idx]

# =====================================================
# PRINT RESULTS
# =====================================================
print("\nInjection Rate | Mean Cl | RMS Cl'")
print("--------------------------------------")
for q, mean, rms in zip(injection_rates, Cl_mean_all, Cl_rms_all):
    print(f"{q:.2f}            {mean:.5f}     {rms:.5f}")

# =====================================================
# PLOT RMS vs INJECTION RATE
# =====================================================
plt.figure(figsize=(6,4))

plt.plot(
    injection_rates,
    Cl_rms_all,
    marker='o',
    linewidth=2
)

plt.xlabel("Injection Rate")
plt.ylabel("RMS of $C_L'$")

plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()