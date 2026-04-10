import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
from scipy.interpolate import PchipInterpolator# =====================================================
# USER INPUT
# =====================================================
t_start = 0.30
t_end   = 0.393

cases = ["0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]

ROOT = Path("last_folder")

MAD_K = 5.0

# =====================================================
# MAD FILTER FUNCTION
# =====================================================
def mad_despike_with_interp(t, y, k=MAD_K):

    m = np.median(y)
    mad = np.median(np.abs(y - m))

    if mad == 0:
        mad = 1e-12

    threshold = k * mad

    spike_mask = np.abs(y - m) > threshold

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
Cd_mean_all = []
Cm_mean_all = []

Cl_rms_all = []
Cd_rms_all = []
Cm_rms_all = []

# =====================================================
# LOOP OVER CASES
# =====================================================
for case in cases:

    coeff_file = ROOT / "re-run" / case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"Skipping {case} (file not found)")
        continue

    print(f"\nProcessing case {case}")

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:,0]

    Cd_raw = data[:,1]
    Cl_raw = data[:,4]
    Cm_raw = data[:,7]

    # --------------------------------------------
    # MAD FILTER
    # --------------------------------------------
    Cd = mad_despike_with_interp(time, Cd_raw)
    Cl = mad_despike_with_interp(time, Cl_raw)
    Cm = mad_despike_with_interp(time, Cm_raw)

    # --------------------------------------------
    # STEADY WINDOW
    # --------------------------------------------
    mask = (time >= t_start) & (time <= t_end)

    if np.sum(mask) < 10:
        print("Not enough steady data")
        continue

    t_window = time[mask]

    Cl_window = Cl[mask]
    Cd_window = Cd[mask]
    Cm_window = Cm[mask]

    T = t_end - t_start

    # --------------------------------------------
    # MEAN VALUES
    # --------------------------------------------
    Cl_mean = np.trapz(Cl_window, t_window) / T
    Cd_mean = np.trapz(Cd_window, t_window) / T
    Cm_mean = np.trapz(Cm_window, t_window) / T

    # --------------------------------------------
    # RMS VALUES
    # --------------------------------------------
    Cl_fluct = Cl_window - Cl_mean
    Cd_fluct = Cd_window - Cd_mean
    Cm_fluct = Cm_window - Cm_mean

    Cl_rms = np.sqrt(np.trapz(Cl_fluct**2, t_window) / T)
    Cd_rms = np.sqrt(np.trapz(Cd_fluct**2, t_window) / T)
    Cm_rms = np.sqrt(np.trapz(Cm_fluct**2, t_window) / T)

    # --------------------------------------------
    # STORE
    # --------------------------------------------
    injection_rates.append(float(case))

    Cl_mean_all.append(Cl_mean)
    Cd_mean_all.append(Cd_mean)
    Cm_mean_all.append(Cm_mean)

    Cl_rms_all.append(Cl_rms)
    Cd_rms_all.append(Cd_rms)
    Cm_rms_all.append(Cm_rms)

# =====================================================
# CONVERT TO NUMPY
# =====================================================
injection_rates = np.array(injection_rates)

Cl_mean_all = np.array(Cl_mean_all)
Cd_mean_all = np.array(Cd_mean_all)
Cm_mean_all = np.array(Cm_mean_all)

Cl_rms_all = np.array(Cl_rms_all)
Cd_rms_all = np.array(Cd_rms_all)
Cm_rms_all = np.array(Cm_rms_all)

# =====================================================
# SORT
# =====================================================
sort_idx = np.argsort(injection_rates)

injection_rates = injection_rates[sort_idx]

Cl_mean_all = Cl_mean_all[sort_idx]
Cd_mean_all = Cd_mean_all[sort_idx]
Cm_mean_all = Cm_mean_all[sort_idx]

Cl_rms_all = Cl_rms_all[sort_idx]
Cd_rms_all = Cd_rms_all[sort_idx]
Cm_rms_all = Cm_rms_all[sort_idx]

# =====================================================
# PRINT TABLE
# =====================================================
print("\n------------------------------------------------------------------")
print("Injection | Mean Cl | RMS Cl | Mean Cd | RMS Cd | Mean Cm | RMS Cm")
print("------------------------------------------------------------------")

for i in range(len(injection_rates)):

    print(f"{injection_rates[i]:>8.2f} | "
          f"{Cl_mean_all[i]:>7.5f} | {Cl_rms_all[i]:>7.5f} | "
          f"{Cd_mean_all[i]:>7.5f} | {Cd_rms_all[i]:>7.5f} | "
          f"{Cm_mean_all[i]:>7.5f} | {Cm_rms_all[i]:>7.5f}")

print("------------------------------------------------------------------")


def smooth_curve(x, y, num=500):
    x_new = np.linspace(x.min(), x.max(), num)
    spline = PchipInterpolator(x, y)   # shape-preserving
    y_smooth = spline(x_new)
    return x_new, y_smooth
# =====================================================
# PLOT MEAN COEFFICIENTS
# =====================================================
x_s, y_s = smooth_curve(injection_rates, Cl_mean_all)

plt.figure(figsize=(10,5))
plt.plot(x_s, y_s, linewidth=2)                  # smooth line
plt.plot(injection_rates, Cl_mean_all, 'o')      # original points

plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_L$")
plt.title("Mean Lift Coefficient vs Injection Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# ✅ SAVE (high-quality landscape)
plt.savefig("Cl_mean_landscape.png", dpi=300, bbox_inches='tight')
plt.show()

x_s, y_s = smooth_curve(injection_rates, Cd_mean_all)

plt.figure(figsize=(10,5))
plt.plot(x_s, y_s, linewidth=2)
plt.plot(injection_rates, Cd_mean_all, 's')

plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_D$")
plt.title("Mean Drag Coefficient vs Injection Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# ✅ SAVE (high-quality landscape)
plt.savefig("Cd_mean_landscape.png", dpi=300, bbox_inches='tight')
plt.show()

x_s, y_s = smooth_curve(injection_rates, Cm_mean_all)

plt.figure(figsize=(10,5))
plt.plot(x_s, y_s, linewidth=2)
plt.plot(injection_rates, Cm_mean_all, '^')

plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_M$")
plt.title("Mean Moment Coefficient vs Injection Rate")
plt.grid(True, alpha=0.3)
plt.tight_layout()
# ✅ SAVE (high-quality landscape)
plt.savefig("Cm_mean_landscape.png", dpi=300, bbox_inches='tight')
plt.show()
# =====================================================
# PLOT RMS COEFFICIENTS
# =====================================================
plt.figure(figsize=(6,4))

plt.plot(injection_rates, Cl_rms_all, marker='o', linewidth=2)

plt.xlabel("Injection Rate")
plt.ylabel("RMS $C_L'$")
plt.title("Lift Coefficient Fluctuation RMS")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))

plt.plot(injection_rates, Cd_rms_all, marker='s', linewidth=2)

plt.xlabel("Injection Rate")
plt.ylabel("RMS $C_D'$")
plt.title("Drag Coefficient Fluctuation RMS")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))

plt.plot(injection_rates, Cm_rms_all, marker='^', linewidth=2)

plt.xlabel("Injection Rate")
plt.ylabel("RMS $C_M'$")
plt.title("Moment Coefficient Fluctuation RMS")

plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()