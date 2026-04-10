import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# USER INPUT
# =====================================================

t_start = 0.17
t_end   = 0.37

cases = ["0", "0.1","0.2", "0.3", "0.4" ,"0.5"]

ROOT = Path("last_folder")

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

    coeff_file = ROOT /"re-run"/ case / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"Skipping {case} (file not found)")
        continue

    print(f"Processing {case}")

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:, 0]
    Cd   = data[:, 1]
    Cl   = data[:, 4]
    Cm   = data[:, 7]   # CmPitch

    mask = (time >= t_start) & (time <= t_end)

    if np.sum(mask) < 2:
        print(f"Not enough data in window for {case}")
        continue

    t_window = time[mask]

    Cd_avg = np.trapz(Cd[mask], t_window) / (t_end - t_start)
    Cl_avg = np.trapz(Cl[mask], t_window) / (t_end - t_start)
    Cm_avg = np.trapz(Cm[mask], t_window) / (t_end - t_start)

    injection_rates.append(float(case.replace("q=", "")))
    Cl_avg_all.append(Cl_avg)
    Cd_avg_all.append(Cd_avg)
    Cm_avg_all.append(Cm_avg)

# Convert to arrays and sort
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

print("\nAveraged Values:")
for q, cl, cd, cm in zip(injection_rates, Cl_avg_all, Cd_avg_all, Cm_avg_all):
    print(f"q = {q:.2f} | Cl = {cl:.5f} | Cd = {cd:.5f} | Cm = {cm:.5f}")

# =====================================================
# PLOT 1: CL
# =====================================================

plt.figure(figsize=(6, 4))
plt.plot(injection_rates, Cl_avg_all, marker='o', linewidth=2)
plt.xlabel("Injection Rate")
plt.ylabel("Mean $C_L$")
plt.title(f"Time-Averaged $C_L$ ({t_start} → {t_end} s)")
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
plt.title(f"Time-Averaged $C_D$ ({t_start} → {t_end} s)")
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
plt.title(f"Time-Averaged $C_M$ ({t_start} → {t_end} s)")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.show()
