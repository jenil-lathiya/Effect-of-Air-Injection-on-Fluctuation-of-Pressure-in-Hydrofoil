import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks

# =============================
# USER INPUT
# =============================

case_dir = Path("0.1")

t_steady_start = 0.39      # Start of statistically steady region
n_cycles = 1              # Number of full cycles to average

# =============================

coeff_file = Path("last_folder")/"re-run"/case_dir/"postProcessing"/"forceCoeffs"/"0"/"coefficient.dat"
data = np.loadtxt(coeff_file, comments="#")

time = data[:,0]
Cl   = data[:,4]

# =====================================================
# STEP 1 — EXTRACT STEADY REGION
# =====================================================

steady_mask = time >= t_steady_start
t_steady = time[steady_mask]
Cl_steady = Cl[steady_mask]

dt = t_steady[1] - t_steady[0]

# Remove mean for FFT
Cl_fluct = Cl_steady - np.mean(Cl_steady)

# =====================================================
# STEP 2 — FFT FOR DOMINANT FREQUENCY
# =====================================================

yf = np.abs(rfft(Cl_fluct))
xf = rfftfreq(len(Cl_fluct), dt)

# Remove zero frequency
xf = xf[1:]
yf = yf[1:]

f_dom = xf[np.argmax(yf)]
T = 1.0 / f_dom

print(f"Dominant frequency = {f_dom:.4f} Hz")
print(f"Dominant period T = {T:.6f} s")

# =====================================================
# STEP 3 — BUILD INTEGER-CYCLE WINDOW
# =====================================================

t_start = t_steady_start
t_end   = t_start + n_cycles * T

avg_mask = (time >= t_start) & (time <= t_end)

t_avg = time[avg_mask]
Cl_avg_window = Cl[avg_mask]

# =====================================================
# STEP 4 — COMPUTE MEAN AND RMS
# =====================================================

Cl_mean = np.trapz(Cl_avg_window, t_avg) / (t_end - t_start)
Cl_rms  = np.sqrt(np.trapz((Cl_avg_window - Cl_mean)**2, t_avg) / (t_end - t_start))

print(f"\nAveraging window: [{t_start:.4f}, {t_end:.4f}]")
print(f"Mean Cl  = {Cl_mean:.6f}")
print(f"RMS  Cl  = {Cl_rms:.6f}")

# =====================================================
# STEP 5 — PLOT
# =====================================================

plt.figure(figsize=(10,5))

plt.plot(time, Cl, alpha=0.35, label="Instantaneous $C_L$")
plt.axvspan(t_start, t_end, alpha=0.15, label="Integer-cycle window")
plt.axhline(Cl_mean, linewidth=2, label=f"Mean $C_L$ = {Cl_mean:.4f}")

plt.xlabel("Time [s]")
plt.ylabel("$C_L$")
plt.grid(alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()