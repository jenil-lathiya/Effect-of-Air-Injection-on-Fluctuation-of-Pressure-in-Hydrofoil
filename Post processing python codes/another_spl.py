import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================= INPUT =================
ROOT = Path("last_folder/re-run")

q_values = ["0","0.1","0.2","0.3","0.4","0.5","0.6"]
tap_index = 5   # same tap for all

t_start = 0.207
t_end   = 0.394
# ========================================

plt.figure(figsize=(12,6))

for q in q_values:

    file_path = ROOT / q / "postProcessing" / "p_taps_all.txt"
    print(f"Processing {q}")

    data = np.loadtxt(file_path, skiprows=1)

    t = data[:,0]
    p = data[:,tap_index]

    # -----------------------------
    # TIME WINDOW
    # -----------------------------
    mask = (t >= t_start) & (t <= t_end)
    t = t[mask]
    p = p[mask]

    # -----------------------------
    # REMOVE MEAN
    # -----------------------------
    p_fluc = p - np.mean(p)

    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    N = len(p_fluc)

    # -----------------------------
    # WINDOW
    # -----------------------------
    window = np.hanning(N)
    p_win = p_fluc * window

    # -----------------------------
    # FFT
    # -----------------------------
    P = np.fft.rfft(p_win)
    freq = np.fft.rfftfreq(N, d=dt)

    P_mag = np.abs(P) * (2.0 / np.sum(window))

    # -----------------------------
    # PLOT
    # -----------------------------
    plt.plot(freq, P_mag, linewidth=1.5, label=f"q = {q}")

# -----------------------------
# PLOT SETTINGS
# -----------------------------
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Frequency Spectrum at Tap (All Injection Rates)")

plt.xlim(0, 200)
plt.grid(True, alpha=0.4)
plt.legend()

plt.tight_layout()
plt.show()