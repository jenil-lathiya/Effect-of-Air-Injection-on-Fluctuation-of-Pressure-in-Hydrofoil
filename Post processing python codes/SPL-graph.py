"""
1/3-octave band SPL comparison at Tap 5
for different air injection rates.

Input  : p_taps_all.txt (time + tap pressures)
Output : Smooth 1/3-octave SPL plot (presentation-ready)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path

# ================= USER INPUT =================
ROOT = Path("last_folder")

q_values = ["0" ,"0.1", "0.2","0.3", "0.4" , "0.5", "0.6"]   # cases to compare
tap_index = 4                  # tap5 (0=time, 1=tap1, ..., 5=tap5)

cutoff_hz = 100.0               # high-pass cutoff
fmin_plot = 50                  # Hz
fmax_plot = 2000                # Hz
p_ref = 1e-6                    # reference pressure (water)
fig_name = "SPL_1_3_octave_tap5.png"
# ==============================================

colors = ["blue", "red", "green", "orange","brown","purple", "black"]

# ---------- Filters ----------
def highpass_filter(signal, fs, cutoff):
    wn = cutoff / (fs / 2.0)
    b, a = butter(4, wn, btype="high")
    return filtfilt(b, a, signal)

# ---------- 1/3-octave bands ----------
def third_octave_spl(freq, P_mag, p_ref):
    f_centers = np.array([
        50, 63, 80, 100, 125, 160, 200, 250, 315, 400,
        500, 630, 800, 1000, 1250, 1600, 2000
    ])

    SPL = []

    for fc in f_centers:
        f1 = fc / (2 ** (1/6))
        f2 = fc * (2 ** (1/6))
        mask = (freq >= f1) & (freq <= f2)

        if np.any(mask):
            p_rms = np.sqrt(np.sum(P_mag[mask]**2))
            SPL.append(20 * np.log10(p_rms / p_ref + 1e-20))
        else:
            SPL.append(np.nan)

    return f_centers, np.array(SPL)

# ---------------- MAIN PLOT ----------------
plt.figure(figsize=(12, 9))

for i, q in enumerate(q_values):

    file_path = ROOT / "re-run"/q / "postProcessing" / "p_taps_all.txt"
    print(f"Processing {file_path}")

    data = np.loadtxt(file_path, skiprows=1)

    t = data[:, 0]
    p = data[:, tap_index]

    # Sampling
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    N = len(p)

    # Pressure fluctuations
    p_fluc = p - np.mean(p)

    # High-pass filter
    p_hp = highpass_filter(p_fluc, fs, cutoff_hz)

    # Windowing
    window = np.hanning(N)
    p_win = p_hp * window

    # FFT
    P = np.fft.rfft(p_win)
    freq = np.fft.rfftfreq(N, d=dt)

    scale = 2.0 / np.sum(window)
    P_mag = np.abs(P) * scale

    # 1/3-octave SPL
    f_centers, SPL_13 = third_octave_spl(freq, P_mag, p_ref)

    plt.semilogx(
        f_centers, SPL_13,
        marker='o', linewidth=1,
        label=q, color=colors[i]
    )

# ---------- Plot formatting ----------
plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB re $10^{-6}$ Pa]")
plt.title("1/3-Octave Band SPL at Tap 5")
plt.xlim(fmin_plot, fmax_plot)
plt.grid(True, which="both", linestyle="--", linewidth=0.6)
plt.legend(title="Injection rate")
plt.tight_layout()
plt.savefig(fig_name, dpi=300)
plt.show()

print(f"\nSaved figure as {fig_name}")
