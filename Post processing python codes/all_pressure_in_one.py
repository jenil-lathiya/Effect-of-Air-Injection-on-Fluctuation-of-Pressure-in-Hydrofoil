import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, rfftfreq
from scipy.interpolate import PchipInterpolator

# ----------------------------------------------------
# GLOBAL PLOT STYLE
# ----------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (12,6),
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12
})

# ----------------------------------------------------
# CASE FILE PATHS
# ----------------------------------------------------

cases = {
    "0.0": "last_folder/re-run/0/postProcessing/p_taps_all.txt",
    "0.1": "last_folder/re-run/0.1/postProcessing/p_taps_all.txt",
    "0.2": "last_folder/re-run/0.2/postProcessing/p_taps_all.txt",
    "0.3": "last_folder/re-run/0.3/postProcessing/p_taps_all.txt",
    "0.4": "last_folder/re-run/0.4/postProcessing/p_taps_all.txt",
    "0.5": "last_folder/re-run/0.5/postProcessing/p_taps_all.txt",
    "0.6": "last_folder/re-run/0.6/postProcessing/p_taps_all.txt"
}

# ----------------------------------------------------
# STEADY STATE WINDOW
# ----------------------------------------------------

t_start = 0.3
t_end   = 0.394

fft_tap = "tap4"

# ----------------------------------------------------
# STORAGE
# ----------------------------------------------------

mean_pressures = {}
rms_pressures  = {}

table_results = []

# ----------------------------------------------------
# PROCESS EACH CASE
# ----------------------------------------------------

for inj, filepath in cases.items():

    print("Processing injection rate:", inj)

    df = pd.read_csv(filepath, delim_whitespace=True)

    # steady-state filtering (time window only)
    df = df[(df['time'] >= t_start) & (df['time'] <= t_end)]

    taps = df.columns[1:]

    mean_vals = []
    rms_vals  = []

    for tap in taps:

        p = df[tap].values   # ← no filtering

        mean_p = np.mean(p)
        rms_p  = np.sqrt(np.mean((p - mean_p)**2))

        mean_vals.append(mean_p)
        rms_vals.append(rms_p)

    mean_pressures[inj] = mean_vals
    rms_pressures[inj]  = rms_vals

    # -----------------------------
    # TABLE DATA
    # -----------------------------

    p_table = df[fft_tap].values   # ← no filtering

    mean_table = np.mean(p_table)
    rms_table  = np.sqrt(np.mean((p_table - mean_table)**2))

    table_results.append({
        "Injection Rate": float(inj),
        "Mean Pressure (Pa)": mean_table,
        "RMS Pressure (Pa)": rms_table
    })


def smooth_curve(x, y, num=500):
    x_new = np.linspace(x.min(), x.max(), num)
    spline = PchipInterpolator(x, y)
    y_smooth = spline(x_new)
    return x_new, y_smooth


    # -----------------------------
    # TIME HISTORY
    # -----------------------------

    p_plot = df[fft_tap].values

    plt.figure(figsize=(10,6))
    plt.plot(df['time'], p_plot, linewidth=1.5)

    plt.xlabel("Time (s)")
    plt.ylabel("Pressure (Pa)")
    plt.title(f"Pressure Time History at {fft_tap} (Injection {inj})")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"time_history_inj_{inj}.png", dpi=300)
    plt.close()

    # -----------------------------
    # FFT
    # -----------------------------

    p = df[fft_tap].values

    time = df['time'].values
    dt = time[1] - time[0]

    freq = rfftfreq(len(p), dt)
    spectrum = np.abs(rfft(p))

    plt.figure()
    plt.plot(freq, spectrum, linewidth=1.5)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.title(f"FFT Spectrum at {fft_tap} (Injection {inj})")

    plt.xlim(0,200)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"fft_inj_{inj}.png", dpi=300)
    plt.close()

# ----------------------------------------------------
# CHORDWISE TAP NUMBERS
# ----------------------------------------------------

tap_numbers = np.arange(1, len(mean_vals)+1)

# ----------------------------------------------------
# MEAN PRESSURE ALONG CHORD
# ----------------------------------------------------

plt.figure()

for inj in cases.keys():
    plt.plot(tap_numbers, mean_pressures[inj],
             marker='o', linewidth=2, label=f"Injection {inj}")

plt.xlabel("Tap Number (Chord Position)")
plt.ylabel("Mean Pressure (Pa)")
plt.title("Mean Pressure Distribution Along the Chord")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_pressure_chord.png", dpi=300)
plt.close()

# ----------------------------------------------------
# RMS ALONG CHORD
# ----------------------------------------------------

plt.figure()

for inj in cases.keys():
    plt.plot(tap_numbers, rms_pressures[inj],
             marker='o', linewidth=2, label=f"Injection {inj}")

plt.xlabel("Tap Number (Chord Position)")
plt.ylabel("RMS Pressure (Pa)")
plt.title("Pressure Fluctuation Distribution Along the Chord")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("rms_chord.png", dpi=300)
plt.close()

# ----------------------------------------------------
# RMS VS INJECTION RATE
# ----------------------------------------------------

inj_rates = []
rms_global = []

for inj in cases.keys():
    inj_rates.append(float(inj))
    rms_global.append(np.mean(rms_pressures[inj]))

plt.figure()
plt.plot(inj_rates, rms_global, marker='o', linewidth=2)

plt.xlabel("Injection Rate")
plt.ylabel("Mean RMS Pressure (Pa)")
plt.title("Pressure Fluctuation Amplitude vs Injection Rate")

plt.grid(True)
plt.tight_layout()
plt.savefig("rms_vs_injection.png", dpi=300)
plt.close()

# ----------------------------------------------------
# TABLE
# ----------------------------------------------------

table_df = pd.DataFrame(table_results)
table_df = table_df.sort_values("Injection Rate")

baseline_rms = table_df.loc[
    table_df["Injection Rate"] == 0.0,
    "RMS Pressure (Pa)"
].values[0]

table_df["Change vs Baseline (%)"] = (
    (table_df["RMS Pressure (Pa)"] - baseline_rms) / baseline_rms * 100
)

table_df["Mean Pressure (Pa)"] = table_df["Mean Pressure (Pa)"].round(0).astype(int)
table_df["RMS Pressure (Pa)"] = table_df["RMS Pressure (Pa)"].round(0).astype(int)
table_df["Change vs Baseline (%)"] = table_df["Change vs Baseline (%)"].round(4)

print("\nPressure Fluctuation Summary\n")

print(f"{'Injection Rate':<15}{'Mean Pressure (Pa)':<22}{'RMS Pressure (Pa)':<22}{'Change vs Baseline (%)'}")

for _, row in table_df.iterrows():
    print(f"{row['Injection Rate']:<15}{row['Mean Pressure (Pa)']:<22}{row['RMS Pressure (Pa)']:<22}{row['Change vs Baseline (%)']}")

# ----------------------------------------------------
# FINAL PLOT
# ----------------------------------------------------

inj_rates = table_df["Injection Rate"].values
rms_values = table_df["RMS Pressure (Pa)"].values

plt.figure(figsize=(8,5))
plt.plot(inj_rates, rms_values, marker='o', linewidth=2)

plt.xlabel("Injection Rate")
plt.ylabel("RMS Pressure (Pa)")
plt.title("RMS Pressure Fluctuation vs Injection Rate")

plt.grid(True)
plt.tight_layout()
plt.savefig("rms_vs_injection_rate.png", dpi=300)

plt.show()

# Data from table
inj_rates = table_df["Injection Rate"].values
rms_values = table_df["RMS Pressure (Pa)"].values

# ---------------------------------------
# SMOOTH CURVE (shape-preserving)
# ---------------------------------------
x_smooth = np.linspace(inj_rates.min(), inj_rates.max(), 500)
spline = PchipInterpolator(inj_rates, rms_values)
y_smooth = spline(x_smooth)

# ---------------------------------------
# PLOT
# ---------------------------------------
plt.figure(figsize=(10,6))

# Smooth curve
plt.plot(x_smooth, y_smooth, linewidth=2)

# Original points
plt.scatter(inj_rates, rms_values, zorder=3)

# Labels
plt.xlabel("Injection Rate")
plt.ylabel("RMS Pressure (Pa)")

# Grid (soft academic style)
plt.grid(True, linestyle='--', alpha=0.6)

# Limits (optional but matches your figure feel)
plt.xlim(0, 0.6)

plt.tight_layout()
plt.savefig("rms_vs_injection_smooth.png", dpi=300)

plt.show()