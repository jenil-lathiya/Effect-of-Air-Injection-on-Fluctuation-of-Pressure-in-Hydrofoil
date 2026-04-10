import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
colors = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#ff7f0e",  # orange
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#17becf"   # cyan
]

# ----------------------------------------------------
# GLOBAL STYLE (REPORT QUALITY)
# ----------------------------------------------------
plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    "figure.figsize": (12,6),
    "font.size": 13,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "legend.fontsize": 11,
    "lines.linewidth": 2,
    "lines.markersize": 6
})

# ----------------------------------------------------
# CASE PATHS
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
# TIME WINDOW (STEADY REGION)
# ----------------------------------------------------
t_start = 0.2
t_end   = 0.4

# ----------------------------------------------------
# STORAGE
# ----------------------------------------------------
mean_pressures = {}
rms_pressures  = {}

# ----------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------
for inj, filepath in cases.items():

    print(f"Processing injection: {inj}")

    df = pd.read_csv(filepath, delim_whitespace=True)

    # ---- Time filtering ONLY ----
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]

    taps = df.columns[1:]

    mean_vals = []
    rms_vals  = []

    for tap in taps:

        p = df[tap].values   # ✅ RAW SIGNAL (NO FILTER)

        # ---- MEAN ----
        p_mean = np.mean(p)

        # ---- RMS ----
        p_rms = np.sqrt(np.mean((p - p_mean)**2))

        mean_vals.append(p_mean)
        rms_vals.append(p_rms)

    mean_pressures[inj] = mean_vals
    rms_pressures[inj]  = rms_vals

# ----------------------------------------------------
# TAP NUMBERS
# ----------------------------------------------------
tap_numbers = np.arange(1, len(mean_vals)+1)

# ----------------------------------------------------
# PLOT STYLES
# ----------------------------------------------------
line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

plt.figure(figsize=(11,6))

for i, inj in enumerate(cases.keys()):

    lw = 2.5 if inj in ["0.0", "0.3", "0.6"] else 1.5
    alpha = 1.0 if inj in ["0.0", "0.3", "0.6"] else 0.6

    plt.plot(
        tap_numbers,
        mean_pressures[inj],
        color=colors[i],
        linewidth=lw,
        alpha=alpha,
        marker='o',
        label=f"{inj}"
    )

plt.xlabel("Tap Number (Chord Position)")
plt.ylabel("Mean Pressure (Pa)")
plt.title("Mean Pressure Distribution Along the Chord")

# Clean grid
plt.grid(True, linestyle=':', alpha=0.5)

# Minimal legend (clean!)
plt.legend(title="Injection Rate", ncol=2, frameon=False)

plt.tight_layout()
plt.savefig("mean_pressure_chord.png", dpi=300)

plt.figure(figsize=(11,6))

plt.figure(figsize=(11,6))

# Professional color palette (for non-baseline)
colors = [
    "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#17becf"
]

# Line styles for non-baseline
line_styles = ['--', '-.', ':', '--', '-.', ':']

# Markers
markers = ['s', '^', 'D', 'v', 'P', 'X']

for i, inj in enumerate(cases.keys()):

    if inj == "0.0":
        # ---- BASELINE ----
        plt.plot(
            tap_numbers,
            rms_pressures[inj],
            color='black',
            linestyle='-',
            marker='o',
            linewidth=3,
            label="0.0 (Baseline)"
        )
    else:
        idx = i - 1  # shift index for styling lists

        plt.plot(
            tap_numbers,
            rms_pressures[inj],
            color=colors[idx],
            linestyle=line_styles[idx],
            marker=markers[idx],
            linewidth=2,
            alpha=0.9,
            label=inj
        )

# Labels
plt.xlabel("Tap Number (Chord Position)")
plt.ylabel("RMS Pressure (Pa)")
plt.title("Pressure Fluctuation Distribution Along the Chord")

# Clean grid
plt.grid(True, linestyle='--', alpha=0.5)

# Legend
plt.legend(title="Injection Rate", ncol=2, frameon=False)

plt.tight_layout()
plt.savefig("rms_chord_final.png", dpi=300)

plt.show()