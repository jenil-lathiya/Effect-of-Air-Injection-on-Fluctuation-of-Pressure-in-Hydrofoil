import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------
# CASE FILES
# ---------------------------------

cases = {
    "0.0": "last_folder/re-run/0/postProcessing/p_taps_all.txt",
    "0.1": "last_folder/re-run/0.1/postProcessing/p_taps_all.txt",
    "0.2": "last_folder/re-run/0.2/postProcessing/p_taps_all.txt",
    "0.3": "last_folder/re-run/0.3/postProcessing/p_taps_all.txt",
    "0.4": "last_folder/re-run/0.4/postProcessing/p_taps_all.txt",
    "0.5": "last_folder/re-run/0.5/postProcessing/p_taps_all.txt",
    "0.6": "last_folder/re-run/0.6/postProcessing/p_taps_all.txt",
    "0.7": "last_folder/re-run/0.7/postProcessing/p_taps_all.txt"
}

tap_name = "tap5"
t_start = 0.3
t_end = 0.394

# ---------------------------------
# STORAGE
# ---------------------------------

inj_rates = []
rms_values = []

# ---------------------------------
# PROCESS CASES
# ---------------------------------

for inj, filepath in cases.items():

    df = pd.read_csv(filepath, delim_whitespace=True)

    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]

    pressure = df[tap_name].values

    # NO FILTER APPLIED

    mean_p = np.mean(pressure)

    rms = np.sqrt(np.mean((pressure - mean_p)**2))

    inj_rates.append(float(inj))
    rms_values.append(rms)

# ---------------------------------
# SORT DATA
# ---------------------------------

inj_rates = np.array(inj_rates)
rms_values = np.array(rms_values)

sort_idx = np.argsort(inj_rates)

inj_rates = inj_rates[sort_idx]
rms_values = rms_values[sort_idx]

# ---------------------------------
# PLOT
# ---------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    inj_rates,
    rms_values,
    marker='o',
    linewidth=2
)

plt.xlabel("Injection Rate")
plt.ylabel("RMS Pressure Fluctuation (Pa)")
plt.title("Variation of Pressure Fluctuation Amplitude with Injection Rate")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "rms_vs_injection_rate.png",
    dpi=300
)

plt.show()