import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------
# MAD FILTER FUNCTION
# ------------------------------------------------
def mad_filter(signal, threshold=5):

    median = np.median(signal)
    deviation = np.abs(signal - median)
    mad = np.median(deviation)

    if mad == 0:
        return signal

    modified_z = 0.6745 * deviation / mad

    filtered = signal.copy()
    filtered[modified_z > threshold] = median

    return filtered


# ------------------------------------------------
# PLOT STYLE
# ------------------------------------------------
plt.rcParams.update({
    "figure.figsize": (12,6),
    "font.size": 14,
    "axes.labelsize": 15,
    "axes.titlesize": 17,
    "legend.fontsize": 12
})

# ------------------------------------------------
# YOUR CASE PATHS
# ------------------------------------------------
cases = {
    "0.0": "last_folder/re-run/0/postProcessing/p_taps_all.txt",
    "0.1": "last_folder/re-run/0.1/postProcessing/p_taps_all.txt",
    "0.2": "last_folder/re-run/0.2/postProcessing/p_taps_all.txt",
    "0.3": "last_folder/re-run/0.3/postProcessing/p_taps_all.txt",
    "0.4": "last_folder/re-run/0.4/postProcessing/p_taps_all.txt",
    "0.5": "last_folder/re-run/0.5/postProcessing/p_taps_all.txt",
    "0.6": "last_folder/re-run/0.6/postProcessing/p_taps_all.txt"
}

# ------------------------------------------------
# SETTINGS
# ------------------------------------------------
tap_name = "tap4"
t_start = 0.3
t_end = 0.40

# ------------------------------------------------
# CREATE PLOT
# ------------------------------------------------
fig, ax = plt.subplots()

for inj, filepath in cases.items():

    print("Processing injection rate:", inj)

    df = pd.read_csv(filepath, delim_whitespace=True)

    # steady-state window
    df = df[(df["time"] >= t_start) & (df["time"] <= t_end)]

    time = df["time"].values
    pressure = df[tap_name].values

    # apply MAD filter
    pressure_filtered = mad_filter(pressure)

    ax.plot(
        time,
        pressure_filtered,
        linewidth=1.5,
        label=f"Injection {inj}"
    )

# ------------------------------------------------
# FORMAT
# ------------------------------------------------
ax.set_xlabel("Time (s)")
ax.set_ylabel("Pressure (Pa)")
ax.set_title(f"Pressure Time History at {tap_name} (MAD Filtered)")

ax.legend(ncol=2)
ax.grid(True)

plt.tight_layout()

plt.savefig(
    "pressure_time_history_MAD_filtered.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Plot saved successfully.")