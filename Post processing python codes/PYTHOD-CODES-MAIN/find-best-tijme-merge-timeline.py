import numpy as np
import pandas as pd

# ==========================
# USER INPUT
# ==========================
file1_path = "last_folder/re-run/0.4/postProcessing/F_Hydrofoil/0/1.dat"
file2_path = "last_folder/re-run/0.4/postProcessing/F_Hydrofoil/0/2.dat"

time_tolerance = 1# tolerance for matching time values

# ==========================
# LOAD DATA (robustly)
# ==========================
def load_data(filepath):
    return pd.read_csv(
        filepath,
        sep=r"\s+",
        comment="#",
        header=None,
        engine="python"
    )

df1 = load_data(file1_path)
df2 = load_data(file2_path)

# Rename first two columns only
df1 = df1.rename(columns={0: "Time", 1: "total_x"})
df2 = df2.rename(columns={0: "Time", 1: "total_x"})

# ==========================
# FIND OVERLAP REGION
# ==========================
overlap_start = max(df1["Time"].min(), df2["Time"].min())
overlap_end   = min(df1["Time"].max(), df2["Time"].max())

df1_overlap = df1[(df1["Time"] >= overlap_start) & (df1["Time"] <= overlap_end)]
df2_overlap = df2[(df2["Time"] >= overlap_start) & (df2["Time"] <= overlap_end)]

# ==========================
# COMPUTE MIN DIFFERENCE
# ==========================
min_diff = np.inf
best_time = None

for t1, val1 in zip(df1_overlap["Time"], df1_overlap["total_x"]):

    diff_times = np.abs(df2_overlap["Time"].values - t1)
    idx = np.argmin(diff_times)

    if diff_times[idx] < time_tolerance:

        val2 = df2_overlap.iloc[idx]["total_x"]
        diff = abs(val1 - val2)

        if diff < min_diff:
            min_diff = diff
            best_time = t1

# ==========================
# RESULT
# ==========================
if best_time is not None:
    print("Best matching time step:", best_time)
    print("Minimum absolute difference (total_x):", min_diff)
else:
    print("No matching time steps found.")