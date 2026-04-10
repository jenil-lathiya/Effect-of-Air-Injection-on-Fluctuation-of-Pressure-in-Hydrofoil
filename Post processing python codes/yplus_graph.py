import pandas as pd
import matplotlib.pyplot as plt

# =====================================
# 1. File path
# =====================================
file_path = "last_folder/q=0.1/postProcessing/yPlus/0/yPlus.dat"

# =====================================
# 2. Read y+ file
# =====================================
df = pd.read_csv(
    file_path,
    delim_whitespace=True,
    comment="#",
    names=["Time", "Patch", "yPlus_min", "yPlus_max", "yPlus_avg"]
)

# =====================================
# 3. Select hydrofoil patch
# =====================================
airfoil = df[df["Patch"] == "CLARK_Y_AIRFOIL"]

# =====================================
# 4. Plot y+ vs time (avg + min/max dotted)
# =====================================
plt.figure(figsize=(7, 4))

# Average y+
plt.plot(
    airfoil["Time"],
    airfoil["yPlus_avg"],
    linewidth=2,
    label="Average y+"
)

# Max y+ (dotted)
plt.plot(
    airfoil["Time"],
    airfoil["yPlus_max"],
    linestyle="--",
    linewidth=1.5,
    label="Max y+"
)

# Min y+ (dotted)
plt.plot(
    airfoil["Time"],
    airfoil["yPlus_min"],
    linestyle="--",
    linewidth=1.5,
    label="Min y+"
)

plt.xlabel("Time [s]")
plt.ylabel("y+")
plt.title("y+ statistics on hydrofoil surface (q = 0.1)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =====================================
# 5. Print report-ready values
# =====================================
print("===== y+ SUMMARY (CLARK_Y_AIRFOIL) =====")
print(f"Time-averaged y+ : {airfoil['yPlus_avg'].mean():.2f}")
print(f"Maximum y+       : {airfoil['yPlus_max'].max():.2f}")
print(f"Minimum y+       : {airfoil['yPlus_min'].min():.2f}")
