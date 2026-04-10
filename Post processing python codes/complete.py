import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================
# USER SETTINGS
# ============================

cases = {
    "q = 0.03": "last_folder/q=0.03/postProcessing/forceCoeffs/0/coefficient.dat",
    "q = 0.05": "last_folder/q=0.05/postProcessing/forceCoeffs/0/coefficient.dat",
    "q = 0.01": "last_folder/q=0.1/postProcessing/forceCoeffs/0/coefficient.dat",
    "q = 0.125": "last_folder/q=0.125/postProcessing/forceCoeffs/0/coefficient.dat",
    "q = 0.20": "last_folder/q=0.2/postProcessing/forceCoeffs/0/coefficient.dat",
    "q = 0.50": "last_folder/q=0.5/postProcessing/forceCoeffs/0/coefficient.dat",
}

# steady-state time window (CHANGE if needed)
t_start = 0.16
t_end   = 0.35

# column names (MATCHES YOUR FILE)
cols = [
    "Time","Cd","Cd_f","Cd_r","Cl","Cl_f","Cl_r",
    "CmPitch","CmRoll","CmYaw","Cs","Cs_f","Cs_r"
]

# ============================
# STORAGE
# ============================

results = []

plt.figure(figsize=(7,4))  # thrust history

# ============================
# LOOP OVER ALL q
# ============================

for label, file in cases.items():

    df = pd.read_csv(
        file,
        delim_whitespace=True,
        comment="#",
        names=cols
    )

    # remove transient
    df = df[(df.Time > t_start) & (df.Time < t_end)]

    # ---- TIME HISTORY: THRUST ----
    plt.plot(df.Time, df.Cd, label=label)

    # ---- RMS CALCULATION ----
    Cd_mean = df.Cd.mean()
    Cm_mean = df.CmPitch.mean()

    Cd_rms = np.sqrt(((df.Cd - Cd_mean)**2).mean())
    Cm_rms = np.sqrt(((df.CmPitch - Cm_mean)**2).mean())

    results.append((label, Cd_rms, Cm_rms))

# ============================
# THRUST TIME HISTORY PLOT
# ============================

plt.xlabel("Time")
plt.ylabel("Thrust coefficient $C_T$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("thrust_time_history.png", dpi=300)
plt.show()

# ============================
# MOMENT TIME HISTORY
# ============================

plt.figure(figsize=(7,4))

for label, file in cases.items():

    df = pd.read_csv(
        file,
        delim_whitespace=True,
        comment="#",
        names=cols
    )

    df = df[(df.Time > t_start) & (df.Time < t_end)]
    plt.plot(df.Time, df.CmPitch, label=label)

plt.xlabel("Time")
plt.ylabel("Pitching moment coefficient $C_M$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("moment_time_history.png", dpi=300)
plt.show()

# ============================
# RMS vs q PLOT
# ============================

q_labels = [r[0] for r in results]
Cd_rms_vals = [r[1] for r in results]
Cm_rms_vals = [r[2] for r in results]

plt.figure(figsize=(7,4))
plt.plot(q_labels, Cd_rms_vals, "-o", label="Thrust RMS $C_T$")
plt.plot(q_labels, Cm_rms_vals, "-s", label="Moment RMS $C_M$")

plt.xlabel("Injection rate q")
plt.ylabel("RMS value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("force_rms_vs_q.png", dpi=300)
plt.show()

# ============================
# PRINT REPORT TABLE
# ============================

print("\n===== FORCE FLUCTUATION SUMMARY =====")
print("Injection rate\tCd_rms\t\tCmPitch_rms")
for r in results:
    print(f"{r[0]}\t{r[1]:.4e}\t{r[2]:.4e}")
