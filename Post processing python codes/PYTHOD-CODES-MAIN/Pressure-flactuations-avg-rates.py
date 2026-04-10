import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")

FILES = {
    0.03: ROOT / "q=0.03" / "postProcessing" / "p_taps_all.txt",
    0.05: ROOT / "q=0.05"  / "postProcessing" / "p_taps_all.txt",
    0.1: ROOT / "q=0.1" / "postProcessing" / "p_taps_all.txt",
    0.2: ROOT / "q=0.2"  / "postProcessing" / "p_taps_all.txt",
    
}

T_START = 0.2
# ==============================================

q_vals = []
p_var_overall = []

for q, file_path in FILES.items():

    if not file_path.exists():
        print(f"[WARNING] Missing file for q={q}")
        continue

    # Load data (skip header)
    data = np.loadtxt(file_path, skiprows=1)

    time = data[:, 0]
    p_taps = data[:, 1:]   # tap1 ... tap11

    # Steady region
    mask = time >= T_START
    p_steady = p_taps[mask, :]

    # Mean pressure per tap
    p_mean = np.mean(p_steady, axis=0)

    # Pressure fluctuation variance per tap (prime2Mean)
    prime2Mean_taps = np.mean((p_steady - p_mean)**2, axis=0)

    # OVERALL pressure fluctuation variance
    prime2Mean_overall = np.mean(prime2Mean_taps)

    q_vals.append(q)
    p_var_overall.append(prime2Mean_overall)

# Sort by flow rate
q_vals = np.array(q_vals)
p_var_overall = np.array(p_var_overall)

idx = np.argsort(q_vals)
q_vals = q_vals[idx]
p_var_overall = p_var_overall[idx]

# =================================================
# Plot: overall pressure fluctuation variance
# =============================================
plt.figure(figsize=(6, 4))
plt.plot(q_vals, p_var_overall, marker='o', linewidth=2)

plt.xlabel("Air injection rate (q)")
plt.ylabel(r"Overall pressure fluctuation variance $\overline{p'^2}$ [Pa$^2$]")
plt.title("Overall Pressure Fluctuation Variance vs Injection Rate")
plt.grid(True)
plt.tight_layout()
plt.show()