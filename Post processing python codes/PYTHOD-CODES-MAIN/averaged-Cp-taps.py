import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")

FILES = {
    "q = 0.03": ROOT / "q=0.03" / "postProcessing" / "p_taps_all.txt",
    "q = 0.05": ROOT / "q=0.05" / "postProcessing" / "p_taps_all.txt",
    "q = 0.1":  ROOT / "q=0.1"  / "postProcessing" / "p_taps_all.txt",
    "q = 0.125": ROOT / "0.125" / "postProcessing" / "p_taps_all.txt",
    "q = 0.2":  ROOT / "q=0.2"  / "postProcessing" / "p_taps_all.txt",
    "q = 0.5":  ROOT / "0.5"    / "postProcessing" / "p_taps_all.txt",
}

T_START = 0.20

# Flow properties (EDIT if required)
rho = 997.0        # kg/m^3
U_inf = 10.45       # m/s
p_ref = 62468.0         # reference pressure
# ==============================================

plt.figure(figsize=(7, 4.5))

for label, file_path in FILES.items():

    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        continue

    # Load data
    data = np.loadtxt(file_path, comments="#", skiprows=1)

    time = data[:, 0]
    p_taps = data[:, 2:]   # tap1 ... tapN

    # Select steady-state region
    mask = time >= T_START
    p_steady = p_taps[mask, :]

    # Time-averaged pressure at taps
    p_mean = np.mean(p_steady, axis=0)

    # Pressure coefficient
    Cp_mean = (p_mean - p_ref) / (0.5 * rho * U_inf**2)

    taps = np.arange(1, Cp_mean.size + 1)

    plt.plot(
        taps,
        Cp_mean,
        marker="o",
        linewidth=2,
        label=label
    )

plt.xlabel("Tap number")
plt.ylabel(r"Mean pressure coefficient $\overline{C_p}$")
plt.title("Time-Averaged Pressure Coefficient Distribution")
plt.grid(True)
plt.legend(title="Injection rate")
plt.tight_layout()
plt.show()
