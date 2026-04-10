import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")

FILES = {
    "q = 0": ROOT / "re-run"/"0" / "postProcessing" /"p_taps_all.txt",
    
}

T_START = 0.16
# ==============================================

plt.figure(figsize=(7, 4.5))

for label, file_path in FILES.items():

    if not file_path.exists():
        print(f"[WARNING] File not found: {file_path}")
        continue

    # ---- FIX: skip header row ----
    data = np.loadtxt(file_path, comments="#", skiprows=1)

    time = data[:, 0]
    p_taps = data[:, 1:]   # tap1 ... tap11

    # Select steady region
    mask = time >= T_START
    p_steady = p_taps[mask, :]

    # Compute prime2Mean
    p_mean = np.mean(p_steady, axis=0)
    prime2Mean = np.mean((p_steady - p_mean)**2, axis=0)

    taps = np.arange(1, prime2Mean.size + 1)

    plt.plot(
        taps,
        prime2Mean,
        marker="o",
        linewidth=2,
        label=label
    )

plt.xlabel("Tap number")
plt.ylabel(r"$\overline{p'^2}$  [Pa$^2$]")
plt.title("Comparison of Pressure Fluctuation Variance at Taps")
plt.grid(True)
plt.legend(title="Flow rate")
plt.tight_layout()
plt.show()
