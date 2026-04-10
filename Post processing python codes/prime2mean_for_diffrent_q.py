import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")
CASES = [ "q=0.5"]     # ["q=0.05", "q=0.1"]
T_START = 0.15                 # steady start time
# ==============================================


def load_p_taps(case):
    """
    Load combined tap file for a case.
    Expected file:
    last_folder/<case>/postProcessing/p_taps_all.txt
    """
    file_path = ROOT / case / "postProcessing" / "p_taps_all.txt"
    data = np.loadtxt(file_path, skiprows=1)

    time = data[:, 0]
    p_all = data[:, 1:]   # tap1 ... tap11
    return time, p_all


def compute_prime2Mean(time, p_all):
    """Compute prime2Mean for each tap"""
    mask = time >= T_START
    p_all = p_all[mask, :]

    prime2 = []
    for i in range(p_all.shape[1]):
        p = p_all[:, i]
        p_mean = np.mean(p)
        prime2.append(np.mean((p - p_mean) ** 2))

    return np.array(prime2)


# ================== MAIN ==================
plt.figure(figsize=(10, 10))

for case in CASES:
    time, p_all = load_p_taps(case)
    prime2_vals = compute_prime2Mean(time, p_all)

    taps = np.arange(1, prime2_vals.size + 1)

    plt.plot(
        taps,
        prime2_vals,
        marker="o",
        linewidth=2,
        label=case
    )
plt.xticks(np.arange(1, 12)) 
plt.xlabel("Tap number")
plt.ylabel(r"$\overline{p'^2}$  [Pa$^2$]")
plt.title("Pressure fluctuation variance (prime2Mean) at taps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
