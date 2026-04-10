import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# USER INPUT
# =====================================================

base_folder = Path("last_folder")

# Flow rate : list of 4 times (phase-consistent)
cases = {
    0.03: [0.2076, 0.2545, 0.3014, 0.3484],
    0.05: [0.2088, 0.2558947, 0.30305, 0.347705882],
    0.10: [0.20564, 0.253045455, 0.300454545, 0.347705882],
    0.20: [0.2065, 0.254, 0.3009, 0.3477],
}

phase_labels = [
    "Phase 1",
    "Phase 2",
    "Phase 3",
    "Phase 4",
]

# =====================================================
# DATA EXTRACTION
# =====================================================

CL_phases = {i: [] for i in range(4)}
q_values = []

for q, times in cases.items():

    file_path = (
        base_folder
        / f"q={q}"
        / "postProcessing"
        / "forceCoeffs"
        / "0"
        / "coefficient.dat"
    )

    if not file_path.exists():
        raise FileNotFoundError(file_path)

    data = np.loadtxt(file_path, comments="#")

    time = data[:, 0]
    CL   = data[:, 4]   # <-- CONFIRMED lift column

    q_values.append(q)

    for i, t_target in enumerate(times):
        idx = np.argmin(np.abs(time - t_target))
        CL_phases[i].append(CL[idx])

        print(
            f"q={q:.2f}, phase {i+1}, "
            f"t_used={time[idx]:.4f}, CL={CL[idx]:.5f}"
        )

# =====================================================
# PLOTTING
# =====================================================

plt.figure(figsize=(7, 5))

for i in range(4):
    plt.plot(
        q_values,
        CL_phases[i],
        "o-",
        linewidth=2,
        label=phase_labels[i]
    )

plt.xlabel("Flow rate (q)")
plt.ylabel("$C_L$")
plt.title("$C_L$ vs Flow Rate at Different Shedding Phases")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
