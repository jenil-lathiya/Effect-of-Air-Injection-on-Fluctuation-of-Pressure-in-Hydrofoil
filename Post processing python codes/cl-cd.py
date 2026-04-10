import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator

# -----------------------------
# DATA
# -----------------------------
inj = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])

cl_rms = np.array([0.28139, 0.27994, 0.28193, 0.28253, 0.28103, 0.27913, 0.27811])
cd_rms = np.array([0.05043, 0.04911, 0.05052, 0.05046, 0.05031, 0.04732, 0.04620])
cm_rms = np.array([0.04445, 0.04473, 0.04411, 0.04357, 0.04360, 0.04352, 0.04360])

# -----------------------------
# SMOOTH FUNCTION
# -----------------------------
def smooth(x, y):
    x_new = np.linspace(x.min(), x.max(), 300)
    y_new = PchipInterpolator(x, y)(x_new)
    return x_new, y_new

# -----------------------------
# FIGURE
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15,5))

# -----------------------------
# CL
# -----------------------------
x_s, y_s = smooth(inj, cl_rms)
axes[0].plot(x_s, y_s, color='black', linewidth=2)
axes[0].scatter(inj, cl_rms, color='black', marker='o')
axes[0].set_title("(a) Lift Coefficient")
axes[0].set_xlabel("Injection Rate")
axes[0].set_ylabel(r"$C_L^{RMS}$")
axes[0].grid(True, linestyle='--', alpha=0.5)

# -----------------------------
# CD
# -----------------------------
x_s, y_s = smooth(inj, cd_rms)
axes[1].plot(x_s, y_s, color='black', linestyle='--', linewidth=2)
axes[1].scatter(inj, cd_rms, color='black', marker='s')
axes[1].set_title("(b) Drag Coefficient")
axes[1].set_xlabel("Injection Rate")
axes[1].set_ylabel(r"$C_D^{RMS}$")
axes[1].grid(True, linestyle='--', alpha=0.5)

# -----------------------------
# CM
# -----------------------------
x_s, y_s = smooth(inj, cm_rms)
axes[2].plot(x_s, y_s, color='black', linestyle='-.', linewidth=2)
axes[2].scatter(inj, cm_rms, color='black', marker='^')
axes[2].set_title("(c) Moment Coefficient")
axes[2].set_xlabel("Injection Rate")
axes[2].set_ylabel(r"$C_M^{RMS}$")
axes[2].grid(True, linestyle='--', alpha=0.5)

# -----------------------------
# FINAL
# -----------------------------
plt.tight_layout()
plt.savefig("coefficients_smooth.png", dpi=300)
plt.show()