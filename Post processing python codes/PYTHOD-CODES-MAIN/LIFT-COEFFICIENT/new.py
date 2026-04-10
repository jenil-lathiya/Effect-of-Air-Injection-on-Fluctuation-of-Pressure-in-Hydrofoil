import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator


# Data
Q = np.array([0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60])
CL = np.array([0.89707, 0.89322, 0.89663, 0.89762, 0.89684, 0.89580, 0.89346])
CD = np.array([0.10739, 0.10671, 0.10772, 0.10795, 0.10775, 0.10602, 0.10528])

# Compute CL/CD
efficiency = CL / CD

# Shape-preserving interpolation
interp = PchipInterpolator(Q, efficiency)

# Smooth curve
Q_smooth = np.linspace(Q.min(), Q.max(), 200)
eff_smooth = interp(Q_smooth)

# Plot
plt.figure()
plt.plot(Q, efficiency, 'o', label='Aerodynamic efficiency Cl/Cd')
plt.plot(Q_smooth, eff_smooth)

plt.xlabel("Injection Rate (Q)")
plt.ylabel("Cl / Cd")
plt.title("Aerodynamic Efficiency vs Injection Rate")
plt.grid(True)
plt.legend()

plt.show()