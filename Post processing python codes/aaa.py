import matplotlib.pyplot as plt
import numpy as np

# Data from your table
injection_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

rms_pressure = [
    41165,
    44527,
    39549,
    39389,
    41152,
    53287,
    53822
]

# Plot
plt.figure(figsize=(10,6))

plt.plot(
    injection_rate,
    rms_pressure,
    marker='o',
    linewidth=2
)

plt.xlabel("Injection Rate")
plt.ylabel("Mean RMS Pressure (Pa)")
plt.title("Pressure Fluctuation Amplitude vs Injection Rate")

plt.grid(True)

plt.tight_layout()

plt.savefig("rms_vs_injection_rate.png", dpi=300)

plt.show()