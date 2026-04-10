
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# USER INPUT
# =========================================================
csv_file = "last_folder/post-processing/q=0.1/Cp_rms.csv"   # <-- change this
LE = np.array([0.0, 0.0])        # Leading edge (x, y)
TE = np.array([0.0693129, -0.00978366])  # Trailing edge (x, y)

# =========================================================
# LOAD CSV SAFELY
# =========================================================
# Expected CSV columns:
# col 0 -> Cp_rms
# col 1 -> x
# col 2 -> y
# col 3 -> z (ignored)

data = pd.read_csv(csv_file, header=None)
data = data.apply(pd.to_numeric, errors='coerce')
data = data.dropna()

Cp_rms = data.iloc[:, 0].values
x = data.iloc[:, 1].values
y = data.iloc[:, 2].values

# =========================================================
# ENSURE RMS IS POSITIVE (IMPORTANT)
# =========================================================
Cp_rms = np.abs(Cp_rms)

# =========================================================
# CHORD GEOMETRY
# =========================================================
chord_vec = TE - LE
chord_len = np.linalg.norm(chord_vec)
chord_unit = chord_vec / chord_len

# =========================================================
# PROJECT POINTS TO x/c
# =========================================================
points = np.vstack((x, y)).T
proj = np.dot(points - LE, chord_unit)
x_over_c = proj / chord_len

# =========================================================
# SPLIT UPPER / LOWER SURFACES
# =========================================================
# Chord-normal vector
normal = np.array([-chord_unit[1], chord_unit[0]])

# Signed distance from chord
signed_dist = np.dot(points - LE, normal)

upper = signed_dist > 0
lower = signed_dist < 0

# =========================================================
# SORT EACH SURFACE ALONG x/c
# =========================================================
idx_u = np.argsort(x_over_c[upper])
idx_l = np.argsort(x_over_c[lower])

x_u = x_over_c[upper][idx_u]
Cp_u = Cp_rms[upper][idx_u]

x_l = x_over_c[lower][idx_l]
Cp_l = Cp_rms[lower][idx_l]

# =========================================================
# PLOT
# =========================================================
plt.figure(figsize=(7, 4.5))

plt.plot(x_u, Cp_u, 'r-', linewidth=2.2, label='Upper surface')
plt.plot(x_l, Cp_l, 'b-', linewidth=2.2, label='Lower surface')

plt.xlabel(r"$x/c$")
plt.ylabel(r"$-C_{p,\mathrm{rms}}$")
plt.ylim(0.2, 0.6)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()