import numpy as np

# ========== EDIT THIS ==========
FILE = "last_folder/q=0.05/postProcessing/p_taps_all.txt"

rho_inf = 997      # air density (kg/m^3)
U_inf   = 10.45       # reference velocity (m/s)
p_inf   = 62468        # reference pressure (Pa)

t_start = 0.15       # averaging window start
t_end   = 0.37       # averaging window end
# ===============================


# -------- load data --------
data = np.loadtxt(FILE, skiprows=1)

time = data[:, 0]
p_taps = data[:, 1:]   # columns: tap1 ... tap11
n_taps = p_taps.shape[1]

# -------- select averaging window --------
mask = (time >= t_start) & (time <= t_end)
t = time[mask]
p = p_taps[mask, :]

if len(t) < 2:
    raise ValueError("Not enough data points in averaging window")

# -------- compute Cp(t) --------
q_dyn = 0.5 * rho_inf * U_inf**2
Cp = (p - p_inf) / q_dyn

# -------- time-averaged Cp (tap-wise) --------
Cp_mean = np.zeros(n_taps)

for i in range(n_taps):
    Cp_mean[i] = np.trapz(Cp[:, i], t) / (t[-1] - t[0])

# -------- save results --------
with open("Cp_mean.txt", "w") as f:
    f.write("tap_index Cp_mean\n")
    for i, val in enumerate(Cp_mean, start=1):
        f.write(f"{i:2d} {val:.6e}\n")

print("DONE ✅")
print("Saved: Cp_mean.txt")

import matplotlib.pyplot as plt

# -------- select taps: tap2 to tap11 --------
tap_indices = np.arange(2, n_taps + 1)      # [2, 3, ..., 11]
Cp_plot = Cp_mean[1:]                        # drop tap1

# -------- plot --------
plt.figure(figsize=(7, 4.5))

plt.plot(tap_indices, Cp_plot,
         marker='o', linewidth=2, markersize=6)

plt.xlabel("Tap number")
plt.ylabel(r"Time-averaged $C_p$")
plt.title(r"Averaged $C_p$ distribution ($t=%.2f$ to $%.2f$)" % (t_start, t_end))

plt.grid(True, alpha=0.4)
plt.xticks(tap_indices)

plt.tight_layout()
plt.show()

