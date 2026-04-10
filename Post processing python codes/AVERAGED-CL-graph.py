import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ====== EDIT ONLY THIS ======
case_dir = Path(r"0")

t0 = 0.15                 # for threshold analysis (UNCHANGED)

# >>> NEW: averaging window <<<
t_start = 0.200           # averaging start
t_end   = 0.370        # averaging end

Cl_high = 3.0
Cl_low  = -2.5
# =============================

coeff_file = Path("last_folder") / case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
if not coeff_file.exists():
    raise FileNotFoundError(f"File not found: {coeff_file}")

print("Reading:", coeff_file)

data = np.loadtxt(coeff_file, comments="#")

time = data[:, 0]
Cl   = data[:, 4]   # TOTAL Cl

# =====================================================
# --- AVERAGING OVER USER-DEFINED TIME RANGE ---
# =====================================================
avg_mask = (time >= t_start) & (time <= t_end)
tt_avg = time[avg_mask]
yy_avg = Cl[avg_mask]

if len(tt_avg) < 2:
    raise ValueError("Not enough points in averaging window")

dt = np.diff(tt_avg)
cum_int = np.concatenate([[0.0],
                           np.cumsum(0.5 * (yy_avg[1:] + yy_avg[:-1]) * dt)])
Cl_runavg = cum_int / (tt_avg - t_start + 1e-30)

Cl_mean = np.trapz(yy_avg, tt_avg) / (t_end - t_start)
print(f"Time-averaged Cl over [{t_start}, {t_end}] = {Cl_mean:.6f}")

# =====================================================
# --- THRESHOLD ANALYSIS (OLD LOGIC, UNCHANGED) ---
# =====================================================
mask_rng = time >= t0
time_rng = time[mask_rng]
Cl_rng   = Cl[mask_rng]

def find_time_ranges(time_arr, condition_mask):
    idx = np.where(condition_mask)[0]
    if len(idx) == 0:
        return []

    ranges = []
    start = idx[0]
    prev = idx[0]

    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            ranges.append((time_arr[start], time_arr[prev]))
            start = prev = i

    ranges.append((time_arr[start], time_arr[prev]))
    return ranges

above_mask = Cl_rng > Cl_high
below_mask = Cl_rng < Cl_low

above_ranges = find_time_ranges(time_rng, above_mask)
below_ranges = find_time_ranges(time_rng, below_mask)

# =====================================================
# ------------------ PLOTTING -------------------------
# =====================================================
plt.figure(figsize=(10, 5))

# Instantaneous signal (FULL)
plt.plot(time, Cl, alpha=0.35, label="Instantaneous $C_L(t)$")

# Running average (ONLY in averaging window)
plt.plot(tt_avg, Cl_runavg, linewidth=2.2,
         label=f"Running avg [$t$={t_start} → {t_end}]")

# Mean line
plt.axhline(Cl_mean, linewidth=2.0,
            label=f"Mean $C_L$ = {Cl_mean:.4f}")

# Threshold lines
plt.axhline(Cl_high, linestyle="--", linewidth=1, alpha=0.8,
            label=f"$C_L$ = {Cl_high}")
plt.axhline(Cl_low, linestyle="--", linewidth=1, alpha=0.8,
            label=f"$C_L$ = {Cl_low}")

# Shade averaging window
plt.axvspan(t_start, t_end, alpha=0.15, label="Averaging window")

# Shade threshold regions (after t0)
for a, b in above_ranges:
    plt.axvspan(a, b, alpha=0.20)
for a, b in below_ranges:
    plt.axvspan(a, b, alpha=0.20)

plt.xlabel("Time [s]")
plt.ylabel("$C_L$")
plt.grid(True, alpha=0.4)
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()
