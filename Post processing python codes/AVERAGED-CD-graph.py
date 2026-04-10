import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# ====== EDIT ONLY THIS ======
case_dir = Path(r"q=0.1")

t0 = 0.14                 # for threshold analysis (UNCHANGED)

# >>> NEW: averaging window <<<
t_start = 0.11            # averaging start
t_end   = 0.36           # averaging end

Cd_high = 0.2            # adjust as needed
Cd_low  = 0.0
# =============================

coeff_file = Path("last_folder") / case_dir / "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"
if not coeff_file.exists():
    raise FileNotFoundError(f"File not found: {coeff_file}")

print("Reading:", coeff_file)

data = np.loadtxt(coeff_file, comments="#")

# Columns based on your header:
# Time Cd Cs(r) Cd(f) Cl Cl(f) Cl(r) CmPitch CmRoll CmYaw Cs Cs(f)
time = data[:, 0]
Cd   = data[:, 1]   # TOTAL Cd

# =====================================================
# --- AVERAGING OVER USER-DEFINED TIME RANGE ---
# =====================================================
avg_mask = (time >= t_start) & (time <= t_end)
tt_avg = time[avg_mask]
yy_avg = Cd[avg_mask]

if len(tt_avg) < 2:
    raise ValueError("Not enough points in averaging window")

dt = np.diff(tt_avg)
cum_int = np.concatenate([[0.0],
                           np.cumsum(0.5 * (yy_avg[1:] + yy_avg[:-1]) * dt)])
Cd_runavg = cum_int / (tt_avg - t_start + 1e-30)

Cd_mean = np.trapz(yy_avg, tt_avg) / (t_end - t_start)
print(f"Time-averaged Cd over [{t_start}, {t_end}] = {Cd_mean:.6f}")

# =====================================================
# --- THRESHOLD ANALYSIS (OLD LOGIC, UNCHANGED) ---
# =====================================================
mask_rng = time >= t0
time_rng = time[mask_rng]
Cd_rng   = Cd[mask_rng]

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

above_mask = Cd_rng > Cd_high
below_mask = Cd_rng < Cd_low

above_ranges = find_time_ranges(time_rng, above_mask)
below_ranges = find_time_ranges(time_rng, below_mask)

# =====================================================
# ------------------ PLOTTING -------------------------
# =====================================================
plt.figure(figsize=(10, 5))

# Instantaneous signal (FULL)
plt.plot(time, Cd, alpha=0.35, label="Instantaneous $C_D(t)$")

# Running average (ONLY in averaging window)
plt.plot(tt_avg, Cd_runavg, linewidth=2.2,
         label=f"Running avg [$t$={t_start} → {t_end}]")

# Mean line
plt.axhline(Cd_mean, linewidth=2.0,
            label=f"Mean $C_D$ = {Cd_mean:.4f}")

# Threshold lines
plt.axhline(Cd_high, linestyle="--", linewidth=1, alpha=0.8,
            label=f"$C_D$ = {Cd_high}")
plt.axhline(Cd_low, linestyle="--", linewidth=1, alpha=0.8,
            label=f"$C_D$ = {Cd_low}")

# Shade averaging window
plt.axvspan(t_start, t_end, alpha=0.15, label="Averaging window")

# Shade threshold regions (after t0)
for a, b in above_ranges:
    plt.axvspan(a, b, alpha=0.20)
for a, b in below_ranges:
    plt.axvspan(a, b, alpha=0.20)

plt.xlabel("Time [s]")
plt.ylabel("$C_D$")
plt.grid(True, alpha=0.4)
plt.legend()

ax = plt.gca()
ax.xaxis.set_major_locator(MultipleLocator(0.01))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.tight_layout()
plt.show()
