import numpy as np
import glob, os, re, csv
import matplotlib.pyplot as plt

# ---------------- USER SETTINGS ----------------
base_dir = "q=0.1"   # change to q=0.2, q=0.3...
chord_c = 0.0567                   # [m] (you provided)
rho = 1000.0                       # water
Uinf = 10.45                       # [m/s]
p_inf = 0.0                        # reference pressure (0 for gauge pressure)
t_start = 0.0                      # set later (e.g. 0.25 to avoid transient)
t_end = None                       # e.g. 0.40, or None = all data
# ------------------------------------------------

qInf = 0.5 * rho * Uinf**2  # 54601.25 Pa

# Header coordinate pattern: # Probe 0 (x y z)
pat = re.compile(r"\(\s*([-\deE.+]+)\s+([-\deE.+]+)\s+([-\deE.+]+)\s*\)")

tap_files = sorted(glob.glob(os.path.join(base_dir, "tap*/0/p")))
if not tap_files:
    raise RuntimeError(f"No tap files found. Check path: {base_dir}/postProcessing/p_taps_all.txt")

tap_data = []  # (tap_name, x, y, z, Cp_mean)

for pf in tap_files:
    tap_name = os.path.basename(os.path.dirname(os.path.dirname(pf)))  # tap1, tap2...

    # --- read header to get (x,y,z) ---
    x = y = z = None
    with open(pf, "r") as f:
        for line in f:
            if "Probe" in line and "(" in line:
                m = pat.search(line)
                if m:
                    x, y, z = map(float, m.groups())
                break
    if x is None:
        raise RuntimeError(f"Could not read probe coordinates from header in: {pf}")

    # --- read time, p ---
    time = []
    p = []
    with open(pf, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            time.append(float(parts[0]))
            p.append(float(parts[1]))

    time = np.array(time)
    p = np.array(p)

    # --- apply time window ---
    mask = time >= t_start
    if t_end is not None:
        mask &= time <= t_end
    if mask.sum() < 2:
        raise RuntimeError(f"Not enough samples in window for {tap_name}. Adjust t_start/t_end.")

    p_w = p[mask]
    Cp_mean = np.mean((p_w - p_inf) / qInf)

    tap_data.append((tap_name, x, y, z, Cp_mean))

# Choose leading edge x as minimum tap x (good for chordwise taps)
x_le = min(row[1] for row in tap_data)

# Sort taps by x
tap_data_sorted = sorted(tap_data, key=lambda r: r[1])

# Write CSV
out_csv = "Cp_vs_xc_q0p1.csv"
with open(out_csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["tap", "x", "x_over_c", "y", "z", "Cp_mean"])
    for tap_name, x, y, z, Cp_mean in tap_data_sorted:
        x_over_c = (x - x_le) / chord_c
        w.writerow([tap_name, x, x_over_c, y, z, Cp_mean])

print(f"Saved CSV: {out_csv}")
print(f"Used x_LE = {x_le:.8g} m, chord c = {chord_c} m")
print(f"Dynamic pressure qInf = {qInf:.2f} Pa")

# Prepare plot arrays
xc = np.array([(r[1] - x_le) / chord_c for r in tap_data_sorted])
cp = np.array([r[4] for r in tap_data_sorted])
labels = [r[0] for r in tap_data_sorted]

# Plot
plt.figure()
plt.plot(xc, cp, marker="o")
plt.xlabel("x/c")
plt.ylabel("Mean Cp")
plt.title(f"Cp vs x/c from taps ({os.path.basename(base_dir)})")
plt.grid(True)

# Optional: annotate points with tap names
for xi, yi, lab in zip(xc, cp, labels):
    plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(5, 5), fontsize=8)

out_png = "Cp_vs_xc_q0p1.png"
plt.tight_layout()
plt.savefig(out_png, dpi=300)
print(f"Saved plot: {out_png}")
