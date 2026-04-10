import numpy as np
import matplotlib.pyplot as plt

# ===================== USER INPUTS =====================
csv_file = r"C_l.csv"   # your file name (even if it contains Cp, keep name as is)

# Airfoil LE and TE (tilted)
LE = np.array([0.0, 0.0])                 # (x_LE, y_LE)
TE = np.array([0.0693129, -0.00978366])   # (x_TE, y_TE)

# If your first column is Cp, set this True to invert y-axis in plot
is_cp = True

# Binning controls (more bins = smoother curve but needs enough points)
n_bins = 80

# Optional: ignore very tiny region near LE where points cluster
xc_min_plot = 0.0   # e.g. 0.01 to drop extreme LE cluster
xc_max_plot = 1.0
# ======================================================

# ---------- Load CSV (no pandas) ----------
# Assumes 1 header row, 4 numeric columns
data = np.genfromtxt(csv_file, delimiter=",", skip_header=1)

val = data[:, 0]   # Cp or Cl (your first column)
x   = data[:, 1]
y   = data[:, 2]
z   = data[:, 3]   # not used, but kept for completeness

# ---------- Compute chordwise coordinate s/c (tilt-corrected) ----------
cvec = TE - LE
c2 = float(np.dot(cvec, cvec))

xc = ((x - LE[0]) * cvec[0] + (y - LE[1]) * cvec[1]) / c2   # normalized projection

# ---------- Compute signed normal coordinate to split upper/lower ----------
# normal to chord (2D): n = (-cy, cx)
n = np.array([-cvec[1], cvec[0]])
n = n / np.linalg.norm(n)

# signed distance from chord line (positive = one side, negative = other)
r = np.column_stack((x - LE[0], y - LE[1]))
eta = r @ n

# midline threshold (robust): use median
eta0 = np.median(eta)

upper_mask = eta >= eta0
lower_mask = eta < eta0

# ---------- Helper: bin-average along x/c ----------
def bin_average(xc_arr, val_arr, nbins=80, xmin=0.0, xmax=1.0):
    # keep range
    m = (xc_arr >= xmin) & (xc_arr <= xmax) & np.isfinite(xc_arr) & np.isfinite(val_arr)
    xc_arr = xc_arr[m]
    val_arr = val_arr[m]

    bins = np.linspace(xmin, xmax, nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    means = np.full(nbins, np.nan)

    for i in range(nbins):
        bmask = (xc_arr >= bins[i]) & (xc_arr < bins[i+1])
        if np.any(bmask):
            means[i] = np.mean(val_arr[bmask])

    # remove empty bins
    ok = np.isfinite(means)
    return centers[ok], means[ok]

# ---------- Build clean curves ----------
xc_u, val_u = bin_average(xc[upper_mask], val[upper_mask], nbins=n_bins,
                          xmin=xc_min_plot, xmax=xc_max_plot)
xc_l, val_l = bin_average(xc[lower_mask], val[lower_mask], nbins=n_bins,
                          xmin=xc_min_plot, xmax=xc_max_plot)

# ---------- Plot ----------
plt.figure(figsize=(7.5, 5))
plt.plot(xc_u, val_u, "-", linewidth=2, label="Upper surface (binned mean)")
plt.plot(xc_l, val_l, "-", linewidth=2, label="Lower surface (binned mean)")

# Optional: show raw points lightly (helps justify)
plt.plot(xc[upper_mask], val[upper_mask], "o", markersize=2, alpha=0.15)
plt.plot(xc[lower_mask], val[lower_mask], "o", markersize=2, alpha=0.15)

plt.xlabel("x/c")
plt.ylabel("Cp" if is_cp else "Cl")
plt.grid(True)
plt.legend()
plt.tight_layout()

if is_cp:
    plt.gca().invert_yaxis()  # standard Cp convention

plt.show()
