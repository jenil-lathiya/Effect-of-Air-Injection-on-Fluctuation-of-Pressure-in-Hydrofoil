import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import cumulative_trapezoid

# =====================================================
# SETTINGS
# =====================================================
cases_to_plot = ["0", "0.2", "0.5"]   # choose 3 injection rates
t_start = 0.23                       # start of steady region
t_end   = 0.40

ROOT = Path("last_folder")

# =====================================================
# PLOT STYLE (Clean, Professional)
# =====================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
})

fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)

ymin, ymax = 0.3, 1.5   # unified y-axis limits

for ax, case_dir in zip(axes, cases_to_plot):

    coeff_file = ROOT / "re-run" / case_dir / \
                 "postProcessing" / "forceCoeffs" / "0" / "coefficient.dat"

    if not coeff_file.exists():
        print(f"File not found for case {case_dir}")
        continue

    data = np.loadtxt(coeff_file, comments="#")

    time = data[:, 0]
    Cl   = data[:, 4]   # RAW Cl (no filtering)

    # ---- Proper running time average ----
    cumulative_integral = cumulative_trapezoid(Cl, time, initial=0)
    running_avg = cumulative_integral / (time - time[0] + 1e-12)

    # ---- Steady mean calculation ----
    mask = (time >= t_start) & (time <= t_end)
    Cl_mean = np.trapz(Cl[mask], time[mask]) / (t_end - t_start)

    # ---- Plot instantaneous Cl (light gray) ----
    line1, = ax.plot(time, Cl, color="0.7", linewidth=1)

    # ---- Plot running average (blue) ----
    line2, = ax.plot(time, running_avg, color="C0", linewidth=2)

    # ---- Vertical steady-start line ----
    line3 = ax.axvline(t_start, color="k", linestyle="--", linewidth=1)

    # ---- Horizontal steady-mean line ----
    line4 = ax.axhline(Cl_mean, color="C1", linestyle="-", linewidth=1.5)

    ax.set_ylim(ymin, ymax)
    ax.set_ylabel(r"$C_L$")
    ax.set_title(f"Injection Rate = {case_dir}")
    ax.set_xlabel("Time (s)")
    ax.grid(alpha=0.2)

# =====================================================
# CLEAN GLOBAL LEGEND
# =====================================================
fig.legend(
    [line1, line2, line3, line4],
    [
        r"$C_L(t)$",
        r"Running average",
        r"$t_{\mathrm{steady}}$",
        r"$\overline{C_L}$"
    ],
    loc="upper center",
    bbox_to_anchor=(0.5, 0.965),
    ncol=4,
    frameon=False,
    handlelength=2.5,
    columnspacing=1.5
)

plt.suptitle("Time Evolution of Lift Coefficient and Statistical Convergence", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()