import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")
CASE = "q=0.05"

# steady window start time (use your steady start)
T_START = 0.16

POST_DIR = ROOT / CASE / "postProcessing"
OUT_FILE = POST_DIR / "p_taps_all.txt"
OUT_RMS_FILE = POST_DIR / "p_rms_by_tap.txt"
# ==============================================


def load_p_for_tap(tap_dir: Path):
    """
    Load and merge p data for one tap across all time folders.
    Each folder contains a file named 'p' with columns: time p
    Returns: time (N,), p (N,)
    """
    blocks = []
    time_dirs = [d for d in tap_dir.iterdir() if d.is_dir()]

    def keyfunc(d):
        try:
            return float(d.name)
        except Exception:
            return 1e99

    for d in sorted(time_dirs, key=keyfunc):
        f = d / "p"
        if not f.exists():
            continue
        data = np.loadtxt(f, comments="#")
        if data.size == 0:
            continue
        # handle case when loadtxt returns shape (2,) for single line
        data = np.atleast_2d(data)
        blocks.append(data)

    if not blocks:
        return None, None

    full = np.vstack(blocks)

    # remove duplicate times (keep last)
    full_rev = full[::-1]
    _, idx = np.unique(full_rev[:, 0], return_index=True)
    full = full_rev[np.sort(idx)][::-1]

    time = full[:, 0]
    p = full[:, 1]
    return time, p


# -------- find taps automatically --------
tap_dirs = sorted(
    [d for d in POST_DIR.iterdir() if d.is_dir() and d.name.startswith("tap")],
    key=lambda x: int(x.name.replace("tap", ""))
)

if len(tap_dirs) != 11:
    print(f"[INFO] Found {len(tap_dirs)} taps (expected 11)")

print("Taps found:", [d.name for d in tap_dirs])

# -------- load all taps --------
time_ref = None
p_all = []

for tap_dir in tap_dirs:
    t, p = load_p_for_tap(tap_dir)
    if t is None:
        raise RuntimeError(f"No p data found in {tap_dir}")

    if time_ref is None:
        time_ref = t
    else:
        # if time stamps are not identical, you can interpolate instead,
        # but for now we enforce exact match (as in your script).
        if not np.array_equal(time_ref, t):
            raise ValueError(
                f"Time mismatch in {tap_dir.name}. "
                f"All taps must have identical time stamps."
            )

    p_all.append(p)

# -------- stack and write combined taps file --------
data_out = np.column_stack([time_ref] + p_all)
header = "time " + " ".join([f"tap{i}" for i in range(1, len(p_all) + 1)])

OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
np.savetxt(
    OUT_FILE,
    data_out,
    header=header,
    comments="",
    fmt="%.8e"
)

print("\nDONE ✅ Combined tap file saved:")
print("Saved file:", OUT_FILE)

# ==========================================================
# 2) Compute p_rms at each tap (steady window) + plot
# ==========================================================

# Select steady region
T_START = 0.16
mask = time_ref >= T_START

prime2Mean_vals = []
for p in p_all:
    p_steady = p[mask]
    p_mean = np.mean(p_steady)
    pprime2 = np.mean((p_steady - p_mean)**2)   # <-- prime2Mean
    prime2Mean_vals.append(pprime2)

prime2Mean_vals = np.array(prime2Mean_vals)
tap_nums = np.arange(1, len(prime2Mean_vals) + 1)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "figure.figsize": (6.5, 4.0),
    "figure.dpi": 300
})

fig, ax = plt.subplots()

ax.plot(tap_nums, prime2Mean_vals, marker='o')

ax.set_xlabel("Tap number")
ax.set_ylabel(r"$\overline{p^{\prime 2}}\;\mathrm{[Pa^2]}$")

ax.tick_params(direction='in', which='both', top=True, right=True)
ax.minorticks_on()

ax.grid(True, which='major', linestyle=':', linewidth=0.6, alpha=0.6)

plt.tight_layout()
plt.show()
