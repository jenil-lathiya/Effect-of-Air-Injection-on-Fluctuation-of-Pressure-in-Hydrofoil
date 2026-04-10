import numpy as np
from pathlib import Path

# ================== EDIT THIS ==================
ROOT = Path("last_folder")
CASE = "0.7"
POST_DIR = ROOT / "re-run"/ CASE / "postProcessing"
OUT_FILE = ROOT / "re-run"/ CASE /"postProcessing" / "p_taps_all.txt"
# ==============================================


def load_p_for_tap(tap_dir):
    """
    Load and merge p data for one tap across all restart folders.
    Returns: time (N,), p (N,)
    """
    blocks = []

    time_dirs = [d for d in tap_dir.iterdir() if d.is_dir()]

    def keyfunc(d):
        try:
            return float(d.name)
        except:
            return 1e99

    for d in sorted(time_dirs, key=keyfunc):
        f = d / "p"
        if not f.exists():
            continue
        data = np.loadtxt(f, comments="#")
        if data.size == 0:
            continue
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
        if not np.array_equal(time_ref, t):
            raise ValueError(
                f"Time mismatch in {tap_dir.name}. "
                f"All taps must have identical time stamps."
            )

    p_all.append(p)

# -------- stack and write --------
# columns: time | tap1 | tap2 | ... | tap11
data_out = np.column_stack([time_ref] + p_all)

header = "time " + " ".join([f"tap{i}" for i in range(1, len(p_all) + 1)])

np.savetxt(
    OUT_FILE,
    data_out,
    header=header,
    comments="",
    fmt="%.8e"
)

print("\nDONE ")
print("Saved file:", OUT_FILE)
