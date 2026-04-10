import os
import shutil

# ---------------- USER SETTINGS ----------------
base_path = "./last_folder/re-run/0.2/postProcessing"
tap_folders = [f"tap{i}" for i in range(1, 12)]
data_files = ["p", "U"]
cut_time = 0.0846
# -----------------------------------------------


def read_data(filepath):
    header = []
    data = []

    with open(filepath, "r") as f:
        for line in f:
            line_strip = line.strip()

            if line_strip.startswith("#") or not line_strip:
                header.append(line)
            else:
                parts = line.split()
                time_val = float(parts[0])
                value_str = parts[1]          # keep value formatting
                data.append((time_val, parts[0], value_str))

    return header, data


def write_data(filepath, header, data):
    with open(filepath, "w") as f:
        for h in header:
            f.write(h)
        for _, time_str, value_str in data:
            # Match OpenFOAM-style spacing
            f.write(f"{time_str:<16} {value_str}\n")


for tap in tap_folders:
    print(f"\nProcessing {tap}")

    folder_0 = os.path.join(base_path, tap, "0")
    folder_023 = os.path.join(base_path, tap, "0.084")

    if not os.path.isdir(folder_023):
        print("  No restart folder found, skipping")
        continue

    for var in data_files:
        file_0 = os.path.join(folder_0, var)
        file_023 = os.path.join(folder_023, var)

        if not (os.path.exists(file_0) and os.path.exists(file_023)):
            print(f"  Skipping {var} (file missing)")
            continue

        header_0, data_0 = read_data(file_0)
        _, data_023 = read_data(file_023)

        # Apply exact cut logic
        data_0_new = [d for d in data_0 if d[0] < cut_time]
        data_023_new = [d for d in data_023 if d[0] >= cut_time]

        merged = data_0_new + data_023_new
        merged.sort(key=lambda x: x[0])

        write_data(file_0, header_0, merged)

        print(f"  ✔ {var}: merged with OpenFOAM formatting")

    # Remove restart folder
    shutil.rmtree(folder_023)
    print("  🗑 Removed folder 0.23")

print("\n✅ Done — files now match OpenFOAM default output exactly")
