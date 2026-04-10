import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# Define all your cases (folder → label on plot)
# ------------------------------------------------------
cases = { 
    "q=0.03":   "last_folder/0.125/postProcessing/forceCoeffs/0/coefficient.dat",
    #"q=0.2": "q=0.2/forceCoeffs/0/coefficient.dat",
    #"q=0.4": "q=0.4/forceCoeffs/0/coefficient.dat",
    #"q=0.6": "q=0.6/forceCoeffs/0/coefficient.dat"
}

# Colors for each line (optional)
colors = ["blue", "red", "green", "purple", "orange"]

plt.figure(figsize=(12,6))

# ------------------------------------------------------
# Function: automatic fully-developed detection
# ------------------------------------------------------
def detect_fully_developed(time, running_avg, slope_tol=1e-4, window=20):
    slopes = np.abs(np.gradient(running_avg, time))

    for i in range(window, len(slopes)):
        if np.mean(slopes[i-window:i]) < slope_tol:
            return i

    return len(running_avg) - 1

# ------------------------------------------------------
# Loop through all q-cases and plot them
# ------------------------------------------------------
for idx, (label, path) in enumerate(cases.items()):
    
    # Load data
    data = np.loadtxt(path, comments="#")
    time = data[:, 0]
    Cl   = data[:, 4]     # lift coefficient
    
    # Compute running average of Cl
    Cl_running = np.cumsum(Cl) / np.arange(1, len(Cl)+1)
    
     #Detect fully-developed time
    dev_idx = detect_fully_developed(time, Cl_running)
    t_dev = time[dev_idx]
    
    # Plot
    plt.plot(time, Cl_running, linewidth=2, label=label, color=colors[idx])
    
    # Add dashed vertical line for fully-developed region
    plt.axvline(t_dev, linestyle='--', color=colors[idx], alpha=0.5)


# ------------------------------------------------------
# Final plot formatting
# ------------------------------------------------------
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Time-averaged $C_L$", fontsize=14)
plt.title("Time-Averaged $C_L$ for Different Injection Rates", fontsize=16)

plt.grid(True, alpha=0.4)
plt.legend(title="Injection rates", fontsize=12)

plt.tight_layout()
plt.show()