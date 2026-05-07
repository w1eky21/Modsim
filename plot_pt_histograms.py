# plot_pt_histograms.py
#
# Run with:
# python3 plot_pt_histograms.py

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------
# Files
# ----------------------------------------

file_pt0   = "pt_output_ptcut_0.txt"
file_pt500 = "pt_output_ptcut_500.txt"

# ----------------------------------------
# Function to load pT values
# ----------------------------------------

def load_pt(filename):

    pt_values = []

    with open(filename, "r") as f:

        for line in f:

            # skip header
            if line.startswith("#"):
                continue

            parts = line.split()

            # last column = pT
            pt = float(parts[-1])

            pt_values.append(pt)

    return np.array(pt_values)

# ----------------------------------------
# Load data
# ----------------------------------------

pt0   = load_pt(file_pt0)
pt500 = load_pt(file_pt500)

print("Loaded", len(pt0),   "particles from ptcut = 0")
print("Loaded", len(pt500), "particles from ptcut = 500")

# ----------------------------------------
# Histogram settings
# ----------------------------------------

bins = np.linspace(0, 2000, 100)

# ========================================
# Plot 1 : pTHatMin = 0
# ========================================

plt.figure(figsize=(10,6))

plt.hist(
    pt0,
    bins=bins,
    histtype='step',
    linewidth=2
)

plt.xlabel(r"$p_T$ [GeV]", fontsize=14)
plt.ylabel(r"$dN/dp_T$", fontsize=14)

plt.title(r"$p_T$ Distribution ($pTHatMin = 0$ GeV)", fontsize=16)

plt.yscale("log")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("pt_histogram_ptcut_0.png", dpi=300)

print("Saved pt_histogram_ptcut_0.png")

# ========================================
# Plot 2 : pTHatMin = 500
# ========================================

plt.figure(figsize=(10,6))

plt.hist(
    pt500,
    bins=bins,
    histtype='step',
    linewidth=2
)

plt.xlabel(r"$p_T$ [GeV]", fontsize=14)
plt.ylabel(r"$dN/dp_T$", fontsize=14)

plt.title(r"$p_T$ Distribution ($pTHatMin = 500$ GeV)", fontsize=16)

plt.yscale("log")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig("pt_histogram_ptcut_500.png", dpi=300)

print("Saved pt_histogram_ptcut_500.png")

# ----------------------------------------
# Show both plots
# ----------------------------------------

plt.show()