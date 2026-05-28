import pandas as pd
import matplotlib.pyplot as plt
import os

csv_file = "phase_space_naive.csv"
outdir = "phase_space_plots"

os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(csv_file)

pt_col = "maxFinalParticlePT"

variables = [
    "x1",
    "x2",
    "xprod",
    "sHat",
    "mHat",
    "pTHat",
    "Q2Fac",
    "Q2Ren",
    "alphaS",
    "nFinalParticles",
    "nChargedFinal",
]

for var in variables:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[var], df[pt_col], s=4, alpha=0.4)
    plt.xlabel(var)
    plt.ylabel("max final-state particle pT [GeV]")
    plt.title(f"{pt_col} vs {var}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/{pt_col}_vs_{var}.png", dpi=200)
    plt.close()

# Useful log plots
log_variables = ["x1", "x2", "xprod", "sHat", "mHat", "pTHat", "Q2Fac"]

for var in log_variables:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[var], df[pt_col], s=4, alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(var)
    plt.ylabel("max final-state particle pT [GeV]")
    plt.title(f"log-log {pt_col} vs {var}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/loglog_{pt_col}_vs_{var}.png", dpi=200)
    plt.close()

# Histograms for all events
for var in variables:
    plt.figure(figsize=(7, 5))
    plt.hist(df[var], bins=60)
    plt.xlabel(var)
    plt.ylabel("events")
    plt.title(f"Distribution of {var}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_{var}.png", dpi=200)
    plt.close()

# Compare all events vs high-pT events
threshold = 500.0
high = df[df[pt_col] > threshold]

compare_vars = ["x1", "x2", "xprod", "pTHat", "Q2Fac", "mHat"]

for var in compare_vars:
    plt.figure(figsize=(7, 5))
    plt.hist(df[var], bins=60, alpha=0.5, label="all events", density=True)
    plt.hist(high[var], bins=60, alpha=0.5, label=f"{pt_col} > {threshold} GeV", density=True)
    plt.xlabel(var)
    plt.ylabel("normalised density")
    plt.title(f"{var}: all vs high-pT events")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/compare_{var}_highpt.png", dpi=200)
    plt.close()

print(f"Saved plots in: {outdir}")
print(f"Total events: {len(df)}")
print(f"Events with {pt_col} > {threshold} GeV: {len(high)}")
