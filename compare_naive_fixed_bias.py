import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

outdir = "bias_comparison_plots"
os.makedirs(outdir, exist_ok=True)

files = {
    "naive": "phase_space_naive.csv",
    "bias_pow4": "phase_space_fixed_bias.csv",
}

thresholds = [10, 20, 50, 100]

rows = []

for label, filename in files.items():
    df = pd.read_csv(filename)

    w = df["weight"].to_numpy()
    pthat = df["pTHat"].to_numpy()

    neff = (w.sum() ** 2) / np.sum(w ** 2)

    row = {
        "label": label,
        "file": filename,
        "N": len(df),
        "sumW": w.sum(),
        "sumW2": np.sum(w ** 2),
        "meanW": w.mean(),
        "stdW": w.std(),
        "Neff": neff,
        "Neff_over_N": neff / len(df),
        "mean_pTHat": pthat.mean(),
        "max_pTHat": pthat.max(),
    }

    for t in thresholds:
        row[f"frac_pTHat_gt_{t}"] = np.mean(pthat > t)

    rows.append(row)

summary = pd.DataFrame(rows)
summary.to_csv("bias_comparison_summary.csv", index=False)
print(summary)

# Weighted pTHat distributions
bins = np.logspace(-1, 2.2, 80)

plt.figure(figsize=(8, 5))

for label, filename in files.items():
    df = pd.read_csv(filename)
    plt.hist(
        df["pTHat"],
        bins=bins,
        weights=df["weight"],
        histtype="step",
        density=True,
        label=label,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("pTHat [GeV]")
plt.ylabel("weighted normalised density")
plt.title("Weighted pTHat distribution: naive vs fixed bias")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/weighted_pTHat_naive_vs_bias.png", dpi=200)
plt.close()

# Unweighted pTHat distributions
plt.figure(figsize=(8, 5))

for label, filename in files.items():
    df = pd.read_csv(filename)
    plt.hist(
        df["pTHat"],
        bins=bins,
        histtype="step",
        density=True,
        label=label,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("pTHat [GeV]")
plt.ylabel("unweighted normalised density")
plt.title("Unweighted pTHat distribution: naive vs fixed bias")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/unweighted_pTHat_naive_vs_bias.png", dpi=200)
plt.close()

# Weight histograms
plt.figure(figsize=(8, 5))

for label, filename in files.items():
    df = pd.read_csv(filename)
    plt.hist(
        df["weight"],
        bins=80,
        histtype="step",
        density=True,
        label=label,
    )

plt.yscale("log")
plt.xlabel("event weight")
plt.ylabel("density")
plt.title("Event weight distribution")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/weight_distribution.png", dpi=200)
plt.close()

print(f"Saved plots in {outdir}")
print("Saved table: bias_comparison_summary.csv")
