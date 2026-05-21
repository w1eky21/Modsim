import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_file = "phase_space_naive.csv"
outdir = "phase_space_extra_plots"
os.makedirs(outdir, exist_ok=True)

df = pd.read_csv(csv_file)

# Remove zero/negative values for log plots
df_log = df[(df["pTHat"] > 0) & (df["xprod"] > 0) & (df["sHat"] > 0) & (df["mHat"] > 0)]

# ------------------------------------------------------------
# 1. pTHat as main y-axis
# ------------------------------------------------------------

xvars = ["x1", "x2", "xprod", "sHat", "mHat", "alphaS", "Q2Fac", "Q2Ren"]

for var in xvars:
    plt.figure(figsize=(7, 5))
    plt.scatter(df[var], df["pTHat"], s=4, alpha=0.35)
    plt.xlabel(var)
    plt.ylabel("pTHat [GeV]")
    plt.title(f"pTHat vs {var}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/pTHat_vs_{var}.png", dpi=200)
    plt.close()

# ------------------------------------------------------------
# 2. Important log-log plots
# ------------------------------------------------------------

log_pairs = [
    ("xprod", "pTHat"),
    ("sHat", "pTHat"),
    ("mHat", "pTHat"),
    ("pTHat", "maxFinalParticlePT"),
]

for x, y in log_pairs:
    temp = df[(df[x] > 0) & (df[y] > 0)]

    plt.figure(figsize=(7, 5))
    plt.scatter(temp[x], temp[y], s=4, alpha=0.35)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"log-log {y} vs {x}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/loglog_{y}_vs_{x}.png", dpi=200)
    plt.close()

# ------------------------------------------------------------
# 3. Conditional histograms: all vs high pTHat
# ------------------------------------------------------------

thresholds = [10, 20, 50]
hist_vars = ["x1", "x2", "xprod", "sHat", "mHat", "alphaS", "Q2Fac", "Q2Ren"]

for threshold in thresholds:
    high = df[df["pTHat"] > threshold]

    for var in hist_vars:
        plt.figure(figsize=(7, 5))

        plt.hist(df[var], bins=60, density=True, alpha=0.5, label="all events")

        if len(high) > 0:
            plt.hist(
                high[var],
                bins=60,
                density=True,
                alpha=0.5,
                label=f"pTHat > {threshold} GeV",
            )

        plt.xlabel(var)
        plt.ylabel("normalised density")
        plt.title(f"{var}: all vs pTHat > {threshold} GeV")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_{var}_pTHat_gt_{threshold}.png", dpi=200)
        plt.close()

# ------------------------------------------------------------
# 4. Efficiency curves: P(pTHat > threshold | variable > cut)
# ------------------------------------------------------------

eff_vars = ["xprod", "sHat", "mHat", "Q2Fac", "Q2Ren"]
thresholds = [10, 20, 50]

for var in eff_vars:
    values = np.linspace(df[var].quantile(0.01), df[var].quantile(0.99), 80)

    for threshold in thresholds:
        fractions_high = []
        fractions_kept = []

        for cut in values:
            selected = df[df[var] > cut]

            if len(selected) == 0:
                fractions_high.append(np.nan)
                fractions_kept.append(0)
            else:
                fractions_high.append((selected["pTHat"] > threshold).mean())
                fractions_kept.append(len(selected) / len(df))

        plt.figure(figsize=(7, 5))
        plt.plot(values, fractions_high, label=f"P(pTHat > {threshold} GeV | {var} > cut)")
        plt.xlabel(f"{var} cut")
        plt.ylabel("conditional probability")
        plt.title(f"High-pTHat probability vs {var} cut")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/efficiency_probability_{var}_pTHat_gt_{threshold}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.plot(fractions_kept, fractions_high)
        plt.xlabel("fraction of all events kept")
        plt.ylabel(f"P(pTHat > {threshold} GeV)")
        plt.title(f"Efficiency tradeoff using {var}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/efficiency_tradeoff_{var}_pTHat_gt_{threshold}.png", dpi=200)
        plt.close()

# ------------------------------------------------------------
# 5. 2D heatmaps / binned averages
# ------------------------------------------------------------

heatmap_pairs = [
    ("x1", "x2"),
    ("xprod", "alphaS"),
    ("mHat", "pTHat"),
    ("xprod", "pTHat"),
]

for x, y in heatmap_pairs:
    plt.figure(figsize=(7, 5))
    h = plt.hist2d(df[x], df[y], bins=80, weights=df["pTHat"], cmap="viridis")
    counts, _, _ = np.histogram2d(df[x], df[y], bins=80)

    plt.colorbar(label="sum of pTHat weights")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"2D weighted density: {y} vs {x}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/heatmap_{y}_vs_{x}.png", dpi=200)
    plt.close()

# ------------------------------------------------------------
# 6. Weight plots
# ------------------------------------------------------------

if "weight" in df.columns:
    plt.figure(figsize=(7, 5))
    plt.hist(df["weight"], bins=60)
    plt.xlabel("event weight")
    plt.ylabel("events")
    plt.title("Event weight distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{outdir}/hist_weight.png", dpi=200)
    plt.close()

    for var in ["pTHat", "xprod", "sHat", "mHat"]:
        plt.figure(figsize=(7, 5))
        plt.scatter(df[var], df["weight"], s=4, alpha=0.35)
        plt.xlabel(var)
        plt.ylabel("event weight")
        plt.title(f"weight vs {var}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{outdir}/weight_vs_{var}.png", dpi=200)
        plt.close()

print(f"Saved extra plots in: {outdir}")
print(f"Total events: {len(df)}")
print("pTHat summary:")
print(df["pTHat"].describe())
