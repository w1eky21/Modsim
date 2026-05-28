# plot_static_importance.py
#
# Use after running:
#   ./test_pythia8_static_is
#
# It reads:
#   static_importance_events.csv
#   static_importance_histogram.csv
#   static_importance_summary.csv
#
# Run with:
#   python3 plot_static_importance.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Input files from test_pythia8_static_is.cc
# ------------------------------------------------------------

EVENT_FILE = "static_importance_events.csv"
HIST_FILE = "static_importance_histogram.csv"
SUMMARY_FILE = "static_importance_summary.csv"

for filename in [EVENT_FILE, HIST_FILE, SUMMARY_FILE]:
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Could not find {filename}. Run ./test_pythia8_static_is first."
        )

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

events = pd.read_csv(EVENT_FILE)
hist = pd.read_csv(HIST_FILE)
summary = pd.read_csv(SUMMARY_FILE)

print("Loaded files:")
print(f"  {EVENT_FILE}: {len(events)} events")
print(f"  {HIST_FILE}:  {len(hist)} histogram bins")
print(f"  {SUMMARY_FILE}: {len(summary)} pTHat bins")

# Add useful columns
hist["ptCenter"] = 0.5 * (hist["ptLow"] + hist["ptHigh"])
hist["absError"] = hist["relativeError"] * hist["dSigma_dpT"]
summary["label"] = summary.apply(
    lambda r: f"{int(r['pTHatMin'])}-{int(r['pTHatMax'])} GeV", axis=1
)
events["label"] = events["binIndex"].map(summary.set_index("binIndex")["label"])

# ------------------------------------------------------------
# Plot 1: final weighted cross-section estimate
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.errorbar(
    hist["ptCenter"],
    hist["dSigma_dpT"],
    yerr=hist["absError"],
    xerr=0.5 * hist["binWidth"],
    fmt="o",
    capsize=3,
)

plt.xlabel(r"hardest final-state $p_T$ [GeV]", fontsize=13)
plt.ylabel(r"$d\sigma/dp_T$ [mb/GeV]", fontsize=13)
plt.title(r"Static importance sampling: weighted $p_T$ spectrum", fontsize=15)
plt.yscale("log")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("static_importance_weighted_spectrum.png", dpi=300)
print("Saved static_importance_weighted_spectrum.png")

# ------------------------------------------------------------
# Plot 2: relative error per final pT bin
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.step(
    hist["ptLow"],
    hist["relativeError"],
    where="post",
    linewidth=2,
)

# draw last horizontal segment up to the final edge
plt.hlines(
    hist["relativeError"].iloc[-1],
    hist["ptLow"].iloc[-1],
    hist["ptHigh"].iloc[-1],
    linewidth=2,
)

plt.xlabel(r"hardest final-state $p_T$ [GeV]", fontsize=13)
plt.ylabel(r"relative error estimate", fontsize=13)
plt.title(r"Statistical uncertainty after static importance sampling", fontsize=15)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("static_importance_relative_error.png", dpi=300)
print("Saved static_importance_relative_error.png")

# ------------------------------------------------------------
# Plot 3: which pTHat bins were generated and their cross sections
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

x = np.arange(len(summary))
plt.bar(x, summary["sigmaGen"])

plt.xticks(x, summary["label"], rotation=30, ha="right")
plt.xlabel(r"generated $\hat{p}_T$ interval", fontsize=13)
plt.ylabel(r"generated cross section $\sigma_\mathrm{gen}$ [mb]", fontsize=13)
plt.title(r"Cross section of each static importance-sampling bin", fontsize=15)
plt.yscale("log")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("static_importance_sigma_per_pthat_bin.png", dpi=300)
print("Saved static_importance_sigma_per_pthat_bin.png")

# ------------------------------------------------------------
# Plot 4: unweighted event coverage per pTHat sample
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

bins = np.linspace(0, 2000, 80)

for label in summary["label"]:
    subset = events[events["label"] == label]
    plt.hist(
        subset["hardestFinalPt"],
        bins=bins,
        histtype="step",
        linewidth=1.8,
        label=label,
    )

plt.xlabel(r"hardest final-state $p_T$ [GeV]", fontsize=13)
plt.ylabel(r"unweighted event count", fontsize=13)
plt.title(r"Phase-space coverage of the separate $\hat{p}_T$ samples", fontsize=15)
plt.yscale("log")
plt.legend(title=r"generated $\hat{p}_T$ bin", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("static_importance_unweighted_coverage.png", dpi=300)
print("Saved static_importance_unweighted_coverage.png")

# ------------------------------------------------------------
# Plot 5: weighted contribution of each pTHat bin to final spectrum
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

for _, row in summary.iterrows():
    label = row["label"]
    subset = events[events["binIndex"] == row["binIndex"]]
    weights = subset["eventWeight"].to_numpy()

    counts, edges = np.histogram(
        subset["hardestFinalPt"].to_numpy(),
        bins=hist["ptLow"].to_list() + [hist["ptHigh"].iloc[-1]],
        weights=weights,
    )

    widths = np.diff(edges)
    density = counts / widths
    centers = 0.5 * (edges[:-1] + edges[1:])

    plt.step(centers, density, where="mid", linewidth=1.8, label=label)

plt.xlabel(r"hardest final-state $p_T$ [GeV]", fontsize=13)
plt.ylabel(r"weighted contribution to $d\sigma/dp_T$ [mb/GeV]", fontsize=13)
plt.title(r"Weighted contribution from each $\hat{p}_T$ sample", fontsize=15)
plt.yscale("log")
plt.legend(title=r"generated $\hat{p}_T$ bin", fontsize=9)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("static_importance_weighted_contributions.png", dpi=300)
print("Saved static_importance_weighted_contributions.png")

plt.show()
