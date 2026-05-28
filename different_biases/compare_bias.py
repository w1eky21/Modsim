import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

outdir = "bias_comparison_plots2"
os.makedirs(outdir, exist_ok=True)

files = {
    "naive": "phase_space_naive.csv",

    "pow1": "fixed_bias_pow1.csv",
    "pow2": "fixed_bias_pow2.csv",
    "pow3": "fixed_bias_pow3.csv",
    "pow4": "fixed_bias_pow4.csv",
    "pow6": "fixed_bias_pow6.csv",

     # Pythia hard pTHat cuts 
    "ptcut_10": "qcd_highpt_events_ptcut_10.csv",
    "ptcut_50": "qcd_highpt_events_ptcut_50.csv",
    "ptcut_100": "qcd_highpt_events_ptcut_100.csv",
    "ptcut_500": "qcd_highpt_events_ptcut_500_eventlevel.csv",

    "sat_n1_p010": "saturating_n1_p010.csv",
    "sat_n2_p010": "saturating_n2_p010.csv",
    "sat_n3_p010": "saturating_n3_p010.csv",
    "sat_n1_p020": "saturating_n1_p020.csv",
    "sat_n2_p020": "saturating_n2_p020.csv",
    "sat_n3_p020": "saturating_n3_p020.csv",
    "sat_n1_p050": "saturating_n1_p050.csv",
    "sat_n2_p050": "saturating_n2_p050.csv",
    "sat_n3_p050": "saturating_n3_p050.csv",

    "powexp_n3_cut200": "powexp_n3_p015_cut200.csv",
    "powexp_n4_cut200": "powexp_n4_p015_cut200.csv",
    "powexp_n4_cut500": "powexp_n4_p015_cut500.csv",
    "powexp_n5_cut500": "powexp_n5_p015_cut500.csv",

    "bump_A50_pc75": "bump_A50_pc75_sig08.csv",
    "bump_A100_pc100": "bump_A100_pc100_sig08.csv",
    "bump_A100_pc150": "bump_A100_pc150_sig10.csv",
}

thresholds = [10, 20, 50, 100]
rows = []

files = {label: fname for label, fname in files.items() if os.path.exists(fname)}

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
summary.to_csv("bias_comparison_summary_all.csv", index=False)
print(summary)

bins = np.logspace(-1, 4, 100)

# Weighted pTHat
plt.figure(figsize=(10, 6))
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
plt.title("Weighted pTHat distribution")
plt.legend(fontsize=6)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/weighted_pTHat_all_biases.png", dpi=200)
plt.close()

# Unweighted pTHat
plt.figure(figsize=(10, 6))
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
plt.title("Unweighted pTHat distribution")
plt.legend(fontsize=6)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/unweighted_pTHat_all_biases.png", dpi=200)
plt.close()

# Weight distributions
plt.figure(figsize=(10, 6))
for label, filename in files.items():
    df = pd.read_csv(filename)
    w = df["weight"]
    w = w[w > 0]
    plt.hist(
        w,
        bins=np.logspace(-8, 8, 120),
        histtype="step",
        density=True,
        label=label,
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("event weight")
plt.ylabel("density")
plt.title("Event weight distribution")
plt.legend(fontsize=6)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/weight_distribution_all_biases.png", dpi=200)
plt.close()

# Neff/N comparison
plt.figure(figsize=(11, 5))
plt.bar(summary["label"], summary["Neff_over_N"])
plt.yscale("log")
plt.ylabel("Neff / N")
plt.title("Effective sample size fraction")
plt.xticks(rotation=60, ha="right")
plt.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/neff_over_n_comparison.png", dpi=200)
plt.close()

# Rare event fractions
plt.figure(figsize=(11, 6))
x = np.arange(len(summary))

for t in thresholds:
    plt.plot(
        x,
        summary[f"frac_pTHat_gt_{t}"],
        marker="o",
        label=f"pTHat > {t} GeV",
    )

plt.yscale("log")
plt.xticks(x, summary["label"], rotation=60, ha="right")
plt.ylabel("fraction of events")
plt.title("Rare high-pTHat event fractions")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/rare_event_fraction_comparison.png", dpi=200)
plt.close()

# Tradeoff plot: Neff/N vs high-pT fraction
plt.figure(figsize=(8, 6))

plt.scatter(summary["Neff_over_N"], summary["frac_pTHat_gt_50"])

for _, row in summary.iterrows():
    plt.annotate(
        row["label"],
        (row["Neff_over_N"], row["frac_pTHat_gt_50"]),
        fontsize=7,
        xytext=(4, 3),
        textcoords="offset points",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Neff / N")
plt.ylabel("fraction pTHat > 50 GeV")
plt.title("Tradeoff: efficiency vs high-pTHat enhancement")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/tradeoff_neff_vs_highpT.png", dpi=200)
plt.close()

# Focus plot: only useful candidates
candidate_labels = [
    "naive",
    "pow1",
    "pow2",
    "pow3",
    "sat_n3_p010",
    "powexp_n3_cut200",
    "powexp_n4_cut200",
    "powexp_n4_cut500",
    "bump_A50_pc75",
    "bump_A100_pc100",
    "bump_A100_pc150",
]

candidate_summary = summary[summary["label"].isin(candidate_labels)]

plt.figure(figsize=(8, 6))
plt.scatter(
    candidate_summary["Neff_over_N"],
    candidate_summary["frac_pTHat_gt_50"],
)

for _, row in candidate_summary.iterrows():
    plt.annotate(
        row["label"],
        (row["Neff_over_N"], row["frac_pTHat_gt_50"]),
        fontsize=8,
        xytext=(4, 3),
        textcoords="offset points",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Neff / N")
plt.ylabel("fraction pTHat > 50 GeV")
plt.title("Best candidate bias functions")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{outdir}/best_candidate_tradeoff.png", dpi=200)
plt.close()

print(f"Saved plots in {outdir}")
print("Saved table: bias_comparison_summary_all.csv")