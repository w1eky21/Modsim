#!/usr/bin/env python3
"""
Plot comparison results from four_run_validation_with_powerlaw.cc.

Usage:
    python plot_four_run_validation.py validation plots_validation_powerlaw

This expects files named:
    validation_naive.csv
    validation_ptcut50.csv
    validation_powerlaw_bias.csv
    validation_hybrid_bias.csv
    validation_summary.csv

By default it uses the column `analysisWeight`. You can cross-check Pythia's own
weights with:
    python plot_four_run_validation.py validation plots_pythia_weight --weight-col pythiaWeight
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SAMPLES = [
    ("naive", "Naive"),
    ("ptcut50", r"Pythia $\hat{p}_T>50$ cut"),
    ("powerlaw_bias", r"Pythia power-law bias $n=5$"),
    ("hybrid_bias", "Hybrid bias"),
]


def savefig(outdir: Path, name: str):
    path = outdir / name
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"saved {path}")


def load_events(prefix: str):
    frames = []
    for key, _ in SAMPLES:
        path = Path(f"{prefix}_{key}.csv")
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        df["sample_key"] = key
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_summary(prefix: str):
    path = Path(f"{prefix}_summary.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return pd.read_csv(path)


def weighted_hist(ax, values, weights, bins, label, density=True):
    values = np.asarray(values)
    weights = np.asarray(weights)
    mask = np.isfinite(values) & np.isfinite(weights) & (values > 0)
    ax.hist(values[mask], bins=bins, weights=weights[mask], histtype="step", linewidth=1.8,
            density=density, label=label)


def plot_pthat_histograms(events, outdir, weight_col):
    bins = np.logspace(np.log10(1.0), np.log10(max(events["pTHat"].max(), 1000.0)), 80)

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for key, label in SAMPLES:
        sub = events[events["sample_key"] == key]
        weighted_hist(ax, sub["pTHat"], np.ones(len(sub)), bins, label, density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{p}_T$ [GeV]")
    ax.set_ylabel("Generated density")
    ax.set_title(r"Generated $\hat{p}_T$ distribution")
    ax.legend()
    savefig(outdir, "01_generated_pthat_density.png")

    plt.figure(figsize=(8, 5))
    ax = plt.gca()
    for key, label in SAMPLES:
        sub = events[events["sample_key"] == key]
        weights = sub[weight_col].to_numpy()
        weighted_hist(ax, sub["pTHat"], weights, bins, label, density=True)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{p}_T$ [GeV]")
    ax.set_ylabel("Weighted density")
    ax.set_title(r"Weighted $\hat{p}_T$ distribution")
    ax.legend()
    savefig(outdir, "02_weighted_pthat_density.png")


def plot_tail_fractions(summary, outdir):
    thresholds = [10, 20, 50, 100, 200, 500, 1000]
    x = np.arange(len(thresholds))
    width = 0.20

    plt.figure(figsize=(10, 5))
    for i, (key, label) in enumerate(SAMPLES):
        row = summary[summary["sample"] == key].iloc[0]
        vals = [row[f"fracPT{t}"] for t in thresholds]
        plt.bar(x + (i - 1.5) * width, vals, width, label=label)
    plt.yscale("log")
    plt.xticks(x, [f">{t}" for t in thresholds])
    plt.xlabel(r"$\hat{p}_T$ threshold [GeV]")
    plt.ylabel("Generated fraction")
    plt.title(r"Generated high-$\hat{p}_T$ fractions")
    plt.legend()
    savefig(outdir, "03_generated_tail_fractions.png")

    plt.figure(figsize=(10, 5))
    for i, (key, label) in enumerate(SAMPLES):
        row = summary[summary["sample"] == key].iloc[0]
        vals = [row[f"wFracPT{t}"] for t in thresholds]
        plt.bar(x + (i - 1.5) * width, vals, width, label=label)
    plt.yscale("log")
    plt.xticks(x, [f">{t}" for t in thresholds])
    plt.xlabel(r"$\hat{p}_T$ threshold [GeV]")
    plt.ylabel("Weighted fraction")
    plt.title(r"Weighted high-$\hat{p}_T$ fractions")
    plt.legend()
    savefig(outdir, "04_weighted_tail_fractions.png")


def plot_neff_and_means(summary, outdir):
    labels = [label for _, label in SAMPLES]
    rows = [summary[summary["sample"] == key].iloc[0] for key, _ in SAMPLES]

    plt.figure(figsize=(9, 5))
    plt.bar(labels, [r["NeffOverN"] for r in rows])
    plt.ylabel(r"$N_{\mathrm{eff}}/N$")
    plt.title("Effective sample-size ratio")
    plt.xticks(rotation=10, ha="right")
    savefig(outdir, "05_neff_over_n_comparison.png")

    plt.figure(figsize=(9, 5))
    plt.bar(labels, [r["meanPTHat"] for r in rows])
    plt.ylabel(r"Mean $\hat{p}_T$ [GeV]")
    plt.title(r"Average generated hardness")
    plt.xticks(rotation=10, ha="right")
    savefig(outdir, "06_mean_pthat_comparison.png")


def plot_weight_diagnostics(events, outdir, weight_col):
    plt.figure(figsize=(8, 5))
    for key, label in SAMPLES:
        sub = events[events["sample_key"] == key]
        w = sub[weight_col].to_numpy()
        w = w[np.isfinite(w) & (w > 0)]
        if len(w) == 0:
            continue
        bins = np.logspace(np.log10(w.min()), np.log10(w.max()), 70) if w.max() > w.min() else 20
        plt.hist(w, bins=bins, histtype="step", linewidth=1.8, label=label, density=True)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(weight_col)
    plt.ylabel("Density")
    plt.title("Weight distribution")
    plt.legend()
    savefig(outdir, "07_weight_distribution.png")


def plot_summary_table(summary, outdir):
    cols = ["sample", "accepted", "NeffOverN", "fracPT50", "wFracPT50", "meanPTHat", "maxPTHat"]
    table_df = summary[cols].copy()
    for c in ["NeffOverN", "fracPT50", "wFracPT50", "meanPTHat", "maxPTHat"]:
        table_df[c] = table_df[c].map(lambda x: f"{x:.4g}")

    plt.figure(figsize=(11, 2.8))
    plt.axis("off")
    table = plt.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.title("Four-run validation")
    savefig(outdir, "08_summary_table.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="Output prefix used by four_run_validation_with_powerlaw.cc, e.g. validation")
    parser.add_argument(
    "outdir",
    nargs="?",
    default="four_run_val_plots"
    )
    parser.add_argument("--weight-col", default="analysisWeight",
                        help="Weight column to use: analysisWeight, pythiaWeight, or manualWeight")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    events = load_events(args.prefix)
    summary = load_summary(args.prefix)

    if args.weight_col not in events.columns:
        raise KeyError(f"Column {args.weight_col!r} not found in event CSVs")

    plot_pthat_histograms(events, outdir, args.weight_col)
    plot_tail_fractions(summary, outdir)
    plot_neff_and_means(summary, outdir)
    plot_weight_diagnostics(events, outdir, args.weight_col)
    plot_summary_table(summary, outdir)

    print(f"\nAll plots saved in: {outdir}")


if __name__ == "__main__":
    main()
