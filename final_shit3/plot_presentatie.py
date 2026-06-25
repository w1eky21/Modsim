#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SAMPLES = [
    ("naive", "Naive MC"),
    ("ptcut50", r"Pythia $\hat{p}_T > 50$ cut"),
    ("powerlaw_bias", r"Pythia power-law bias $n=5$"),
]


def load_sample(prefix, key):
    path = Path(f"{prefix}_{key}.csv")
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def plot_samples(samples, outpath, title, weight_col=None):
    all_pthat = np.concatenate([df["pTHat"].to_numpy() for df, _ in samples])
    bins = np.logspace(np.log10(1.0), np.log10(max(all_pthat.max(), 1000.0)), 80)

    plt.figure(figsize=(9, 5.5))
    ax = plt.gca()

    for df, label in samples:
        values = df["pTHat"].to_numpy()

        if weight_col is None:
            weights = np.ones(len(df))
            ylabel = "Generated density"
        else:
            weights = df[weight_col].to_numpy()
            ylabel = "Weighted density"

        ax.hist(
            values,
            bins=bins,
            weights=weights,
            histtype="step",
            linewidth=2.0,
            density=True,
            label=label,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\hat{p}_T$ [GeV]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()

    print(f"Saved {outpath}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prefix", help="File prefix, e.g. validation")
    parser.add_argument("outdir", nargs="?", default="progressive_plots")
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Use weighted distributions instead of generated distributions",
    )
    parser.add_argument(
        "--weight-col",
        default="analysisWeight",
        help="Weight column to use if --weighted is enabled",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    data = {}
    for key, label in SAMPLES:
        data[key] = (load_sample(args.prefix, key), label)

    weight_col = args.weight_col if args.weighted else None
    tag = "weighted" if args.weighted else "generated"

    # 1. Naive MC only
    plot_samples(
        [data["naive"]],
        outdir / f"01_{tag}_naive_only.png",
        rf"{'Weighted' if args.weighted else 'Generated'} $\hat{{p}}_T$ distribution: naive MC",
        weight_col,
    )

    # 2. Naive MC + pT cut
    plot_samples(
        [data["naive"], data["ptcut50"]],
        outdir / f"02_{tag}_naive_plus_ptcut50.png",
        rf"{'Weighted' if args.weighted else 'Generated'} $\hat{{p}}_T$ distribution: naive MC vs $\hat{{p}}_T>50$ cut",
        weight_col,
    )

    # 3. Naive MC + pT cut + power-law bias
    plot_samples(
        [data["naive"], data["ptcut50"], data["powerlaw_bias"]],
        outdir / f"03_{tag}_naive_ptcut50_powerlaw.png",
        rf"{'Weighted' if args.weighted else 'Generated'} $\hat{{p}}_T$ distribution: adding power-law bias",
        weight_col,
    )


if __name__ == "__main__":
    main()
