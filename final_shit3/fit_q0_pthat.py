#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def q0_model(pT, A, p0, n):
    return A * (1.0 + pT / p0) ** (-n)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="e.g. validation_naive.csv")
    parser.add_argument("--out", default="q0_fit")
    parser.add_argument("--min", type=float, default=2.0)
    parser.add_argument("--max", type=float, default=1000.0)
    parser.add_argument("--bins", type=int, default=60)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    pT = df["pTHat"].to_numpy()
    pT = pT[np.isfinite(pT)]
    pT = pT[(pT > args.min) & (pT < args.max)]

    bins = np.logspace(np.log10(args.min), np.log10(args.max), args.bins)
    hist, edges = np.histogram(pT, bins=bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])

    mask = hist > 0
    x = centers[mask]
    y = hist[mask]

    popt, pcov = curve_fit(
        q0_model,
        x,
        y,
        p0=[1.0, 10.0, 5.0],
        bounds=([0.0, 0.1, 0.0], [np.inf, 1000.0, 20.0]),
        maxfev=100000,
    )

    A, p0, n = popt

    print("\nFitted naive distribution:")
    print(f"q0(pT) = A * (1 + pT/p0)^(-n)")
    print(f"A  = {A:.6e}")
    print(f"p0 = {p0:.6f}")
    print(f"n  = {n:.6f}")

    print("\nSuggested inverse bias:")
    print(f"b(pT) ∝ (1 + pT/{p0:.6f})^{n:.6f}")

    print("\nFor your current hybrid form, a rough starting value is:")
    print(f"a_pT ≈ {n:.3f}")

    xx = np.logspace(np.log10(args.min), np.log10(args.max), 500)

    plt.figure(figsize=(8, 5))
    plt.hist(pT, bins=bins, density=True, histtype="step", label="Naive MC")
    plt.plot(xx, q0_model(xx, *popt), label="Fit")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\hat{p}_T$ [GeV]")
    plt.ylabel(r"$q_0(\hat{p}_T)$")
    plt.title("Fit of naive Pythia distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out + "_q0_fit.png", dpi=200)
    plt.close()

    with open(args.out + "_params.txt", "w") as f:
        f.write("q0(pT) = A * (1 + pT/p0)^(-n)\n")
        f.write(f"A = {A:.12e}\n")
        f.write(f"p0 = {p0:.12f}\n")
        f.write(f"n = {n:.12f}\n")
        f.write("\n")
        f.write(f"Suggested bias: b(pT) = (1 + pT/{p0:.12f})^{n:.12f}\n")

if __name__ == "__main__":
    main()