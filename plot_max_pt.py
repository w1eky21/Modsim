# plot_event_maxpt.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = {
    "pTHatMin = 0 GeV": "qcd_highpt_events_ptcut_0.csv",
    "pTHatMin = 500 GeV": "qcd_highpt_events_ptcut_500.csv",
}

bins = np.linspace(0, 2000, 100)

for label, filename in files.items():

    df = pd.read_csv(filename)

    # take the hardest final-state particle per event
    max_pt_per_event = df.groupby("event")["pT"].max()

    plt.figure(figsize=(10, 6))

    plt.hist(
        max_pt_per_event,
        bins=bins,
        histtype="step",
        linewidth=2
    )

    plt.xlabel(r"hardest final-state particle $p_T$ [GeV]")
    plt.ylabel(r"number of events")
    plt.title(label)
    plt.yscale("log")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    outname = filename.replace(".csv", "_maxpt_hist.png")
    plt.savefig(outname, dpi=300)

    print(f"Saved {outname}")
    print(f"{label}: mean max pT = {max_pt_per_event.mean():.2f} GeV")
    print(f"{label}: events with max pT > 500 GeV = {(max_pt_per_event > 500).sum()} / {len(max_pt_per_event)}")

plt.show()
