"""Plot physical-core-speed trajectories with the 10 evaluation targets
highlighted in red.

Mirrors paper Figure (Core_Speed_Highlights). All 100 machines in transparent
blue; the 10 randomly-sampled targets (paper + new cluster, seed=42) in red.
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "Physical_Core_Speed_Sequences")

# 12 candidate targets (6 paper-cluster + 6 new fast-degraders)
CANDIDATES = [4, 26, 27, 53, 80, 97, 9, 18, 48, 51, 55, 82]
# Sample 10 (seed=42) — matches paper_table.py's TARGETS list
_rng = np.random.default_rng(seed=42)
HIGHLIGHTED = set(int(m) for m in
                   _rng.choice(CANDIDATES, size=10, replace=False))


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "*.csv")),
                    key=lambda f: int(os.path.basename(f).split(".")[0]))
    print(f"Loading {len(files)} trajectories; "
          f"highlighted ({len(HIGHLIGHTED)}): {sorted(HIGHLIGHTED)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Background: all non-highlighted machines
    for f in files:
        m = int(os.path.basename(f).split(".")[0])
        if m in HIGHLIGHTED:
            continue
        y = pd.read_csv(f)["Physical_core_speed_rpm"].values
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="steelblue", alpha=0.18, linewidth=0.7)

    # Foreground: highlighted targets
    for f in files:
        m = int(os.path.basename(f).split(".")[0])
        if m not in HIGHLIGHTED:
            continue
        y = pd.read_csv(f)["Physical_core_speed_rpm"].values
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="crimson", alpha=0.9, linewidth=1.4)

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Physical Core Speed (rpm)", fontsize=12)
    ax.set_title(
        f"Physical core speed trajectories "
        f"(red: target machines for evaluation, n={len(HIGHLIGHTED)})",
        fontsize=11)
    ax.grid(alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], color="steelblue", alpha=0.5, linewidth=1.0,
               label=f"Sources (n={len(files) - len(HIGHLIGHTED)})"),
        Line2D([0], [0], color="crimson", alpha=0.9, linewidth=1.5,
               label=f"Targets (n={len(HIGHLIGHTED)})"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=9)

    plt.tight_layout()
    out_pdf = os.path.join(HERE, "Core_Speed_Highlights.pdf")
    out_png = os.path.join(HERE, "Core_Speed_Highlights.png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=150)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
