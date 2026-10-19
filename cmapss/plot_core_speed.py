"""Physical-core-speed trajectories with the 10 evaluation targets highlighted
(paper Fig. fig:selected_sources, Core_Speed_Highlights.pdf).

All 100 engines in transparent blue; the 10 target engines in red.

Usage:
    python plot_core_speed.py          # figures/Core_Speed_Highlights.{pdf,png}
"""
from __future__ import annotations

import matplotlib.pyplot as plt
plt.rcParams["pdf.fonttype"] = 42   # embed TrueType, not Type 3 (conference PDF requirement)
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import experiment as E
from dataset import COLUMN


def main():
    files = sorted(E.SEQ_DIR.glob("*.csv"), key=lambda f: int(f.stem))
    highlighted = set(E.TARGETS)
    print(f"Loading {len(files)} trajectories; "
          f"highlighted ({len(highlighted)}): {sorted(highlighted)}")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Background: all non-highlighted engines
    for f in files:
        if int(f.stem) in highlighted:
            continue
        y = pd.read_csv(f)[COLUMN].values
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="steelblue", alpha=0.18, linewidth=0.7)

    # Foreground: highlighted targets
    for f in files:
        if int(f.stem) not in highlighted:
            continue
        y = pd.read_csv(f)[COLUMN].values
        x = np.arange(1, len(y) + 1)
        ax.plot(x, y, color="crimson", alpha=0.9, linewidth=1.4)

    ax.set_xlabel("Cycle", fontsize=12)
    ax.set_ylabel("Physical Core Speed (rpm)", fontsize=12)
    ax.grid(alpha=0.3)

    legend_elems = [
        Line2D([0], [0], color="steelblue", alpha=0.5, linewidth=1.0,
               label=f"Other engines (n={len(files) - len(highlighted)})"),
        Line2D([0], [0], color="crimson", alpha=0.9, linewidth=1.5,
               label=f"Target engines (n={len(highlighted)})"),
    ]
    ax.legend(handles=legend_elems, loc="best", fontsize=9)

    plt.tight_layout()
    E.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = E.FIGURES_DIR / "Core_Speed_Highlights.pdf"
    out_png = E.FIGURES_DIR / "Core_Speed_Highlights.png"
    plt.savefig(out_pdf, metadata={'CreationDate': None})
    plt.savefig(out_png, dpi=150)
    print(f"Saved -> {out_pdf}")
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
