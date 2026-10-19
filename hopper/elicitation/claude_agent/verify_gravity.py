"""Cross-check gravity estimates using a histogram-mode method.

The apex estimator may be biased by joint-induced internal forces averaging
non-zero at the apex. As an independent check, compute the *mode* of the
z_ddot distribution. During free flight, z_ddot ~= -g for many consecutive
timesteps, producing a sharp peak in the histogram at -g.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008


def estimate_g_mode(path: Path) -> dict:
    df = pd.read_csv(path)
    z = df["z"].values
    z_dot = df["z_dot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    z_ddot = np.diff(z_dot) / DT
    valid = (terminal[:-1] == 0) & (episode[:-1] == episode[1:])
    z_ddot = z_ddot[valid]
    z_at = z[:-1][valid]

    # Restrict to definitely-airborne samples: z above 90th percentile.
    high_z = z_at > np.percentile(z_at, 90)
    z_ddot_high = z_ddot[high_z]

    # Histogram mode in a range covering plausible -g values.
    hist, bin_edges = np.histogram(z_ddot_high, bins=400, range=(-30, 30))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    mode_idx = np.argmax(hist)
    g_mode = -centers[mode_idx]

    # Robust mean over a tight window around the mode (within ±0.5 m/s²).
    near_mode = np.abs(z_ddot_high - centers[mode_idx]) < 0.5
    g_local_mean = -np.mean(z_ddot_high[near_mode]) if near_mode.sum() > 0 else np.nan

    return {
        "file": path.name,
        "g_mode": float(g_mode),
        "g_local_mean": float(g_local_mean),
        "n_high_z": int(high_z.sum()),
        "n_near_mode": int(near_mode.sum()),
    }


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    results = []
    for f in files:
        r = estimate_g_mode(f)
        results.append(r)
        print(
            f"{r['file']}: g_mode={r['g_mode']:+.3f}  "
            f"g_local_mean={r['g_local_mean']:+.3f}  "
            f"n_high_z={r['n_high_z']}  n_near_mode={r['n_near_mode']}"
        )


if __name__ == "__main__":
    main()
