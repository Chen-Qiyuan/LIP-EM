"""Robust gravity estimation using ballistic-fit method.

For each detected hop, fit a parabola to z(t) during the airborne phase:
  z(t) = z0 + v0*t - 0.5*g*t^2

We identify airborne phases as windows where z is monotonically increasing
then decreasing AND z stays high (above standing baseline) for several
consecutive steps. Then fit g = -2*coefficient on t^2.

This is more direct than finite-differencing z_dot: the integration error
of MuJoCo cancels because we're fitting the trajectory directly.

Also report z statistics per source for sanity.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008


def fit_g_from_hops(path: Path) -> dict:
    df = pd.read_csv(path)
    z = df["z"].values
    z_dot = df["z_dot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    # Detect apex points: z_dot[t] > 0 and z_dot[t+1] <= 0, same episode, z is high.
    z_thresh = np.percentile(z, 75)
    apex_mask = (
        (z_dot[:-1] > 0)
        & (z_dot[1:] <= 0)
        & (terminal[:-1] == 0)
        & (episode[:-1] == episode[1:])
        & (z[:-1] > z_thresh)
    )
    apex_indices = np.where(apex_mask)[0]

    # For each apex, fit a parabola to z over a window of ±W steps.
    # Only use windows where the same episode persists and no terminal occurs.
    W = 5
    g_estimates = []
    for idx in apex_indices:
        lo = idx - W
        hi = idx + W + 1
        if lo < 0 or hi > len(z):
            continue
        ep_window = episode[lo:hi]
        term_window = terminal[lo:hi]
        if not np.all(ep_window == ep_window[0]):
            continue
        if np.any(term_window[:-1] == 1):
            continue
        # Fit z(t) = a*t^2 + b*t + c
        t = np.arange(-W, W + 1) * DT
        zs = z[lo:hi]
        coeffs = np.polyfit(t, zs, 2)
        a = coeffs[0]
        # z(t) = z0 + v0*t - 0.5*g*t^2  =>  a = -0.5*g  =>  g = -2*a
        g_estimates.append(-2.0 * a)

    g_estimates = np.array(g_estimates)
    if len(g_estimates) == 0:
        return {"file": path.name, "g_ballistic": None, "n_hops": 0}

    return {
        "file": path.name,
        "g_ballistic_mean": float(np.mean(g_estimates)),
        "g_ballistic_median": float(np.median(g_estimates)),
        "g_ballistic_std": float(np.std(g_estimates)),
        "n_hops": int(len(g_estimates)),
        "z_max": float(z.max()),
        "z_p99": float(np.percentile(z, 99)),
        "z_median": float(np.median(z)),
    }


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    for f in files:
        r = fit_g_from_hops(f)
        if r["n_hops"] == 0:
            print(f"{r['file']}: no hops detected")
            continue
        print(
            f"{r['file']}: g_ballistic_median={r['g_ballistic_median']:+.3f}  "
            f"g_ballistic_mean={r['g_ballistic_mean']:+.3f}  "
            f"std={r['g_ballistic_std']:.3f}  N={r['n_hops']}  "
            f"z_max={r['z_max']:.2f}  z_p99={r['z_p99']:.2f}  "
            f"z_med={r['z_median']:.2f}"
        )


if __name__ == "__main__":
    main()
