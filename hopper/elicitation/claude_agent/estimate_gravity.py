"""Estimate gravity from each source's telemetry.

Strategy
--------
During free flight (foot off ground), the torso's vertical acceleration is
dominated by gravity: z_ddot ~= -g. Internal forces from joint torques can
perturb the torso's vertical acceleration (the COM's z_ddot is exactly -g,
but the torso is not the COM since the limbs can swing). However:

  * Over many samples, the average effect of internal forces averages out.
  * The mean of z_ddot during flight steps is very close to -g.

We identify "flight-like" steps using a simple proxy: at the apex of each
hop, z_dot crosses zero. Steps where |z_dot| is small AND z is high are
unambiguously flight steps.

We also report a lower-percentile estimate, since z_ddot >= -g during contact
implies the lower tail of the z_ddot distribution lies near -g.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008


def estimate_gravity_for_file(path: Path) -> dict:
    df = pd.read_csv(path)
    z = df["z"].values
    z_dot = df["z_dot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    # z_ddot[t] from finite differences of z_dot
    z_ddot_all = np.diff(z_dot) / DT
    valid = (terminal[:-1] == 0) & (episode[:-1] == episode[1:])
    z_ddot = z_ddot_all[valid]
    z_at = z[:-1][valid]
    z_dot_at = z_dot[:-1][valid]

    # Apex-of-hop estimator: z_dot near zero and z above a high threshold
    # (well above any standing height to ensure the foot is off ground).
    high_z_thresh = np.percentile(z_at, 90)
    apex_mask = (np.abs(z_dot_at) < 0.2) & (z_at > high_z_thresh)
    g_apex = -np.mean(z_ddot[apex_mask]) if apex_mask.sum() > 0 else np.nan

    # Lower-percentile estimator: z_ddot >= -g, so the 1st percentile is ~ -g.
    g_pct1 = -np.percentile(z_ddot, 1)
    g_pct5 = -np.percentile(z_ddot, 5)

    # Median of z_ddot among "high z" steps (foot likely off ground).
    high_mask = z_at > high_z_thresh
    g_high_median = -np.median(z_ddot[high_mask])

    # Aggregate: trust apex when we have enough samples; fall back to
    # high-z median otherwise.
    if apex_mask.sum() >= 200:
        g_best = g_apex
        method = f"apex(N={apex_mask.sum()})"
    else:
        g_best = g_high_median
        method = f"high_z_median(N={high_mask.sum()})"

    return {
        "file": path.name,
        "g_apex": float(g_apex) if not np.isnan(g_apex) else None,
        "n_apex": int(apex_mask.sum()),
        "g_pct1": float(g_pct1),
        "g_pct5": float(g_pct5),
        "g_high_median": float(g_high_median),
        "g_best": float(g_best),
        "method": method,
        "high_z_thresh": float(high_z_thresh),
        "n_total_valid": int(valid.sum()),
    }


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    print(f"Found {len(files)} source files\n")
    results = []
    for f in files:
        r = estimate_gravity_for_file(f)
        results.append(r)
        print(
            f"{r['file']}: g_best={r['g_best']:.3f}  "
            f"g_apex={r['g_apex']}  "
            f"g_pct1={r['g_pct1']:.3f}  g_pct5={r['g_pct5']:.3f}  "
            f"g_high_med={r['g_high_median']:.3f}  "
            f"method={r['method']}  high_z={r['high_z_thresh']:.3f}"
        )
    return results


if __name__ == "__main__":
    main()
