"""Final precision check: look at the cleanest possible estimate per source.

Strategy: find LONG continuous flight phases (>= 20 timesteps where z is high),
fit a parabola to z(t) for each, keep only fits with low residual error,
report median g.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008


def long_flight_g(path: Path, min_len: int = 20) -> dict:
    df = pd.read_csv(path)
    z = df["z"].values
    z_dot = df["z_dot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    # Define airborne: z above its 80th percentile (well above standing).
    z_thresh = np.percentile(z, 80)
    airborne = z > z_thresh

    # Find runs of consecutive airborne timesteps within an episode and not
    # crossing terminal.
    g_list = []
    rss_list = []  # residual sum of squares for each fit
    n = len(z)
    i = 0
    while i < n:
        if not airborne[i]:
            i += 1
            continue
        # Start of a run
        j = i
        while (
            j + 1 < n
            and airborne[j + 1]
            and episode[j + 1] == episode[i]
            and terminal[j] == 0
        ):
            j += 1
        # Run is [i, j]
        run_len = j - i + 1
        if run_len >= min_len:
            t = np.arange(run_len) * DT
            zs = z[i : j + 1]
            coeffs, residuals, *_ = np.polyfit(t, zs, 2, full=True)
            a = coeffs[0]
            g = -2.0 * a
            rss = float(residuals[0]) if len(residuals) > 0 else 0.0
            # Only keep if fit is reasonable (small residual per sample)
            rmse = np.sqrt(rss / run_len)
            if rmse < 0.05:  # 5 cm RMSE — generous threshold
                g_list.append(g)
                rss_list.append(rmse)
        i = j + 1

    if not g_list:
        return {"file": path.name, "g_long_flight": None, "n_runs": 0}

    g_arr = np.array(g_list)
    return {
        "file": path.name,
        "g_long_flight_median": float(np.median(g_arr)),
        "g_long_flight_mean": float(np.mean(g_arr)),
        "g_long_flight_std": float(np.std(g_arr)),
        "g_long_flight_p25": float(np.percentile(g_arr, 25)),
        "g_long_flight_p75": float(np.percentile(g_arr, 75)),
        "n_runs": int(len(g_arr)),
    }


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    for f in files:
        r = long_flight_g(f)
        if r["n_runs"] == 0:
            print(f"{r['file']}: no long flights detected")
            continue
        print(
            f"{r['file']}: median={r['g_long_flight_median']:+.3f}  "
            f"mean={r['g_long_flight_mean']:+.3f}  "
            f"std={r['g_long_flight_std']:.3f}  "
            f"p25={r['g_long_flight_p25']:+.3f}  "
            f"p75={r['g_long_flight_p75']:+.3f}  "
            f"N_runs={r['n_runs']}"
        )


if __name__ == "__main__":
    main()
