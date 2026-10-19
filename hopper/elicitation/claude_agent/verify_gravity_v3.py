"""Final gravity estimator: filter for samples with minimal joint motion.

Theory
------
The torso vertical acceleration equals the COM vertical acceleration only
when the joints are not accelerating relative to torso. During free flight
with rigid joint angles, z_ddot_torso = z_ddot_COM = -g exactly.

Filter: |qd_thigh|, |qd_leg|, |qd_foot| all below a small threshold AND
the hopper is at high z (clearly airborne) AND z_dot[t] also small. The
fewer constraints, the more samples.

We try several filter strictnesses and report all of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008


def estimate_g_minimal_joint_motion(path: Path) -> dict:
    df = pd.read_csv(path)
    z = df["z"].values
    z_dot = df["z_dot"].values
    qd_thigh = df["qd_thigh"].values
    qd_leg = df["qd_leg"].values
    qd_foot = df["qd_foot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    z_ddot_all = np.diff(z_dot) / DT
    valid = (terminal[:-1] == 0) & (episode[:-1] == episode[1:])

    z_ddot = z_ddot_all[valid]
    z_at = z[:-1][valid]
    qd_t = qd_thigh[:-1][valid]
    qd_l = qd_leg[:-1][valid]
    qd_f = qd_foot[:-1][valid]

    # Ensure airborne: z > 90th percentile.
    z_thresh = np.percentile(z_at, 90)
    high_z = z_at > z_thresh

    out = {"file": path.name}
    for tol in (0.5, 0.3, 0.1):
        small_qd = (
            (np.abs(qd_t) < tol) & (np.abs(qd_l) < tol) & (np.abs(qd_f) < tol)
        )
        m = high_z & small_qd
        if m.sum() < 30:
            out[f"g_qd<{tol}"] = None
            out[f"n_qd<{tol}"] = int(m.sum())
            continue
        out[f"g_qd<{tol}"] = float(-np.median(z_ddot[m]))
        out[f"g_qd<{tol}_mean"] = float(-np.mean(z_ddot[m]))
        out[f"n_qd<{tol}"] = int(m.sum())
    return out


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    for f in files:
        r = estimate_g_minimal_joint_motion(f)
        print(
            f"{r['file']}: "
            f"g_qd<0.5={r.get('g_qd<0.5')} (mean={r.get('g_qd<0.5_mean')}, N={r.get('n_qd<0.5')})  "
            f"g_qd<0.3={r.get('g_qd<0.3')} (mean={r.get('g_qd<0.3_mean')}, N={r.get('n_qd<0.3')})  "
            f"g_qd<0.1={r.get('g_qd<0.1')} (mean={r.get('g_qd<0.1_mean')}, N={r.get('n_qd<0.1')})"
        )


if __name__ == "__main__":
    main()
