"""Compute gravity from approximate COM trajectory.

The torso vertical acceleration deviates from -g during flight because
joints can swing the legs (so internal forces accelerate the torso). The
COM, however, has z_ddot_COM = -g exactly during flight regardless of
internal motion.

Approximate COM using standard MuJoCo Hopper geometry (link lengths and
masses). The torso, thigh, leg, and foot centers in world coordinates can
be computed from z_torso, pitch, and joint angles. Then take a weighted
sum to get z_COM.

Hopper-v3 link masses (computed from MuJoCo XML defaults, density=1000):
- torso: 3.6 kg (capsule L=0.4, r=0.05)
- thigh: 4.06 kg (capsule L=0.45, r=0.05)
- leg:   2.78 kg (capsule L=0.5, r=0.04)
- foot:  5.31 kg (capsule L=0.39, r=0.06)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DT = 0.008

# Approximate masses for standard MuJoCo Hopper (capsule mass ~ pi*r^2*L*density
# + 2*(2/3)*pi*r^3*density), density = 1000.
M_TORSO = 3.6
M_THIGH = 4.06
M_LEG = 2.78
M_FOOT = 5.31
M_TOTAL = M_TORSO + M_THIGH + M_LEG + M_FOOT

# Link lengths
L_THIGH = 0.45
L_LEG = 0.5
L_FOOT_OFFSET = 0.05  # foot capsule offset along leg
L_THIGH_OFFSET_FROM_TORSO = 0.1  # thigh joint at z_torso - 0.1

# Pivot angles: in MuJoCo Hopper, the pitch and joint angles act about the y-axis.


def compute_z_com(z_torso, pitch, q_thigh, q_leg, q_foot):
    """
    Approximate world z-coordinates of body centers.

    Hopper kinematics (planar, all joints rotate about y-axis):
    - Torso center is at (x_torso, z_torso). Pitch tilts the torso about its
      center. The thigh joint is offset L_THIGH_OFFSET_FROM_TORSO below the
      torso center along the torso's z-axis (which tilts with pitch).
    - Thigh center is offset L_THIGH/2 along the thigh's axis from the thigh
      joint. The thigh's tilt is pitch + q_thigh.
    - Leg joint is at thigh_joint - L_THIGH along thigh axis. Leg center is
      L_LEG/2 along leg axis. Leg tilt is pitch + q_thigh + q_leg.
    - Foot center is at leg_end + L_FOOT_OFFSET along leg axis (approximation,
      foot is small relative to leg).
    """
    # Helper: world delta from a point to another along a body axis.
    # Body axis direction in world for tilt theta: z-component is cos(theta),
    # x-component is sin(theta) (for forward lean = positive pitch, the body
    # tilts forward).
    theta_torso = pitch
    theta_thigh = pitch + q_thigh
    theta_leg = pitch + q_thigh + q_leg
    # foot: same orientation as leg with a small extra offset (we ignore q_foot
    # angular offset for COM estimate; foot is ~along leg).

    # Position of thigh joint (below torso center along torso axis):
    z_thigh_joint = z_torso - L_THIGH_OFFSET_FROM_TORSO * np.cos(theta_torso)

    # Thigh center
    z_thigh_center = z_thigh_joint - 0.5 * L_THIGH * np.cos(theta_thigh)

    # Leg joint (end of thigh)
    z_leg_joint = z_thigh_joint - L_THIGH * np.cos(theta_thigh)

    # Leg center
    z_leg_center = z_leg_joint - 0.5 * L_LEG * np.cos(theta_leg)

    # Foot center (approx: at the end of leg with small offset along leg axis)
    z_foot_center = z_leg_joint - L_LEG * np.cos(theta_leg) - L_FOOT_OFFSET * np.cos(theta_leg)

    z_com = (
        M_TORSO * z_torso
        + M_THIGH * z_thigh_center
        + M_LEG * z_leg_center
        + M_FOOT * z_foot_center
    ) / M_TOTAL
    return z_com


def estimate_g_via_com(path: Path) -> dict:
    df = pd.read_csv(path)
    z_torso = df["z"].values
    pitch = df["pitch"].values
    q_thigh = df["q_thigh"].values
    q_leg = df["q_leg"].values
    q_foot = df["q_foot"].values
    terminal = df["terminal"].values
    episode = df["episode"].values

    z_com = compute_z_com(z_torso, pitch, q_thigh, q_leg, q_foot)

    # Compute z_com_ddot via second-order central differences within each
    # episode. Use a 5-point stencil for less noise:
    # f''(t) ~= (f(t+2) - 2*f(t) + f(t-2)) / (2*dt)^2
    z_ddot = np.full_like(z_com, np.nan)
    for offset, denom in [(2, (2 * DT) ** 2)]:
        z_ddot[offset:-offset] = (
            z_com[2 * offset :] - 2 * z_com[offset:-offset] + z_com[: -2 * offset]
        ) / denom

    # Validity: middle of a non-terminal segment within one episode
    n = len(z_com)
    valid = np.zeros(n, dtype=bool)
    for i in range(2, n - 2):
        ep = episode[i - 2 : i + 3]
        tm = terminal[i - 2 : i + 2]  # termination of step k means step k+1 is new ep
        if np.all(ep == ep[0]) and not np.any(tm == 1):
            valid[i] = True

    valid[np.isnan(z_ddot)] = False
    z_ddot_valid = z_ddot[valid]
    z_com_valid = z_com[valid]
    z_torso_valid = z_torso[valid]

    # Filter for airborne: z_torso above 90th percentile.
    z_thresh = np.percentile(z_torso_valid, 90)
    high_z = z_torso_valid > z_thresh

    g_median = -np.median(z_ddot_valid[high_z])
    g_mean = -np.mean(z_ddot_valid[high_z])

    # Histogram mode
    hist, edges = np.histogram(z_ddot_valid[high_z], bins=400, range=(-30, 30))
    centers = 0.5 * (edges[:-1] + edges[1:])
    g_mode = -centers[np.argmax(hist)]

    return {
        "file": path.name,
        "g_com_median": float(g_median),
        "g_com_mean": float(g_mean),
        "g_com_mode": float(g_mode),
        "n_high_z": int(high_z.sum()),
    }


def main():
    data_dir = Path(__file__).parent / "data"
    files = sorted(data_dir.glob("source_*.csv"))
    for f in files:
        r = estimate_g_via_com(f)
        print(
            f"{r['file']}: g_com_median={r['g_com_median']:+.3f}  "
            f"g_com_mean={r['g_com_mean']:+.3f}  "
            f"g_com_mode={r['g_com_mode']:+.3f}  N={r['n_high_z']}"
        )


if __name__ == "__main__":
    main()
