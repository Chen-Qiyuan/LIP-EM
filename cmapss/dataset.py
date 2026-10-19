"""Data loading and spline basis for the C-MAPSS case study.

One CSV per engine, single column `Physical_core_speed_rpm` (sensor 9 of
C-MAPSS FD001; see `prepare_data.py`). Cycle indices `x` are 1, 2, ..., N
(one per row).

The basis is a **natural cubic regression spline** with K=5 equally spaced
interior knots in the fixed raw-cycle window [0, X_MAX]. Natural cubic means
linear beyond the boundary knots, which gives stable linear extrapolation.
K=5 with X_MAX=300 puts the knots at [50, 100, 150, 200, 250].

Truncated-power form: the basis is

    [1, x,  d_0(x) - d_{K-2}(x),  d_1(x) - d_{K-2}(x),  d_2(x) - d_{K-2}(x)]

where d_k(x) = ((x - xi_k)_+^3 - (x - xi_{K-1})_+^3) / (xi_{K-1} - xi_k).
Total dimension = K = 5.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# Basis hyperparameters (paper App. C-MAPSS).
N_KNOTS = 5
X_MAX = 300.0

COLUMN = "Physical_core_speed_rpm"


def basis(x: np.ndarray) -> np.ndarray:
    """Natural cubic regression spline design matrix at points `x`.

    Returns shape `(N, K)` where K = `N_KNOTS` (5).
    """
    knots = np.linspace(0.0, X_MAX, N_KNOTS + 2)[1:-1]
    xi_max = knots[-1]

    def d(k: int, xv: np.ndarray) -> np.ndarray:
        denom = max(xi_max - knots[k], 1e-12)
        return (np.maximum(xv - knots[k], 0) ** 3
                - np.maximum(xv - xi_max, 0) ** 3) / denom

    x_arr = np.asarray(x, dtype=float)
    cols = [np.ones_like(x_arr), x_arr]
    d_last = d(N_KNOTS - 2, x_arr)
    for k in range(N_KNOTS - 2):
        cols.append(d(k, x_arr) - d_last)
    return np.column_stack(cols)


def load_machine(filepath,
                 cutoff: int | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single-engine CSV and build its basis design matrix.

    Returns `(x_raw, X, y)`:
      * `x_raw` — cycle indices [1, ..., N], truncated to `cutoff` if given.
      * `X` — `basis(x_raw)`, shape (N, K).
      * `y` — physical core speed values, same length as `x_raw`.
    """
    y = pd.read_csv(filepath)[COLUMN].values
    x_raw = np.arange(1, len(y) + 1, dtype=float)
    if cutoff:
        x_raw = x_raw[:cutoff]
        y = y[:cutoff]
    return x_raw, basis(x_raw), y


def save_as_clean_text(y_data: np.ndarray, output_path) -> None:
    """Dump (cycle, y) pairs as plain text, the format the LLM ingests."""
    with open(output_path, "w") as f:
        for i, val in enumerate(y_data):
            f.write(f"({i + 1}, {val:.2f})\n")
